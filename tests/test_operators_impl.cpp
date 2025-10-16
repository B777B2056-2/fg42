//
// Created by B777B2056-2 on 2025/9/30.
//
#include "tests/test_operators_impl.h"
#include <fstream>
#include "memory/Common.h"
#include "model/BaseModel.h"
#include "model/layers/Qwen2Layers.h"
#include "operator/ArithmeticOperator.h"
#include "operator/ActivationOperator.h"
#include "operator/Attention.h"
#include "operator/EmbeddingOperator.h"
#include "operator/NormOperator.h"
#include "Eigen/Core"
#ifdef HAVE_CUDA
#include <cuda_bf16.h>
#include "memory/NvidiaGPUMemoryAllocator.h"
#endif

#define INF_FLOAT 9999

bool is_float_equal(float a, float b, float abs=4e-2f) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    if (std::isinf(a) && std::isinf(b)) {
        return a == b;
    }

    if (std::abs(a-b) < abs) {
        return true;
    }
    return false;
}

bool is_tensor_equal(const fg42::Tensor& a, const fg42::Tensor& b) {
    if (a.data_type() != b.data_type()) {
        return false;
    }

    if (a.device_type() != b.device_type()) {
        return false;
    }

    auto a_size = a.size();
    auto b_size = b.size();
    if (a_size != b_size) {
        return false;
    }

    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    if (a_shape != b_shape) {
        return false;
    }

    for (std::size_t i = 0; i < a_size; ++i) {
        void* a_ptr = static_cast<char*>(a.raw_ptr()) + i * fg42::data_type_size(a.data_type());
        void* b_ptr = static_cast<char*>(b.raw_ptr()) + i * fg42::data_type_size(b.data_type());
        switch (a.data_type()) {
            case fg42::DataType::FP32:
                if (!is_float_equal(*static_cast<float*>(a_ptr), *static_cast<float*>(b_ptr))) {
                    return false;
                }
                break;
            case fg42::DataType::BF16: {
                float a_f = fg42::util::bfloat16_to_float(a.device_type(), a_ptr);
                float b_f = fg42::util::bfloat16_to_float(b.device_type(), b_ptr);
                if (!is_float_equal(a_f, b_f)) {
                    return false;
                }
            }
                break;
            case fg42::DataType::Int32:
                if (*static_cast<std::int32_t*>(a_ptr) != *static_cast<std::int32_t*>(b_ptr)) {
                    return false;
                }
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
    }
    return true;
}

fg42::DataType get_data_type_enum(const std::string& data_type) {
    fg42::DataType d_type = fg42::DataType::Unknown;
    if (data_type == "fp32") {
        d_type = fg42::DataType::FP32;
    } else if (data_type == "bf16") {
        d_type = fg42::DataType::BF16;
    } else if (data_type == "int32") {
        d_type = fg42::DataType::Int32;
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    return d_type;
}

fg42::Tensor make_tensor_from_json(const nlohmann::json& json,
    const std::string& data_type, fg42::DeviceType device_type) {
    auto shape = json["shape"].get<std::vector<std::size_t>>();
    fg42::DataType d_type = get_data_type_enum(data_type);

    fg42::Tensor tensor(d_type, device_type, shape);

    switch (d_type) {
        case fg42::DataType::FP32: {
            auto data = json["data"].get<std::vector<float>>();
            for (std::size_t i = 0; i < data.size(); ++i) {
                if (is_float_equal(data[i], -INF_FLOAT)) {
                    data[i] = -std::numeric_limits<float>::infinity();
                }
            }
            auto bytes = data.size() * fg42::data_type_size(d_type);
            fg42::PtrDeviceWrapper dst(device_type, tensor.raw_ptr());
            fg42::PtrDeviceWrapper src(fg42::DeviceType::CPU, data.data());
            fg42::memcpy_between_device(dst, src, bytes);
            break;
        }
        case fg42::DataType::BF16: {
            auto tmp = json["data"].get<std::vector<float>>();
            for (std::size_t i = 0; i < tmp.size(); ++i) {
                if (is_float_equal(tmp[i], -INF_FLOAT)) {
                    tmp[i] = -std::numeric_limits<float>::infinity();
                }
            }
            if (device_type == fg42::DeviceType::CPU) {
                std::vector<Eigen::bfloat16> data;
                for (const auto& f : tmp) {
                    data.emplace_back(f);
                }
                auto bytes = tensor.size() * fg42::data_type_size(d_type);
                fg42::PtrDeviceWrapper dst(device_type, tensor.raw_ptr());
                fg42::PtrDeviceWrapper src(fg42::DeviceType::CPU, data.data());
                fg42::memcpy_between_device(dst, src, bytes);
            }
#ifdef HAVE_CUDA
            else if (device_type == fg42::DeviceType::NvidiaGPU) {
                std::vector<__nv_bfloat16> data;
                for (const auto& f : tmp) {
                    data.push_back(__float2bfloat16(f));
                }
                auto bytes = tensor.size() * fg42::data_type_size(d_type);
                fg42::PtrDeviceWrapper dst(device_type, tensor.raw_ptr());
                fg42::PtrDeviceWrapper src(fg42::DeviceType::CPU, data.data());
                fg42::memcpy_between_device(dst, src, bytes);
            }
#endif
            break;
        }
        case fg42::DataType::Int32: {
            auto data = json["data"].get<std::vector<std::int32_t>>();
            auto bytes = data.size() * fg42::data_type_size(d_type);
            fg42::PtrDeviceWrapper dst(device_type, tensor.raw_ptr());
            fg42::PtrDeviceWrapper src(fg42::DeviceType::CPU, data.data());
            fg42::memcpy_between_device(dst, src, bytes);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type");
    }
    return tensor;
}

OperatorTester::OperatorTester(const std::string& global_data_type,
    fg42::DeviceType device_type, const nlohmann::json& config)
    : device_type_(device_type), op_(nullptr) {
    name_ = config["name"].get<std::string>();

    if (config.contains("inputs")) {
        for (const auto& input_json : config["inputs"]) {
            add_input(global_data_type, input_json);
        }
    }

    if (config.contains("weights")) {
        for (const auto& weight_json : config["weights"]) {
            add_weight(global_data_type, weight_json);
        }
    }

    if (config.contains("biases")) {
        for (const auto& bias_json : config["biases"]) {
            add_bias(global_data_type, bias_json);
        }
    }

    if (config.contains("expected_output")) {
        set_expected_output(global_data_type, config["expected_output"]);
    }

    this->set_operator(config["operator"]);
}

std::string OperatorTester::error_message() const {
    std::stringstream ss;
    ss << " test failed on " << name_ << std::endl;
    return ss.str();
}

bool OperatorTester::test() {
    std::vector<const fg42::Tensor*> op_inputs;
    for (const auto& [name, tensor] : inputs_) {
        op_inputs.push_back(&tensor);
    }
    auto output = op_->forward(op_inputs, nullptr);
    output.to_device(fg42::DeviceType::CPU);

    bool is_correct = is_tensor_equal(output, expected_output_);
    return is_correct;
}

void OperatorTester::set_operator(const nlohmann::json& op_json) {
    auto operator_name = op_json["name"].get<std::string>();

    if (operator_name == "add") {
        op_ = std::make_unique<fg42::kernel::AddOperator>();
    } else if (operator_name == "neg") {
        op_ = std::make_unique<fg42::kernel::NegateOperator>();
    } else if (operator_name == "outer") {
        op_ = std::make_unique<fg42::kernel::VecOuterOperator>();
    } else if (operator_name == "mul") {
        op_ = std::make_unique<fg42::kernel::MulOperator>();
    } else if (operator_name == "bmm") {
        op_ = std::make_unique<fg42::kernel::MatmulOperator>();
    } else if (operator_name == "transpose") {
        auto dim0 = op_json["dim_0"].get<std::size_t>();
        auto dim1 = op_json["dim_1"].get<std::size_t>();
        op_ = std::make_unique<fg42::kernel::TransposeOperator>(dim0, dim1);
    } else if (operator_name == "argmax") {
        auto num_samples = op_json["num_samples"].get<std::size_t>();
        op_ = std::make_unique<fg42::kernel::VecOrMatrixArgmaxOperator>(num_samples);
    } else if (operator_name == "rotate_half") {
        op_ = std::make_unique<fg42::kernel::RotateHalfOperator>();
    } else if (operator_name == "cosine") {
        op_ = std::make_unique<fg42::kernel::CosineOperator>();
    } else if (operator_name == "sine") {
        op_ = std::make_unique<fg42::kernel::SineOperator>();
    } else if (operator_name == "concat_by_row") {
        op_ = std::make_unique<fg42::kernel::ConcatByRowWiseOperator>();
    } else if (operator_name == "concat_by_col") {
        op_ = std::make_unique<fg42::kernel::ConcatByColWiseOperator>();
    } else if (operator_name == "causal_mask") {
        auto data_type = op_json["data_type"].get<std::string>();
        auto data_type_enum = get_data_type_enum(data_type);
        auto num_rows = op_json["num_rows"].get<std::size_t>();
        auto num_cols = op_json["num_cols"].get<std::size_t>();
        op_ = std::make_unique<fg42::kernel::CausalMaskOperator>(
            data_type_enum, device_type_, num_rows, num_cols);
    } else if (operator_name == "softmax") {
        op_ = std::make_unique<fg42::kernel::SoftmaxActivationOperator>();
    } else if (operator_name == "silu") {
        op_ = std::make_unique<fg42::kernel::SiLUActivationOperator>();
    } else if (operator_name == "rms_norm") {
        auto eps = op_json["eps"].get<float>();
        op_ = std::make_unique<fg42::kernel::RMSNormOperator>(eps);
    } else if (operator_name == "embedding") {
        const fg42::Tensor& weight = get_weight_by_name("embedding_table");
        op_ = std::make_unique<fg42::kernel::EmbeddingOperator>(weight);
    } else if (operator_name == "sdpa") {
        op_ = std::make_unique<fg42::kernel::ScaledDotProductAttention>();
    } else if (operator_name == "multi-query-attention") {
        auto num_heads = op_json["num_heads"].get<std::size_t>();
        auto num_kv_heads = op_json["num_kv_heads"].get<std::size_t>();

        const auto& q_proj_weights = get_weight_by_name("q_proj");
        const auto& k_proj_weights = get_weight_by_name("k_proj");
        const auto& v_proj_weights = get_weight_by_name("v_proj");
        const auto& o_proj_weights = get_weight_by_name("o_proj");

        const auto& q_proj_bias = get_bias_by_name("q_proj");
        const auto& k_proj_bias = get_bias_by_name("k_proj");
        const auto& v_proj_bias = get_bias_by_name("v_proj");

        auto mqa = std::make_unique<fg42::kernel::MultiQueryAttention>(
            num_heads, num_kv_heads, 0, nullptr,
            &q_proj_weights, &k_proj_weights, &v_proj_weights, &o_proj_weights,
            &q_proj_bias, &k_proj_bias, &v_proj_bias, nullptr);
        if (op_json.contains("max_position_embeddings") && op_json.contains("rope_theta")) {
            fg42::DataType data_type_enum = q_proj_weights.data_type();
            auto device_type = q_proj_weights.device_type();
            std::size_t hidden_size = q_proj_weights.shape().back();
            auto max_position_embeddings = op_json["max_position_embeddings"].get<std::size_t>();
            auto rope_theta = op_json["rope_theta"].get<float>();

            std::size_t seq_len = op_json["seq_len"].get<std::size_t>();

            fg42::ModelConfig model_config{};
            model_config.num_attention_heads = num_heads;
            model_config.hidden_size = hidden_size;
            model_config.max_position_embeddings = max_position_embeddings;
            model_config.rope_theta = rope_theta;

            auto rope_impl = std::make_shared<fg42::Qwen2RotaryEmbeddingLayer>(data_type_enum,
                device_type, model_config);

            mqa->set_rope_apply_func([rope_impl, seq_len](
                const fg42::Tensor& q, const fg42::Tensor& k) -> std::tuple<fg42::Tensor, fg42::Tensor> {
                return rope_impl->apply_rotary_pos_emb(q, k, seq_len);
            }
        );
        }
        op_ = std::move(mqa);
    }
}

void OperatorTester::add_input(const std::string& global_data_type, const nlohmann::json& input_json) {
    std::string data_type = global_data_type;
    if (input_json.contains("data_type") && input_json["data_type"].get<std::string>() != "auto") {
        data_type = input_json["data_type"].get<std::string>();
    }

    auto tensor = make_tensor_from_json(input_json, data_type, device_type_);
    inputs_.emplace_back(input_json["name"].get<std::string>(), tensor);
}

void OperatorTester::add_weight(const std::string& global_data_type, const nlohmann::json& weight_json) {
    std::string data_type = global_data_type;
    if (weight_json.contains("data_type") && weight_json["data_type"].get<std::string>() != "auto") {
        data_type = weight_json["data_type"].get<std::string>();
    }

    auto tensor = make_tensor_from_json(weight_json, data_type, device_type_);
    weights_.emplace_back(weight_json["name"].get<std::string>(), tensor);
}

void OperatorTester::add_bias(const std::string& global_data_type, const nlohmann::json& bias_json) {
    std::string data_type = global_data_type;
    if (bias_json.contains("data_type") && bias_json["data_type"].get<std::string>() != "auto") {
        data_type = bias_json["data_type"].get<std::string>();
    }

    auto tensor = make_tensor_from_json(bias_json, data_type, device_type_);
    biases_.emplace_back(bias_json["name"].get<std::string>(), tensor);
}

void OperatorTester::set_expected_output(const std::string& global_data_type, const nlohmann::json& expected_output_json) {
    std::string data_type = global_data_type;
    if (expected_output_json.contains("data_type") && expected_output_json["data_type"].get<std::string>() != "auto") {
        data_type = expected_output_json["data_type"].get<std::string>();
    }

    expected_output_ = make_tensor_from_json(expected_output_json, data_type, device_type_);
    expected_output_.to_device(fg42::DeviceType::CPU);
}

const fg42::Tensor& OperatorTester::get_input_by_name(const std::string& name) const {
    for (const auto& input : inputs_) {
        if (input.first == name) {
            return input.second;
        }
    }
    throw std::runtime_error("Input not found: " + name);
}

const fg42::Tensor & OperatorTester::get_weight_by_name(const std::string& name) const {
    for (const auto& weight : weights_) {
        if (weight.first == name) {
            return weight.second;
        }
    }
    throw std::runtime_error("Weight not found: " + name);
}

const fg42::Tensor & OperatorTester::get_bias_by_name(const std::string& name) const {
    for (const auto& bias : biases_) {
        if (bias.first == name) {
            return bias.second;
        }
    }
    throw std::runtime_error("Bias not found: " + name);
}

OperatorTesterManager::OperatorTesterManager(
    const std::filesystem::path& test_cases_path, fg42::DeviceType device_type) {
    // 1. 读取测试用例json文件
    std::ifstream file(test_cases_path / "operators.json");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test cases file.");
    }
    nlohmann::ordered_json test_cases_json;
    file >> test_cases_json;

    // 2. 初始化各个测试器
    auto test_cases = test_cases_json["test_suite"]["test_cases"];
    for (const auto& cases : test_cases) {
        auto tester_name = cases["name"].get<std::string>();
        // 2.1 获取全局数据类型（如果有）
        if (cases.contains("data_types")) {
            auto global_data_types = cases["data_types"].get<std::vector<std::string>>();
            // 2.1.1 创建测试器
            for (const auto& global_data_type : global_data_types) {
                testers_map_[tester_name].emplace_back(std::make_unique<OperatorTester>(global_data_type, device_type, cases));
            }
        } else {
            testers_map_[tester_name].emplace_back(std::make_unique<OperatorTester>("", device_type, cases));
        }
    }
}

testing::AssertionResult OperatorTesterManager::run_test(const std::string &tester_name) {
    if (testers_map_.find(tester_name) == testers_map_.end()) {
        return testing::AssertionFailure() << "Tester " << tester_name << " not found";
    }

    for (const auto& tester : testers_map_[tester_name]) {
        if (!tester->test()) {
            return testing::AssertionFailure()
                << "Tester " << tester_name << " failed:  " << tester->error_message();
        }
    }
    return testing::AssertionSuccess();
}
