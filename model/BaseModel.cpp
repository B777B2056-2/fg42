//
// Created by 19373 on 2025/9/4.
//
#include <filesystem>
#include <fstream>
#include "model/BaseModel.h"

namespace fg42 {
    namespace fs = std::filesystem;

    bool ends_with(const std::string& str, const std::string &suffix) {
        if (suffix.length() > str.length()) { return false; }
        return (str.rfind(suffix) == (str.length() - suffix.length()));
    }

    class BaseModelWeightsLoader {
    public:
        virtual ~BaseModelWeightsLoader() = default;
        virtual void load(const std::filesystem::path& path, StateDict& state_dict, DeviceType device_type) = 0;
    };

    class SafetensorsWeightsLoader final : public BaseModelWeightsLoader {
    public:
        ~SafetensorsWeightsLoader() override = default;

        // 加载safetensors格式的权重到内存(https://huggingface.co/docs/safetensors/index)
        void load(const std::filesystem::path& dir_path, StateDict& state_dict, DeviceType device_type) override {
            // 寻找所有safetensors文件
            std::vector<std::string> safetensors_file_names;
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                if (ends_with(entry.path().string(), ".safetensors")) {
                    safetensors_file_names.push_back(entry.path().string());
                }
            }
            if (safetensors_file_names.empty()) {
                throw std::runtime_error("No safetensors files found");
            }

            // 如果只有一个safetensors文件，直接读取
            if (safetensors_file_names.size() == 1) {
                const auto weights_file_path = dir_path / safetensors_file_names[0];
                std::string a = weights_file_path.string();
                SafetensorsWeightsLoader::read_one_safetensors(weights_file_path, state_dict, device_type);
                return;
            }

            // 如果有多个，循环读取（不并发的原因：读文件是存储密集型任务，实际速率受制于硬件）
            for (const auto& file_name : safetensors_file_names) {
                const fs::path weights_file_path = dir_path / file_name;
                SafetensorsWeightsLoader::read_one_safetensors(weights_file_path, state_dict, device_type);
            }
        }
    private:
        static nlohmann::ordered_json read_header(std::ifstream& ifs) {
            // 读取文件头长度
            std::uint64_t header_length = 0;
            ifs.read(reinterpret_cast<char*>(&header_length), sizeof(std::uint64_t));
            // 读取文件头内容
            std::string header_json(header_length, '0');
            ifs.read(header_json.data(), static_cast<std::streamsize>(header_length));
            // 解析json
            return nlohmann::ordered_json::parse(header_json);
        }

        static void read_data(std::ifstream& ifs, const nlohmann::ordered_json& header,
            StateDict& state_dict, DeviceType device_type) {
            // 清空原数据
            state_dict.clear();
            // 计算有多少个Tensor，提前为map分配内存（-1是因为不需要__metadata__内的东西）
            state_dict.reserve(header.size() - 1);
            // 解析文件
            for (auto& x : header.items()) {
                if (x.key() == "__metadata__") {
                    continue;
                }

                // 获取张量名称
                std::string tensor_name = x.key();
                // 获取张量数据类型
                auto tensor_d_type = x.value()["dtype"].get<std::string>();
                // 获取张量形状
                auto tensor_shape = x.value()["shape"].get<std::vector<std::size_t>>();
                // 构造内存Tensor
                DataType data_type = DataType::Unknown;
                if (tensor_d_type == "F32") {
                    data_type = DataType::FP32;
                } else if (tensor_d_type == "BF16") {
                    data_type = DataType::BF16;
                } else if (tensor_d_type == "I8") {
                    data_type = DataType::Int8;
                } else if (tensor_d_type == "U8") {
                    data_type = DataType::UInt8;
                }
                Tensor tensor(data_type, device_type, tensor_shape);
                // 获取张量在权重文件的范围
                auto tensor_offset_range = x.value()["data_offsets"].get<std::vector<int>>();
                // 校验张量字节长度
                if (tensor.bytes_size() != (tensor_offset_range[1] - tensor_offset_range[0])) {
                    throw std::runtime_error("Fail on parse tensor offsets");
                }
                // 读取张量二进制数据
                ifs.seekg(tensor_offset_range[0], std::ios::beg);
                ifs.read(static_cast<char*>(tensor.raw_ptr()), static_cast<std::streamsize>(tensor.bytes_size()));
                // 存入权重map
                state_dict.emplace(tensor_name, tensor);
            }
        }

        static void read_one_safetensors(const fs::path& weights_file_path, StateDict& state_dict, DeviceType device_type) {
            // 打开文件
            std::ifstream ifs(weights_file_path, std::ios::in | std::ios::binary);
            if (!ifs) {
                throw std::runtime_error("Could not open file " + weights_file_path.string());
            }
            // 读取文件头
            auto header = SafetensorsWeightsLoader::read_header(ifs);
            // 读取数据部分
            SafetensorsWeightsLoader::read_data(ifs, header, state_dict, device_type);
        }
    };

    static BaseModelWeightsLoader* model_weights_loader_factory(const std::filesystem::path& dir_path) {
        static SafetensorsWeightsLoader safetensors_weights_loader;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (ends_with(entry.path().string(), ".safetensors")) {
                return &safetensors_weights_loader;
            }
        }
        return nullptr;
    }

    BaseModel::BaseModel(const std::string& dir_path, DeviceType device_type) : device_type_(device_type) {
        this->load_model_config(dir_path);
        this->load_model_weights(dir_path);
    }

    void BaseModel::load_model_config(const std::string& dir_path) {
        fs::path dir = dir_path;
        fs::path config_path = dir / "config.json";
        auto config_file_path = dir / config_path;

        std::ifstream config;
        config.open(config_file_path, std::ios::in);

        if (!config) {
            throw std::runtime_error("Could not open config file " + config_file_path.string());
        }
        this->model_config_ = nlohmann::json::parse(config);
    }

    void BaseModel::load_model_weights(const std::string& path) {
        auto* loader = model_weights_loader_factory(path);
        if (loader == nullptr) {
            throw std::runtime_error("Could not load model weights from dir: " + path);
        }
        loader->load(path, this->state_dict_, this->device_type_);
    }
} // fg42