//
// Created by B777B2056-2 on 2025/9/4.
//
#include <filesystem>
#include <fstream>
#include "model/BaseModel.h"
#include "model/weights_loaders/Factory.h"

namespace fg42 {
    namespace fs = std::filesystem;

    BaseModel::BaseModel(const std::string& dir_path, DeviceType device_type,
        std::int32_t padding_idx, DataType data_type, KVCacheImpl kv_cache_impl)
    : device_type_(device_type), padding_idx_(padding_idx), kv_cache_impl_(kv_cache_impl) {
        this->load_model_config(dir_path, data_type);
    }

    void BaseModel::load_model_config(const std::string& dir_path, DataType data_type) {
        fs::path dir = dir_path;
        fs::path config_path = dir / "config.json";
        auto config_file_path = dir / config_path;

        std::ifstream config;
        config.open(config_file_path, std::ios::in);

        if (!config) {
            throw std::runtime_error("Could not open config file " + config_file_path.string());
        }
        auto raw_model_config_ = nlohmann::json::parse(config);

        auto architectures = raw_model_config_["architectures"].get<std::vector<std::string>>();
        if (architectures.empty()) {
            throw std::runtime_error("Could not find model architecture in config file.");
        }
        if (architectures.size() != 1) {
            throw std::runtime_error("Unsupported multi model architecture.");
        }

        this->model_config_.architecture = architectures[0];
        this->model_config_.bos_token_id = raw_model_config_["bos_token_id"].get<std::size_t>();
        this->model_config_.eos_token_id = raw_model_config_["eos_token_id"].get<std::size_t>();
        this->model_config_.num_attention_heads = raw_model_config_["num_attention_heads"].get<std::size_t>();
        this->model_config_.num_key_value_heads = raw_model_config_["num_key_value_heads"].get<std::size_t>();
        this->model_config_.hidden_size = raw_model_config_["hidden_size"].get<std::size_t>();
        this->model_config_.num_hidden_layers = raw_model_config_["num_hidden_layers"].get<std::size_t>();
        this->model_config_.vocab_size = raw_model_config_["vocab_size"].get<std::size_t>();
        this->model_config_.max_position_embeddings = raw_model_config_["max_position_embeddings"].get<std::size_t>();
        this->model_config_.rms_norm_eps = raw_model_config_["rms_norm_eps"].get<float>();
        this->model_config_.rope_theta = raw_model_config_["rope_theta"].get<float>();
        this->model_config_.use_cache = raw_model_config_["use_cache"].get<bool>();

        if (data_type == DataType::Unknown) {
            auto torch_dtype = raw_model_config_["torch_dtype"].get<std::string>();
            if (torch_dtype == "bfloat16") {
                model_config_.data_type = DataType::BF16;
            } else if (torch_dtype == "float32") {
                model_config_.data_type = DataType::FP32;
            } else if (torch_dtype == "int8") {
                model_config_.data_type = DataType::Int8;
            }  else if (torch_dtype == "uint8") {
                model_config_.data_type = DataType::UInt8;
            } else if (torch_dtype == "int32") {
                model_config_.data_type = DataType::Int32;
            } else {
                throw std::runtime_error("Unsupported torch_dtype: " + torch_dtype);
            }
        } else {
            model_config_.data_type = data_type;
        }

        // 按需初始化kv cache实例
        if (this->model_config_.use_cache) {
            this->kv_cache_ = kv_cache_factory(kv_cache_impl_, device_type_, model_config_);
        }
    }

    void BaseModel::load_model_weights(const std::string& path, DataType data_type) {
        auto* loader = model_weights_loader_factory(path);
        if (loader == nullptr) {
            throw std::runtime_error("Could not load model weights from dir: " + path);
        }
        loader->set_weight_need_transpose_func([this](const std::string& key)->bool {
            return this->weight_need_transpose(key);
        });
        loader->load(path, this->state_dict_, this->device_type_, data_type);
    }

    std::vector<std::vector<std::int32_t>> BaseModel::generate(SamplerConfig sampler_config,
                                                               const std::vector<std::vector<std::int32_t>>& input_ids,
                                                               std::size_t max_length, StreamHandler stream_handler) {
        // 1. left padding
        auto after_padding = input_ids;
        this->left_padding(after_padding);

        // 2. 转为Tensor
        Tensor input_ids_batch(DataType::Int32, DeviceType::CPU,
        {after_padding.size(), after_padding[0].size()});
        for (std::size_t i = 0; i < after_padding.size(); ++i) {
            for (std::size_t j = 0; j < after_padding[i].size(); ++j) {
                input_ids_batch.index_fill({i, j}, static_cast<float>(after_padding[i][j]));
            }
        }

        // 3. 初始化各batch的生成状态
        this->init_token_generate_status(input_ids_batch.shape().at(0));

        // 4. 生成
        // prefill
        auto [prefilled, seq_len] = this->prefill(sampler_config, input_ids_batch);
        // decode
        if (!this->model_config_.use_cache) {
            max_length += after_padding.at(0).size();
        }
        auto output_tokens = this->decode(sampler_config, prefilled, seq_len, max_length, stream_handler);

        // 5. 后处理
        this->output_tokens_post_processing(after_padding.at(0).size(), output_tokens);
        return output_tokens;
    }

    Tensor BaseModel::get_token_from_last_logits(SamplerConfig config, const Tensor& outputs) const {
        // outputs.shape: (batch_size, seq_len, vocab_size)
        // 1. 取最后一个时间步输出：(batch_size, 1, vocab_size)
        // 由于只实现了vec或者matrix的采样，所以转换为(batch_size, vocab_size)
        std::size_t batch_size = outputs.shape().at(0);
        std::size_t seq_len = outputs.shape().at(1);
        std::size_t vocab_size = outputs.shape().at(2);

        // 每一个batch，取最后一个输出
        Tensor last_logits(outputs.data_type(), outputs.device_type(), {batch_size, vocab_size});
        for (std::size_t i = 0; i < batch_size; ++i) {
            void* data = outputs.data({i, seq_len - 1});
            Tensor last_logits_batch = outputs.view({vocab_size}, data);
            last_logits.copy_from(last_logits_batch, i);
        }

        // 2. 通过采样器，获取next_token：(batch_size, 1)
        auto next_token_tensor = apply_sampling(config, last_logits);
        return next_token_tensor;
    }

    void BaseModel::parse_tokens_from_tensor(const Tensor& tokens_tensor,
        std::vector<std::vector<std::int32_t>>& tokens, StreamHandler stream_handler) const {
        // 如果是非cpu设备，先转换到cpu
        Tensor t = tokens_tensor.clone(DeviceType::CPU);

        // 按batch解析
        std::size_t batch_size = tokens_tensor.shape().at(0);
        std::size_t seq_len = tokens_tensor.shape().at(1);
        for (std::size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // 跳过已经生成完成的batch
            if (this->token_generate_status_.find(batch_idx) != token_generate_status_.end() &&
                this->token_generate_status_.at(batch_idx)) {
                continue;
            }
            // 解析token
            for (std::size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
                auto token = *static_cast<std::int32_t*>(t.data({batch_idx, seq_idx}));
                tokens[batch_idx].push_back(token);
            }
            // 流式处理回调
            if (stream_handler != nullptr) {
                stream_handler(batch_idx, tokens[batch_idx].back());
            }
        }
    }

    void BaseModel::init_token_generate_status(std::size_t batch_size) {
        for (std::size_t i = 0; i < batch_size; ++i) {
            token_generate_status_[i] = false;
        }
    }

    bool BaseModel::check_finished(
        const std::vector<std::vector<std::int32_t>>& tokens, std::size_t max_length) {
        std::size_t batch_size = tokens.size();
        for (std::size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            if (tokens[batch_idx].size() >= max_length ||
                tokens[batch_idx].back() == this->model_config_.eos_token_id) {
                token_generate_status_[batch_idx] = true;
            }
        }

        // 检查是否所有batch都生成完成
        return std::all_of(token_generate_status_.begin(), token_generate_status_.end(),
            [](const auto& p)->bool {
                return p.second;
            });
    }

    // prefill阶段，只执行一次forward，初始化kv cache
    std::tuple<Tensor, std::size_t> BaseModel::prefill(SamplerConfig config, const Tensor& input_ids_batch) {
        // 1. input_ids_batch.shape: (batch_size, max_seq_len) ，为kv cache设置prefill阶段最大的seq len
        std::size_t seq_len = input_ids_batch.shape().at(1);
        this->set_seq_len(seq_len);
        // 2. forward输出：(batch_size, max_seq_len, vocab_size)
        auto outputs = this->forward(input_ids_batch);
        // 3. 获取最后一个时间步输出
        auto last_output = this->get_token_from_last_logits(config, outputs);
        // 4. 输出
        if (model_config_.use_cache) {
            // 如果开了kv cache，则取最后一个时间步的输出，维度为：(batch_size, 1)
            return std::make_tuple(last_output, seq_len + 1);
        } else {
            // 如果不开kv cache，则将input_ids_batch和last_output拼接在一起，维度为：(batch_size, max_seq_len + 1)
            auto output = Tensor::concat(input_ids_batch, last_output,  Tensor::ConcatDim::eColWise);
            return std::make_tuple(output, seq_len + 1);
        }
    }

    // decode阶段，执行多次forward，使用kv cache
    std::vector<std::vector<std::int32_t>> BaseModel::decode(SamplerConfig config,
        const Tensor& prefilled_batch, std::size_t seq_len,
        std::size_t max_length, StreamHandler stream_handler) {
        // 1. 初始化生成token序列
        std::size_t batch_size = prefilled_batch.shape().at(0);
        std::vector<std::vector<std::int32_t>> tokens(batch_size);
        parse_tokens_from_tensor(prefilled_batch, tokens, stream_handler);

        // 2. 循环生成
        Tensor generated = prefilled_batch;
        for (;;) {
            this->set_seq_len(seq_len);
            Tensor outputs = this->forward(generated);
            auto tokens_tensor  = this->get_token_from_last_logits(config, outputs);
            parse_tokens_from_tensor(tokens_tensor, tokens, stream_handler);
            if (this->check_finished(tokens, max_length)) {
                break;
            }

            ++seq_len;

            if (model_config_.use_cache) {
                // 如果使用cache，则直接将next_token作为下一个时间步的输入
                generated = tokens_tensor;
            } else {
                // 如果不使用cache，则将next_token拼接在generated下方，维度为：(batch_size, seq_len + 1)
                generated = Tensor::concat(generated, tokens_tensor,  Tensor::ConcatDim::eColWise);
            }
        }
        // 3. 清空kv cache
        if (model_config_.use_cache) {
            this->clear_kv_cache();
        }
        // 4. 重置状态
        this->token_generate_status_.clear();
        return tokens;
    }

    void BaseModel::left_padding(std::vector<std::vector<std::int32_t>>& input_ids) const {
        // 左填充，使每个batch的序列长度一致
        std::size_t max_seq_len = 0;
        for (const auto& ids : input_ids) {
            if (ids.size() > max_seq_len) {
                max_seq_len = ids.size();
            }
        }

        for (auto& ids : input_ids) {
            while (ids.size() < max_seq_len) {
                ids.insert(ids.begin(), padding_idx_);
            }
        }
    }

    void BaseModel::output_tokens_post_processing(std::size_t after_padding_seq_len,
        std::vector<std::vector<std::int32_t>>& tokens) const {
        // 移除每个batch的填充token（开启kv cache），以及padding后的输入序列（未开启kv cache）
        for (auto& ids : tokens) {
            if (model_config_.use_cache) {
                ids.erase(std::remove(ids.begin(), ids.end(), padding_idx_), ids.end());
            } else {
                ids.erase(ids.begin(),
                ids.begin() + static_cast<std::ptrdiff_t>(after_padding_seq_len));
            }
        }
    }
} // fg42