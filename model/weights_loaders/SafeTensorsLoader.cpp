//
// Created by B777B2056-2 on 2025/10/10.
//
#include "model/weights_loaders/SafeTensorsLoader.h"
#include "util/util.h"

namespace fg42 {
    // 加载safetensors格式的权重到内存(https://huggingface.co/docs/safetensors/index)
    void SafeTensorsLoader::load(const std::filesystem::path& dir_path,
        StateDict& state_dict, DeviceType device_type) {
        // 寻找所有safetensors文件
        std::vector<std::string> safetensors_file_names;
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (util::ends_with(entry.path().string(), ".safetensors")) {
                safetensors_file_names.push_back(entry.path().string());
            }
        }
        if (safetensors_file_names.empty()) {
            throw std::runtime_error("No safetensors files found");
        }

        for (const auto& file_name : safetensors_file_names) {
            const std::filesystem::path weights_file_path = dir_path / file_name;
            this->read_one_safetensors(weights_file_path, state_dict, device_type);
        }
    }

    nlohmann::ordered_json SafeTensorsLoader::read_header(WeightsFileReader& ifs) {
        // 读取文件头长度
        std::uint64_t header_length = 0;
        ifs.read(reinterpret_cast<char*>(&header_length), sizeof(std::uint64_t));
        // 读取文件头内容
        std::string header_json(header_length, '0');
        ifs.read(header_json.data(), static_cast<std::streamsize>(header_length));
        header_offset_ = ifs.offset();
        // 解析json
        return nlohmann::ordered_json::parse(header_json);
    }

    void SafeTensorsLoader::read_data(WeightsFileReader& ifs,
        const nlohmann::ordered_json& header, StateDict& state_dict,
        DeviceType device_type) {
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
            DataType real_data_type = DataType::Unknown;
            if (tensor_d_type == "F32") {
                real_data_type = DataType::FP32;
            } else if (tensor_d_type == "BF16") {
                real_data_type = DataType::BF16;
            }

            Tensor tensor(real_data_type, device_type, tensor_shape);
            // 获取张量在权重文件的范围
            auto tensor_offset_range = x.value()["data_offsets"].get<std::vector<std::size_t>>();
            // 校验张量字节长度
            if (tensor.bytes_size() != (tensor_offset_range[1] - tensor_offset_range[0])) {
                throw std::runtime_error("Fail on parse tensor offsets");
            }
            // 读取张量二进制数据
            ifs.set_offset(header_offset_ + tensor_offset_range[0]);
            ifs.read_into_tensor(tensor);
            // 判断是否需要转置
            if (weight_need_transpose_func_ && weight_need_transpose_func_(tensor_name)) {
                tensor = tensor.transpose();
            }
            // 存入权重map
            state_dict.emplace(tensor_name, tensor);
        }
    }

    void SafeTensorsLoader::read_one_safetensors(const std::filesystem::path& weights_file_path, StateDict& state_dict,
        DeviceType device_type) {
        // 打开文件
        WeightsFileReader ifs(weights_file_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Could not open file " + weights_file_path.string());
        }
        // 读取文件头
        auto header = SafeTensorsLoader::read_header(ifs);
        // 读取数据部分
        this->read_data(ifs, header, state_dict, device_type);
    }
} // fg42