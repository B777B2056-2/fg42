//
// Created by B777B2056-2 on 2025/10/10.
//

#ifndef FG42_SAFETENSORSLOADER_H
#define FG42_SAFETENSORSLOADER_H
#include "model/weights_loaders/BaseWeightsLoader.h"
#include "nlohmann/json.hpp"

namespace fg42 {
    class SafeTensorsLoader final : public BaseWeightsLoader {
    public:
        ~SafeTensorsLoader() override = default;
        void load(const std::filesystem::path& dir_path,
            StateDict& state_dict, DeviceType device_type, DataType data_type) override;

    private:
        std::size_t header_offset_ = 0;

        nlohmann::ordered_json read_header(WeightsFileReader& ifs);

        void read_data(WeightsFileReader& ifs, const nlohmann::ordered_json& header,
            StateDict& state_dict, DeviceType device_type, DataType data_type);

        void read_one_safetensors(const std::filesystem::path& weights_file_path,
            StateDict& state_dict, DeviceType device_type, DataType data_type);
    };
} // fg42

#endif //FG42_SAFETENSORSLOADER_H