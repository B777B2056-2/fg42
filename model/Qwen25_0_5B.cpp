//
// Created by 19373 on 2025/9/5.
//
#include "model/Qwen25_0_5B.h"

namespace fg42 {
    Qwen25_0_5B::Qwen25_0_5B(const std::string& path, DeviceType device_type) : BaseModel(path, device_type) {}

    Tensor Qwen25_0_5B::forward(const Tensor& input) {
        throw std::runtime_error("Not implemented");
    }
} // fg42