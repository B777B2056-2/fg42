//
// Created by 19373 on 2025/9/5.
//

#ifndef FG42_QWEN25_0_5B_H
#define FG42_QWEN25_0_5B_H
#include "model/BaseModel.h"

namespace fg42 {
    class Qwen25_0_5B : public BaseModel {
    public:
        Qwen25_0_5B() = delete;
        explicit Qwen25_0_5B(const std::string& path, DeviceType device_type);
        ~Qwen25_0_5B() override = default;

        Tensor forward(const Tensor& input) override;
    };
} // fg42

#endif //FG42_QWEN25_0_5B_H