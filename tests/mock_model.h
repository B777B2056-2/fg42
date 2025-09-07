//
// Created by 19373 on 2025/9/7.
//

#ifndef FG42_MOCK_MODEL_H
#define FG42_MOCK_MODEL_H
#include "model/BaseModel.h"
#include "operator/VecAddOperator.h"

class MockModelLoader : fg42::BaseModel {
public:
    MockModelLoader() = delete;

    explicit MockModelLoader(const std::string& path, fg42::DeviceType device_type)
        : fg42::BaseModel(path, device_type) {}

    ~MockModelLoader() override = default;

    [[nodiscard]] const fg42::Tensor& mock_input() const {
        return this->state_dict_.begin()->second;
    }

    [[nodiscard]] const fg42::StateDict& state_dict() const { return this->state_dict_; }

    [[nodiscard]] const fg42::Tensor& get_weight_by_name(const std::string& name) const {
        return this->state_dict_.at(name);
    }

    fg42::Tensor forward(const fg42::Tensor& input) override {
        // 使用加法mock前向传播
        fg42::Tensor input_copy(input);

        std::vector<const fg42::Tensor*> input_tensors{&input, &input_copy};

        fg42::kernel::VecAddOperator op(input.device_type());
        return op.forward(input_tensors, nullptr);
    }
};

#endif //FG42_MOCK_MODEL_H