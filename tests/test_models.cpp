//
// Created by 19373 on 2025/9/6.
//
#include "Eigen/Core"
#include <gtest/gtest.h>
#include "model/BaseModel.h"
#include "operator/VecAddOperator.h"

class MockModelLoader : fg42::BaseModel {
    public:
        MockModelLoader() = delete;

        explicit MockModelLoader(const std::string& path, fg42::DeviceType device_type)
            : fg42::BaseModel(path, device_type) {}

        ~MockModelLoader() override = default;

        const fg42::Tensor& mock_input() const {
            return this->state_dict_.begin()->second;
        }

        fg42::Tensor forward(const fg42::Tensor& input) override {
            // 使用加法mock前向传播
            fg42::Tensor input_copy(input);

            std::vector<const fg42::Tensor*> input_tensors{&input, &input_copy};

            fg42::Tensor output(input_copy.data_type(), input_copy.device_type(), input_copy.shape());
            std::vector<fg42::Tensor*> output_tensors({&output});

            fg42::kernel::VecAddOperator op(input.device_type());
            op.forward(input_tensors, output_tensors, nullptr);
            return output;
        }
};

extern bool is_float_equal(float a, float b);

TEST(ModelTest, ModelFilesLoaderCPU) {
    using bfloat16 = Eigen::bfloat16;

    MockModelLoader model(R"(C:\Users\19373\Downloads)", fg42::DeviceType::CPU);

    const fg42::Tensor& mock_input = model.mock_input();
    auto output = model.forward(mock_input);

    auto* input_ptr = static_cast<bfloat16*>(mock_input.raw_ptr());
    auto* output_ptr = static_cast<bfloat16*>(output.raw_ptr());
    std::size_t tensor_size = mock_input.size();
    for (auto i = 0; i < tensor_size; ++i) {
        auto val = static_cast<float>(*output_ptr);
        auto expected_val = static_cast<float>(*input_ptr) + static_cast<float>(*input_ptr);
        EXPECT_TRUE(is_float_equal(val, expected_val));
        ++input_ptr;
        ++output_ptr;
    }
}