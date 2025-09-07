//
// Created by 19373 on 2025/9/6.
//
#include "Eigen/Core"
#include <gtest/gtest.h>
#include "tests/mock_model.h"

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