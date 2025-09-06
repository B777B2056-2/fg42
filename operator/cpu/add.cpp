//
// Created by 19373 on 2025/9/4.
//
#include "Eigen/Core"
#include "operator/cpu/add.h"

namespace fg42::kernel {
    using Int8Matrix = Eigen::Map<Eigen::Matrix<std::int8_t, Eigen::Dynamic, 1>>;
    using BFloat16Matrix = Eigen::Map<Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, 1>>;
    using Float32Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>;

    void add_kernel_cpu(const Tensor& input1, const Tensor& input2, Tensor& output) {
        switch (input1.data_type()) {
        case DataType::Int8: {
            Int8Matrix input_vec1(static_cast<std::int8_t*>(input1.raw_ptr()), input1.size());
            Int8Matrix input_vec2(static_cast<std::int8_t*>(input2.raw_ptr()), input2.size());
            Int8Matrix output_vec(static_cast<std::int8_t*>(output.raw_ptr()), output.size());
            output_vec = input_vec1 + input_vec2;
            break;
        }
        case DataType::BF16: {
            BFloat16Matrix input_vec1(static_cast<Eigen::bfloat16*>(input1.raw_ptr()), input1.size());
            BFloat16Matrix input_vec2(static_cast<Eigen::bfloat16*>(input2.raw_ptr()), input2.size());
            BFloat16Matrix output_vec(static_cast<Eigen::bfloat16*>(output.raw_ptr()), output.size());
            output_vec = input_vec1 + input_vec2;
            break;
        }
        case DataType::FP32: {
            Float32Matrix input_vec1(static_cast<float*>(input1.raw_ptr()), input1.size());
            Float32Matrix input_vec2(static_cast<float*>(input2.raw_ptr()), input2.size());
            Float32Matrix output_vec(static_cast<float*>(output.raw_ptr()), output.size());
            output_vec = input_vec1 + input_vec2;
            break;
        }
        default:
            throw std::runtime_error("unsupported data type");
        }
    }
}