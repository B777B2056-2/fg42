//
// Created by B777B2056-2 on 2025/9/8.
//
#include "util/util.h"
#include "operator/cpu/kernel_impl.h"

namespace fg42::kernel {
    Tensor add_kernel_cpu(const Tensor& input1, const Tensor& input2) {
        Tensor output(input1.data_type(), input1.device_type(), input1.shape());

        // 如果二者维度相同，直接相加
        if (Tensor::shape_equal(input1.shape(), input2.shape())) {
            DATA_TYPE_SWITCH(input1.data_type(), add_kernel_cpu_impl, input1, input2, output);
        } else {
            DATA_TYPE_SWITCH(input1.data_type(), boardcast_add_kernel_cpu_impl, input1, input2, output);
        }
        return output;
    }

    Tensor negate_kernel_cpu(const Tensor& input) {
        Tensor output(input.data_type(), input.device_type(), input.shape());
        DATA_TYPE_SWITCH(input.data_type(), negated_kernel_cpu_impl, input, output);
        return output;
    }

    Tensor vec_outer_kernel_cpu(const Tensor& input1, const Tensor& input2) {
        Tensor output(input1.data_type(), input1.device_type(), {input1.shape().at(0), input2.shape().at(0)});
        DATA_TYPE_SWITCH(input1.data_type(), vec_outer_kernel_cpu_impl, input1, input2, output);
        return output;
    }

    Tensor mul_kernel_cpu(const Tensor& input1, const Tensor& input2) {
        Tensor output(input1.data_type(), input1.device_type(), input1.shape());

        // 如果二者维度相同，直接相乘
        if (Tensor::shape_equal(input1.shape(), input2.shape())) {
            DATA_TYPE_SWITCH(input1.data_type(), mul_kernel_cpu_impl, input1, input2, output);
        } else {
            DATA_TYPE_SWITCH(input1.data_type(), boardcast_mul_kernel_cpu_impl, input1, input2, output);
        }

        return output;
    }

    Tensor mul_with_constant_value_kernel_cpu(float value, const Tensor& input2) {
        Tensor output(input2.data_type(), input2.device_type(), input2.shape());
        DATA_TYPE_SWITCH(input2.data_type(), mul_with_constant_kernel_cpu_impl, input2, output, value);
        return output;
    }

    // 普通二维矩阵点乘
    static Tensor matrix_matmul_kernel_cpu(const Tensor& input1, const Tensor& input2) {
        const auto& shape1 = input1.shape();
        const auto& shape2 = input2.shape();

        if (shape1.size() != 2 || shape2.size() != 2) {
            throw std::runtime_error("input tensor shape must not be vector");
        }

        std::vector<std::size_t> output_shape = shape1;
        output_shape[output_shape.size() - 2] = shape1.at(shape1.size() - 2);
        output_shape[output_shape.size() - 1] = shape2.at(shape2.size() - 1);

        Tensor output(input1.data_type(), input1.device_type(), output_shape);

        const std::size_t n_rows_1 = shape1.at(0);
        const std::size_t n_cols_1 = shape1.at(1);
        const std::size_t n_cols_2 = shape2.at(1);

        DATA_TYPE_SWITCH(input1.data_type(), matmul_kernel_cpu_impl,
            input1, input2, output, n_rows_1, n_cols_1, n_cols_2);

        return output;
    }

    Tensor matmul_kernel_cpu(const Tensor& input1, const Tensor& input2) {
        const auto& shape1 = input1.shape();
        const auto& shape2 = input2.shape();

        if (shape1.size() < 2 || shape2.size() < 2) {
            throw std::runtime_error("input tensor shape must not be vector");
        }

        std::vector<std::size_t> output_shape = shape1;
        output_shape[output_shape.size() - 2] = shape1.at(shape1.size() - 2);
        output_shape[output_shape.size() - 1] = shape2.at(shape2.size() - 1);

        Tensor output(input1.data_type(), input1.device_type(), output_shape);

        if (shape1.size() < 2) {
            throw std::runtime_error("unsupported shape for matmul");
        }

        if (shape1.size() == 2) {
            return matrix_matmul_kernel_cpu(input1, input2);
        }

        // 矩阵批量点乘
        DATA_TYPE_SWITCH(input1.data_type(), batch_matmul_kernel_cpu_impl,
            input1, input2, output);
        return output;
    }

    Tensor transpose_kernel_cpu(const Tensor& input, std::size_t dim0, std::size_t dim1) {
        const auto& shape = input.shape();
        if (shape.size() < 2) {
            Tensor output = input.clone(input.device_type());
            output.reshape({shape.at(0), 1});
            return output;
        }

        if (shape.size() == 2) {
            const std::size_t n_rows = shape.at(0);
            const std::size_t n_cols = shape.at(1);
            Tensor output(input.data_type(), input.device_type(), {n_cols, n_rows});
            DATA_TYPE_SWITCH(input.data_type(), matrix_transpose_kernel_cpu_impl, input, output, n_rows, n_cols);
            return output;
        }

        auto new_shape = input.shape();
        std::swap(new_shape[dim0], new_shape[dim1]);

        Tensor output(input.data_type(), input.device_type(), new_shape);

        if (shape.size() == 3) {
            DATA_TYPE_SWITCH(input.data_type(), transpose_3d_tensor_kernel_cpu_impl, input, output, dim0, dim1);
        } else if (shape.size() == 4) {
            DATA_TYPE_SWITCH(input.data_type(), transpose_4d_tensor_kernel_cpu_impl, input, output, dim0, dim1);
        } else if (shape.size() == 5) {
            DATA_TYPE_SWITCH(input.data_type(), transpose_5d_tensor_kernel_cpu_impl, input, output, dim0, dim1);
        }
        return output;
    }

    Tensor embedding_kernel_cpu(const Tensor* weight_tensor, const Tensor& input_tensor) {
         // 获取权重张量的信息
         const auto embedding_dim = weight_tensor->shape().at(1);
         // 获取输入张量信息
         std::size_t batch_size = input_tensor.shape().at(0);
         std::size_t seq_length = input_tensor.shape().at(1);
         // 设置输出张量维度
         Tensor output(weight_tensor->data_type(), weight_tensor->device_type(),
             {batch_size, seq_length, embedding_dim});
        DATA_TYPE_SWITCH(weight_tensor->data_type(), embedding_kernel_cpu_impl,
             weight_tensor, input_tensor, output);
         return output;
    }

    Tensor repeat_kv_kernel_cpu(const Tensor& x, std::size_t n_rep) {
        auto batch = x.shape().at(0);
        auto num_kv_heads = x.shape().at(1);
        auto seq_len = x.shape().at(2);
        auto head_dim = x.shape().at(3);
        std::vector<std::size_t> output_shape({batch, num_kv_heads * n_rep, seq_len, head_dim});
        Tensor output(x.data_type(), x.device_type(), output_shape);
        DATA_TYPE_SWITCH(x.data_type(), repeat_kv_cpu_impl, x, output, n_rep);
        return output;
    }

    Tensor softmax_kernel_cpu(const Tensor& input, std::optional<float> t) {
        const auto& shape = input.shape();

        Tensor output(input.data_type(), input.device_type(), input.shape());
        if (shape.size() == 1) {
            DATA_TYPE_SWITCH(input.data_type(), matrix_softmax_kernel_cpu_impl, input, output, 1, shape.back(), t);
        } else if (shape.size() == 2) {
            const std::size_t n_rows = shape.at(0);
            const std::size_t n_cols = shape.at(1);
            DATA_TYPE_SWITCH(input.data_type(), matrix_softmax_kernel_cpu_impl, input, output, n_rows, n_cols, t);
        } else if (shape.size() > 2) {
            // 1. 转换为大的二维矩阵
            std::size_t n_rows = 1;
            for (std::size_t i = 0; i < shape.size() - 1; ++i) {
                n_rows *= shape[i];
            }
            const std::size_t n_cols = shape.back();
            // 2. 调用二维矩阵的softmax实现
            DATA_TYPE_SWITCH(input.data_type(), matrix_softmax_kernel_cpu_impl, input, output, n_rows, n_cols, t);
        } else {
            throw std::runtime_error("unsupported shape for softmax");
        }
        return output;
    }

    Tensor silu_kernel_cpu(const Tensor &input) {
        const auto& shape = input.shape();

        Tensor output(input.data_type(), input.device_type(), input.shape());
        DATA_TYPE_SWITCH(input.data_type(), vec_silu_kernel_cpu_impl, input, output);
        return output;
    }

    Tensor rme_norm_kernel_cpu(const Tensor& input, float eps) {
        Tensor output(input.data_type(), input.device_type(), input.shape());
        DATA_TYPE_SWITCH(input.data_type(), rme_norm_kernel_cpu_impl, input, output, eps);
        return output;
    }

    Tensor vec_or_matrix_argmax_kernel_cpu(const Tensor& input, std::size_t n) {
        const auto& shape = input.shape();
        std::size_t batch_size = (shape.size() == 1) ? 1 : shape.at(0);
        std::size_t n_col = (shape.size() == 1) ? shape.at(0) : shape.at(1);

        fg42::Tensor output(fg42::DataType::Int32, input.device_type(), {batch_size, n});

        // 按行取
        for (std::size_t b = 0; b < batch_size; ++b) {
            void* data = (shape.size() == 1) ? input.raw_ptr() : input.data({b});
            Tensor row_view = (shape.size() == 1) ? input.view() : input.view({1, n_col}, data);
            Tensor output_view = output.view({1, n}, output.data({b}));
            DATA_TYPE_SWITCH(input.data_type(), find_max_k_idx_in_row, row_view, output_view);
        }
        return output;
    }

    Tensor rotate_half_kernel_cpu(const Tensor& input) {
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < input.shape().size() - 1; ++i) {
            total_batch *= input.shape().at(i);
        }
        std::vector<std::size_t> view_shape;
        view_shape.push_back(total_batch);
        view_shape.push_back(input.shape().back());

        Tensor input_view = input.view(view_shape);
        Tensor output(input.data_type(), input.device_type(), view_shape);
        DATA_TYPE_SWITCH(input.data_type(), rotate_half_impl, input_view, output);
        output.reshape(input.shape());
        return output;
    }

    Tensor cos_kernel_cpu(const Tensor& input) {
        Tensor output(input.data_type(), input.device_type(), input.shape());
        DATA_TYPE_SWITCH(input.data_type(), cosine_impl, input, output);
        return output;
    }

    Tensor sin_kernel_cpu(const Tensor& input) {
        Tensor output(input.data_type(), input.device_type(), input.shape());
        DATA_TYPE_SWITCH(input.data_type(), sine_impl, input, output);
        return output;
    }

    Tensor concat_by_col_wise_kernel_cpu(const Tensor& x1, const Tensor& x2) {
        auto new_shape = x1.shape();
        new_shape.back() = x1.shape().back() + x2.shape().back();

        Tensor output(x1.data_type(), x1.device_type(), new_shape);
        DATA_TYPE_SWITCH(x1.data_type(), concat_by_col_wise, x1, x2, output);
        return output;
    }

    Tensor concat_by_row_wise_kernel_cpu(const Tensor& x1, const Tensor& x2) {
        auto new_shape = x1.shape();
        new_shape[new_shape.size() - 2] = x1.shape()[x1.shape().size() - 2] + x2.shape()[x2.shape().size() - 2];

        Tensor output(x1.data_type(), x1.device_type(), new_shape);
        DATA_TYPE_SWITCH(x1.data_type(), concat_by_row_wise, x1, x2, output);
        return output;
    }

    Tensor causal_mask_kernel_cpu(DataType data_type, std::size_t l, std::size_t s) {
        Tensor output(data_type, fg42::DeviceType::CPU, {l, s});
        DATA_TYPE_SWITCH(data_type, causal_mask_impl, l, s, output);
        return output;
    }

    Tensor multinomial_kernel_cpu(const Tensor& x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos) {
        Tensor output(fg42::DataType::Int32, x.device_type(), {x.shape().at(0), num_samples});
        DATA_TYPE_SWITCH(x.data_type(), multinomial_impl, x, output, row_end_pos);
        return output;
    }
}
