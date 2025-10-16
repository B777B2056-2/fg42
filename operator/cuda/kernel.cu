//
// Created by B777B2056-2 on 2025/10/13.
//
#include <random>
#include <stdexcept>
#include <cuda_bf16.h>
#include "operator/cuda/kernel.cuh"
#include "operator/cuda/kernel_impl.cuh"

#define CUDA_DATA_TYPE_SWITCH(op, stream, block_num, thread_num, data_type, ...)    \
{   \
    switch (data_type) {    \
        case DataType::FP32: {  \
            if (stream == nullptr) {    \
                op##_cuda_impl<float><<<block_num, thread_num>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<float><<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
            }   \
        }   \
            break;  \
        case DataType::BF16:{   \
            if (stream == nullptr) {    \
                op##_cuda_impl<__nv_bfloat16><<<block_num, thread_num>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<__nv_bfloat16><<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
                }   \
        }   \
            break;  \
        case DataType::Int32:{   \
            if (stream == nullptr) {    \
                op##_cuda_impl<std::int32_t><<<block_num, thread_num>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<std::int32_t><<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
            }   \
        }   \
            break;  \
        default:    \
            throw std::runtime_error("Unsupported data type");  \
    }   \
}   \

#define CUDA_DATA_TYPE_SWITCH_WITH_SHARED_MEM(op, stream, block_num, thread_num, shared_mem_size, data_type, ...)    \
{   \
    switch (data_type) {    \
        case DataType::FP32: {  \
            if (stream == nullptr) {    \
                op##_cuda_impl<float><<<block_num, thread_num, shared_mem_size>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<float><<<block_num, thread_num, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
            }   \
        }   \
            break;  \
        case DataType::BF16:{   \
            if (stream == nullptr) {    \
                op##_cuda_impl<__nv_bfloat16><<<block_num, thread_num, shared_mem_size>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<__nv_bfloat16><<<block_num, thread_num, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
                }   \
        }   \
            break;  \
        case DataType::Int32:{   \
            if (stream == nullptr) {    \
                op##_cuda_impl<std::int32_t><<<block_num, thread_num, shared_mem_size>>>(__VA_ARGS__);  \
            } else {    \
                op##_cuda_impl<std::int32_t><<<block_num, thread_num, shared_mem_size, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__); \
            }   \
        }   \
            break;  \
        default:    \
            throw std::runtime_error("Unsupported data type");  \
    }   \
}   \

namespace fg42::kernel {
    static constexpr std::int32_t DEFAULT_1D_THREAD_NUM = 256;
    static constexpr dim3 DEFAULT_2D_THREAD_NUM(16, 16);
    static constexpr dim3 DEFAULT_3D_THREAD_NUM(8, 8, 4);

    inline dim3 block_num_1d(std::int32_t x, std::int32_t thread_num=DEFAULT_1D_THREAD_NUM) {
        return (x + thread_num - 1) / thread_num;
    }

    inline dim3 block_num_2d(std::int32_t x, dim3 thread_num=DEFAULT_2D_THREAD_NUM) {
        dim3 block_num;
        block_num.x = (x + thread_num.x - 1) / thread_num.x;
        block_num.y = (x + thread_num.y - 1) / thread_num.y;
        return block_num;
    }

    inline dim3 block_num_3d(std::int32_t x, dim3 thread_num=DEFAULT_3D_THREAD_NUM) {
        dim3 block_num;
        block_num.x = (x + thread_num.x - 1) / thread_num.x;
        block_num.y = (x + thread_num.y - 1) / thread_num.y;
        block_num.z = (x + thread_num.z - 1) / thread_num.z;
        return block_num;
    }

    Tensor add_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream) {
        auto data_type = input1.data_type();
        auto input1_size = static_cast<std::int32_t>(input1.size());
        auto input2_size = static_cast<std::int32_t>(input2.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(input1_size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input1.device_type(), input1.shape());
        CUDA_DATA_TYPE_SWITCH(add, stream, block_num, thread_num, data_type, input1_size, input2_size,
            input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor negate_kernel_cuda(const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(negate, stream, block_num, thread_num, data_type, size, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor vec_outer_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream) {
        auto data_type = input1.data_type();
        auto input1_size = static_cast<std::int32_t>(input1.size());
        auto input2_size = static_cast<std::int32_t>(input2.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(input1_size * input2_size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input1.device_type(), {input1.shape().at(0), input2.shape().at(0)});
        CUDA_DATA_TYPE_SWITCH(outer, stream, block_num, thread_num, data_type, input1_size, input2_size,
            input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor mul_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream) {
        auto data_type = input1.data_type();
        auto input1_size = static_cast<std::int32_t>(input1.size());
        auto input2_size = static_cast<std::int32_t>(input2.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(input1_size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input1.device_type(), input1.shape());
        CUDA_DATA_TYPE_SWITCH(mul, stream, block_num, thread_num, data_type, input1_size, input2_size,
            input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor mul_with_constant_value_kernel_cuda(float value, const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(mul_with_constant_value, stream, block_num, thread_num, data_type, size, value, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor matmul_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream) {
        auto data_type = input1.data_type();

        std::int32_t batch_size = 1;
        for (std::size_t i = 0; i < input1.shape().size() - 2; ++i) {
            batch_size *= static_cast<std::int32_t>(input1.shape().at(i));
        }

        std::int32_t batch_size_2 = 1;
        for (std::size_t i = 0; i < input2.shape().size() - 2; ++i) {
            batch_size_2 *= static_cast<std::int32_t>(input2.shape().at(i));
        }

        auto in1_n_rows = static_cast<std::int32_t>(input1.shape().at(input1.shape().size() - 2));
        auto in1_n_cols = static_cast<std::int32_t>(input1.shape().at(input1.shape().size() - 1));
        auto in2_n_cols = static_cast<std::int32_t>(input2.shape().at(input2.shape().size() - 1));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(batch_size * in1_n_rows * in2_n_cols, DEFAULT_1D_THREAD_NUM));

        auto output_shape = input1.shape();
        output_shape.back() = input2.shape().back();

        Tensor output(data_type, input1.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(matmul, stream, block_num, thread_num, data_type, batch_size, batch_size_2,
            in1_n_rows, in1_n_cols, in2_n_cols, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
        return output;
    }

    static Tensor transpose_1d_kernel_cuda(const Tensor& input) {
        Tensor output = input.clone(input.device_type());
        output.reshape({input.shape().at(0), 1});
        return output;
    }

    static Tensor transpose_2d_kernel_cuda(const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size, DEFAULT_1D_THREAD_NUM));

        auto n_rows = static_cast<std::int32_t>(input.shape().at(0));
        auto n_cols = static_cast<std::int32_t>(input.shape().at(1));

        std::vector<std::size_t> output_shape{
            static_cast<std::size_t>(n_cols),
            static_cast<std::size_t>(n_rows),
        };

        Tensor output(data_type, input.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(transpose_2d, stream, block_num, thread_num, data_type, n_rows, n_cols, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    static Tensor transpose_3d_kernel_cuda(const Tensor& input, std::size_t dim0, std::size_t dim1, void* stream) {
        auto data_type = input.data_type();
        auto n_first_dim = static_cast<std::int32_t>(input.shape().at(0));
        auto n_rows = static_cast<std::int32_t>(input.shape().at(1));
        auto n_cols = static_cast<std::int32_t>(input.shape().at(2));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_first_dim * n_rows * n_cols, DEFAULT_1D_THREAD_NUM));

        auto output_shape = input.shape();
        std::swap(output_shape.at(dim0), output_shape.at(dim1));
        Tensor output(data_type, input.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(transpose_3d, stream, block_num, thread_num, data_type,
            n_first_dim, n_rows, n_cols, dim0, dim1, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    static Tensor transpose_4d_kernel_cuda(const Tensor& input, std::size_t dim0, std::size_t dim1, void* stream) {
        auto data_type = input.data_type();
        auto n_first_dim = static_cast<std::int32_t>(input.shape().at(0));
        auto n_second_dim = static_cast<std::int32_t>(input.shape().at(1));
        auto n_rows = static_cast<std::int32_t>(input.shape().at(2));
        auto n_cols = static_cast<std::int32_t>(input.shape().at(3));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_first_dim * n_second_dim * n_rows * n_cols, DEFAULT_1D_THREAD_NUM));

        auto output_shape = input.shape();
        std::swap(output_shape.at(dim0), output_shape.at(dim1));
        Tensor output(data_type, input.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(transpose_4d, stream, block_num, thread_num, data_type,
            n_first_dim, n_second_dim, n_rows, n_cols, dim0, dim1, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    static Tensor transpose_5d_kernel_cuda(const Tensor& input, std::size_t dim0, std::size_t dim1, void* stream) {
        auto data_type = input.data_type();
        auto batch_size = static_cast<std::int32_t>(input.shape().at(0));
        auto n_first_dim = static_cast<std::int32_t>(input.shape().at(1));
        auto n_second_dim = static_cast<std::int32_t>(input.shape().at(2));
        auto n_rows = static_cast<std::int32_t>(input.shape().at(3));
        auto n_cols = static_cast<std::int32_t>(input.shape().at(4));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(batch_size * n_first_dim * n_second_dim * n_rows * n_cols,
            DEFAULT_1D_THREAD_NUM));

        auto output_shape = input.shape();
        std::swap(output_shape.at(dim0), output_shape.at(dim1));
        Tensor output(data_type, input.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(transpose_5d, stream, block_num, thread_num, data_type, batch_size,
            n_first_dim, n_second_dim, n_rows, n_cols, dim0, dim1, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor transpose_kernel_cuda(const Tensor& input, std::size_t dim0, std::size_t dim1, void* stream) {
        const auto& shape = input.shape();
        if (shape.size() == 1) {
            return transpose_1d_kernel_cuda(input);
        } else if (shape.size() == 2) {
            return transpose_2d_kernel_cuda(input, stream);
        } else if (shape.size() == 3) {
            return transpose_3d_kernel_cuda(input, dim0, dim1, stream);
        } else if (shape.size() == 4) {
            return transpose_4d_kernel_cuda(input, dim0, dim1, stream);
        } else if (shape.size() == 5) {
            return transpose_5d_kernel_cuda(input, dim0, dim1, stream);
        }
        throw std::runtime_error("transpose_kernel_cuda: invalid shape");
    }

    Tensor embedding_kernel_cuda(const Tensor* weight_tensor, const Tensor& input_tensors, void* stream) {
        auto data_type = weight_tensor->data_type();
        auto embedding_dim = static_cast<std::int32_t>(weight_tensor->shape().at(0));
        auto vocab_size = static_cast<std::int32_t>(weight_tensor->shape().at(1));
        auto batch_size = static_cast<std::int32_t>(input_tensors.shape().at(0));
        auto seq_length = static_cast<std::int32_t>(input_tensors.shape().at(1));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(batch_size * seq_length * vocab_size,
            DEFAULT_1D_THREAD_NUM));

        std::vector<std::size_t> output_shape{
            static_cast<std::size_t>(batch_size),
            static_cast<std::size_t>(seq_length),
            static_cast<std::size_t>(vocab_size)};

        Tensor output(data_type, weight_tensor->device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(embedding, stream, block_num, thread_num, data_type,
            embedding_dim, vocab_size, batch_size, seq_length,
            weight_tensor->raw_ptr(), static_cast<std::int32_t*>(input_tensors.raw_ptr()), output.raw_ptr());
        return output;
    }

    Tensor repeat_kv_kernel_cuda(const Tensor& x, std::size_t n_rep, void* stream) {
        auto data_type = x.data_type();
        auto batch_size = static_cast<std::int32_t>(x.shape().at(0));
        auto n_kv_heads = static_cast<std::int32_t>(x.shape().at(1));
        auto n_rows = static_cast<std::int32_t>(x.shape().at(2));
        auto n_cols = static_cast<std::int32_t>(x.shape().at(3));

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(batch_size * n_kv_heads * n_rows * n_cols,
            DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, x.device_type(), {
            static_cast<std::size_t>(batch_size),
            static_cast<std::size_t>(n_rep * n_kv_heads),
            static_cast<std::size_t>(n_rows),
            static_cast<std::size_t>(n_cols)
        });
        CUDA_DATA_TYPE_SWITCH(repeat_kv, stream, block_num, thread_num, data_type, n_rep,
            batch_size, n_kv_heads, n_rows, n_cols, x.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor softmax_kernel_cuda(const Tensor& input, std::optional<float> t, void* stream) {
        auto data_type = input.data_type();

        std::size_t n_rows = 1;
        for (int i = 0; i < input.shape().size() - 1; ++i) {
            n_rows *= input.shape().at(i);
        }
        auto n_cols = input.shape().back();

        float* temp_t = nullptr;
        if (t.has_value()) {
            temp_t = &t.value();
        }

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_rows * n_cols,
            DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(softmax, stream, block_num, thread_num, data_type,
            1, n_rows, n_cols, temp_t, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor silu_kernel_cuda(const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size,
            DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(silu, stream, block_num, thread_num, data_type, size, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor rme_norm_kernel_cuda(const Tensor& input, float eps, void* stream) {
        auto data_type = input.data_type();

        std::size_t batch_size = 1;
        for (int i = 0; i < input.shape().size() - 2; ++i) {
            batch_size *= input.shape().at(i);
        }

        auto n_rows = input.shape().size() > 1 ? input.shape().at(input.shape().size() - 2) : 1;
        auto n_cols = input.shape().back();

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(batch_size * n_rows * n_cols,
            DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(rms_norm, stream, block_num, thread_num, data_type,
            batch_size, n_rows, n_cols, input.raw_ptr(), output.raw_ptr(), eps);
        return output;
    }

    Tensor vec_or_matrix_argmax_kernel_cuda(const Tensor& input, std::size_t n, void* stream) {
        auto data_type = input.data_type();
        auto n_rows = input.shape().at(0);
        auto n_cols = input.shape().at(1);

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_rows, DEFAULT_1D_THREAD_NUM));

        Tensor output(DataType::Int32, input.device_type(), {n_rows, 1});
        CUDA_DATA_TYPE_SWITCH(vec_or_matrix_argmax, stream, block_num, thread_num, data_type, n_rows, n_cols,
            input.raw_ptr(), static_cast<std::int32_t*>(output.raw_ptr()));
        return output;
    }

    Tensor rotate_half_kernel_cuda(const Tensor& input, void* stream) {
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < input.shape().size() - 1; ++i) {
            total_batch *= input.shape().at(i);
        }
        std::vector<std::size_t> view_shape;
        view_shape.push_back(total_batch);
        view_shape.push_back(input.shape().back());

        Tensor input_view = input.view(view_shape);
        Tensor output(input.data_type(), input.device_type(), view_shape);

        auto data_type = input.data_type();
        auto n_rows = view_shape.at(0);
        auto n_cols = view_shape.at(1);

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_rows * n_cols, DEFAULT_1D_THREAD_NUM));

        CUDA_DATA_TYPE_SWITCH(rotate_half, stream, block_num, thread_num, data_type, n_rows, n_cols, input_view.raw_ptr(), output.raw_ptr());
        output.reshape(input.shape());
        return output;
    }

    Tensor cos_kernel_cuda(const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(cosine, stream, block_num, thread_num, data_type, size, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor sin_kernel_cuda(const Tensor& input, void* stream) {
        auto data_type = input.data_type();
        auto size = static_cast<std::int32_t>(input.size());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(size, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, input.device_type(), input.shape());
        CUDA_DATA_TYPE_SWITCH(sine, stream, block_num, thread_num, data_type, size, input.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor concat_by_col_wise_kernel_cuda(const Tensor& x1, const Tensor& x2, void* stream) {
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < x1.shape().size() - 1; ++i) {
            total_batch *= x1.shape().at(i);
        }

        auto data_type = x1.data_type();
        auto n_rows = static_cast<std::int32_t>(total_batch);
        auto n_cols_1 = static_cast<std::int32_t>(x1.shape().back());
        auto n_cols_2 = static_cast<std::int32_t>(x2.shape().back());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(n_rows * (n_cols_1 + n_cols_2), DEFAULT_1D_THREAD_NUM));

        auto output_shape = x1.shape();
        output_shape.back() = x1.shape().back() + x2.shape().back();
        Tensor output(data_type, x1.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(concat_by_col_wise, stream, block_num, thread_num, data_type,
            n_rows, n_cols_1, n_cols_2, x1.raw_ptr(), x2.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor concat_by_row_wise_kernel_cuda(const Tensor& x1, const Tensor& x2, void* stream) {
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < x1.shape().size() - 2; ++i) {
            total_batch *= x1.shape().at(i);
        }

        auto data_type = x1.data_type();
        auto n_rows_1 = static_cast<std::int32_t>(x1.shape().at(x1.shape().size() - 2));
        auto n_rows_2 = static_cast<std::int32_t>(x2.shape().at(x2.shape().size() - 2));
        auto n_cols = static_cast<std::int32_t>(x1.shape().back());

        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(total_batch * (n_rows_1 + n_rows_2) * n_cols, DEFAULT_1D_THREAD_NUM));

        auto output_shape = x1.shape();
        output_shape.at(output_shape.size() - 2) = x1.shape().at(x1.shape().size() - 2) +
            x2.shape().at(x2.shape().size() - 2);
        Tensor output(data_type, x1.device_type(), output_shape);
        CUDA_DATA_TYPE_SWITCH(concat_by_row_wise, stream, block_num, thread_num, data_type,
            total_batch, n_rows_1, n_rows_2, n_cols,
            x1.raw_ptr(), x2.raw_ptr(), output.raw_ptr());
        return output;
    }

    Tensor causal_mask_kernel_cuda(DataType data_type, std::size_t l, std::size_t s, void* stream) {
        dim3 thread_num(DEFAULT_1D_THREAD_NUM);
        dim3 block_num(block_num_1d(l * s, DEFAULT_1D_THREAD_NUM));

        Tensor output(data_type, DeviceType::NvidiaGPU, {l, s});
        CUDA_DATA_TYPE_SWITCH(causal_mask, stream, block_num, thread_num, data_type, l, s, output.raw_ptr());
        return output;
    }

    Tensor multinomial_kernel_cuda(const Tensor& x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos, void* stream) {
        auto data_type = x.data_type();
        Tensor output(x.data_type(), x.device_type(), {x.shape().at(0), num_samples});

        std::size_t num_rows = x.shape().at(0);
        std::size_t num_cols = x.shape().at(1);

        auto* d_output = static_cast<std::int32_t*>(output.raw_ptr());

        // 如果提供了row_end_pos，需要将其传输到设备
        int* d_row_end_pos = nullptr;
        std::vector<int> h_row_end_pos;

        if (row_end_pos) {
            h_row_end_pos.resize(num_rows);
            for (std::size_t i = 0; i < num_rows; ++i) {
                h_row_end_pos[i] = static_cast<int>(row_end_pos(i));
            }
            cudaMalloc(&d_row_end_pos, num_rows * sizeof(int));
            cudaMemcpy(d_row_end_pos, h_row_end_pos.data(),
                       num_rows * sizeof(int), cudaMemcpyHostToDevice);
        }

        // 生成随机种子
        std::random_device rd;
        unsigned long long seed = rd();

        // 计算共享内存大小（每行需要存储累积概率）
        std::size_t shared_mem_size = num_cols * sizeof(float);

        // 启动核函数
        // 每个线程块处理一行，每个线程处理一个样本
        dim3 blocks(num_rows);
        dim3 threads(num_samples);

        CUDA_DATA_TYPE_SWITCH_WITH_SHARED_MEM(multinomial, stream, blocks, threads, shared_mem_size, data_type,
            x.raw_ptr(), d_output, d_row_end_pos, num_rows, num_cols, num_samples, seed);

        cudaDeviceSynchronize();

        // 清理设备内存
        if (d_row_end_pos != nullptr) {
            cudaFree(d_row_end_pos);
        }
        return output;
    }
#undef CUDA_DATA_TYPE_SWITCH
#undef CUDA_DATA_TYPE_SWITCH_WITH_SHARED_MEM
}   // namespace fg42::kernel