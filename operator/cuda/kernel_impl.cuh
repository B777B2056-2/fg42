//
// Created by B777B2056-2 on 2025/10/13.
//

#ifndef FG42_KERNEL_IMPL_H
#define FG42_KERNEL_IMPL_H
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

namespace fg42::kernel {
    template<typename DATA_TYPE>
    __global__ void add_cuda_impl(std::int32_t in1_size, std::int32_t in2_size,
        const void* in1, const void* in2, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= in1_size) {
            return;
        }
        
        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = input1_ptr[tid] + input2_ptr[tid % in2_size];
    }

    template<typename DATA_TYPE>
    __global__ void negate_cuda_impl(std::int32_t size, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = -input_ptr[tid];
    }

    template<typename DATA_TYPE>
    __global__ void outer_cuda_impl(std::int32_t in1_size, std::int32_t in2_size,
        const void* in1, const void* in2, void* out) {
        // .z for batch, .y for row, .x for column
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = in1_size * in2_size;

        if (tid >= total_elements) {
            return;
        }

        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto row_idx = tid / in2_size;  // 行索引
        auto col_idx = tid % in2_size;  // 列索引
        output_ptr[tid] = input1_ptr[row_idx] * input2_ptr[col_idx];
    }

    template<typename DATA_TYPE>
    __global__ void mul_cuda_impl(std::int32_t in1_size, std::int32_t in2_size,
        const void* in1, const void* in2, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= in1_size) {
            return;
        }
        
        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = input1_ptr[tid] * input2_ptr[tid % in2_size];
    }

    template<typename DATA_TYPE>
    __global__ void mul_with_constant_value_cuda_impl(std::int32_t size, float value, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = input_ptr[tid] * static_cast<DATA_TYPE>(value);
    }

    template<typename DATA_TYPE>
    __global__ void matmul_cuda_impl(std::int32_t batch_size, std::int32_t batch_size_2,
        std::int32_t in1_n_rows, std::int32_t in1_n_cols, std::int32_t in2_n_cols,
        const void* in1, const void* in2, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto n_batch_elements = in1_n_rows * in2_n_cols;
        auto total_elements = batch_size * n_batch_elements;

        if (tid >= total_elements) {
            return;
        }

        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto batch_idx = tid / n_batch_elements;
        auto tid_in_batch = tid % n_batch_elements;
        auto n_rows = tid_in_batch / in2_n_cols;
        auto n_cols = tid_in_batch % in2_n_cols;

        float sum = 0.0f;
        for (std::int32_t i = 0; i < in1_n_cols; ++i) {
            auto in1_offset = batch_idx * (in1_n_rows * in1_n_cols) + n_rows * in1_n_cols + i;

            // in2的索引：如果batch_size_2为1，则所有batch共享同一个in2矩阵；否则每个batch使用对应的in2矩阵
            auto in2_batch_idx = (batch_size_2 == 1) ? 0 : batch_idx;
            auto in2_offset = in2_batch_idx * (in1_n_cols * in2_n_cols) + i * in2_n_cols + n_cols;

            sum += static_cast<float>(input1_ptr[in1_offset] * input2_ptr[in2_offset]);
        }
        output_ptr[tid] = static_cast<DATA_TYPE>(sum);
    }

    template<typename DATA_TYPE>
    __global__ void transpose_2d_cuda_impl(std::int32_t n_rows, std::int32_t n_cols, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto row_idx = tid / n_cols;
        auto col_idx = tid % n_cols;

        output_ptr[col_idx * n_rows + row_idx] = input_ptr[row_idx * n_cols + col_idx];
    }

    template<typename DATA_TYPE>
    __global__ void transpose_3d_cuda_impl(std::int32_t n_first_dim, std::int32_t n_rows, std::int32_t n_cols,
        std::int32_t dim0, std::int32_t dim1, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = n_first_dim * n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        // 计算输入坐标
        auto elements_per_slice = n_rows * n_cols;
        decltype(tid) input_coords[3] = {
            tid / elements_per_slice,                    // 第一维
            (tid % elements_per_slice) / n_cols,         // 第二维
            tid % n_cols                                 // 第三维
        };

        // 输入维度
        int input_dims[3] = {n_first_dim, n_rows, n_cols};

        // 输出坐标初始化为输入坐标
        decltype(tid) output_coords[3] = {input_coords[0], input_coords[1], input_coords[2]};

        // 交换指定的两个维度
        output_coords[dim0] = input_coords[dim1];
        output_coords[dim1] = input_coords[dim0];

        // 输出维度（交换对应的维度大小）
        int output_dims[3] = {input_dims[0], input_dims[1], input_dims[2]};
        output_dims[dim0] = input_dims[dim1];
        output_dims[dim1] = input_dims[dim0];

        // 计算输出索引
        decltype(tid) output_index = output_coords[0] * (output_dims[1] * output_dims[2]) +
                          output_coords[1] * output_dims[2] +
                          output_coords[2];

        output_ptr[output_index] = input_ptr[tid];
    }

    template<typename DATA_TYPE>
    __global__ void transpose_4d_cuda_impl(std::int32_t n_first_dim, std::int32_t n_second_dim,
        std::int32_t n_rows, std::int32_t n_cols,
        std::int32_t dim0, std::int32_t dim1, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = n_first_dim * n_second_dim * n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        // 计算输入张量中的四维坐标
        auto elements_per_3d = n_second_dim * n_rows * n_cols;
        auto elements_per_2d = n_rows * n_cols;
        auto elements_per_row = n_cols;

        auto first_idx = tid / elements_per_3d;
        auto residual_3d = tid % elements_per_3d;
        auto second_idx = residual_3d / elements_per_2d;
        auto residual_2d = residual_3d % elements_per_2d;
        auto row_idx = residual_2d / elements_per_row;
        auto col_idx = residual_2d % elements_per_row;

        decltype(tid) input_coords[4] = {
            first_idx,
            second_idx,
            row_idx,
            col_idx
        };

        // 输入维度
        int input_dims[4] = {n_first_dim, n_second_dim, n_rows, n_cols};

        // 输出坐标初始化为输入坐标
        decltype(tid) output_coords[4] = {input_coords[0], input_coords[1], input_coords[2], input_coords[3]};

        // 交换指定的两个维度
        output_coords[dim0] = input_coords[dim1];
        output_coords[dim1] = input_coords[dim0];

        // 输出维度（交换对应的维度大小）
        int output_dims[4] = {input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
        output_dims[dim0] = input_dims[dim1];
        output_dims[dim1] = input_dims[dim0];

        // 计算输出索引
        decltype(tid) output_index = output_coords[0] * (output_dims[1] * output_dims[2] * output_dims[3]) +
                          output_coords[1] * (output_dims[2] * output_dims[3]) +
                          output_coords[2] * output_dims[3] +
                          output_coords[3];

        output_ptr[output_index] = input_ptr[tid];
    }

    template<typename DATA_TYPE>
    __global__ void transpose_5d_cuda_impl(std::int32_t batch_size,
        std::int32_t n_first_dim, std::int32_t n_second_dim,
        std::int32_t n_rows, std::int32_t n_cols,
        std::int32_t dim0, std::int32_t dim1, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = batch_size * n_first_dim * n_second_dim * n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        // 计算输入张量中的四维坐标
        auto elements_per_4d = n_first_dim * n_second_dim * n_rows * n_cols;
        auto elements_per_3d = n_second_dim * n_rows * n_cols;
        auto elements_per_2d = n_rows * n_cols;
        auto elements_per_row = n_cols;

        auto batch_idx = tid / elements_per_4d;
        auto residual_4d = tid % elements_per_4d;
        auto first_idx = residual_4d / elements_per_3d;
        auto residual_3d = residual_4d % elements_per_3d;
        auto second_idx = residual_3d / elements_per_2d;
        auto residual_2d = residual_3d % elements_per_2d;
        auto row_idx = residual_2d / elements_per_row;
        auto col_idx = residual_2d % elements_per_row;

        decltype(tid) input_coords[5] = {
            batch_idx,
            first_idx,
            second_idx,
            row_idx,
            col_idx
        };

        // 输入维度
        int input_dims[5] = {batch_size, n_first_dim, n_second_dim, n_rows, n_cols};

        // 输出坐标初始化为输入坐标
        decltype(tid) output_coords[5] = {
            input_coords[0], input_coords[1], input_coords[2], input_coords[3], input_coords[4]
        };

        // 交换指定的两个维度
        output_coords[dim0] = input_coords[dim1];
        output_coords[dim1] = input_coords[dim0];

        // 输出维度（交换对应的维度大小）
        int output_dims[5] = {
            input_dims[0], input_dims[1], input_dims[2], input_dims[3], input_dims[4]
        };
        output_dims[dim0] = input_dims[dim1];
        output_dims[dim1] = input_dims[dim0];

        // 计算输出索引
        decltype(tid) output_index = output_coords[0] * (output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4]) +
                          output_coords[1] * (output_dims[2] * output_dims[3] * output_dims[4]) +
                          output_coords[2] * (output_dims[3] * output_dims[4]) +
                          output_coords[3] * output_dims[4] +
                          output_coords[4];

        output_ptr[output_index] = input_ptr[tid];
    }

    template<typename DATA_TYPE>
    __global__ void embedding_cuda_impl(std::int32_t embedding_dim, std::int32_t vocab_size,
        std::int32_t batch_size, std::int32_t seq_length, const void* embedding, const std::int32_t* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = batch_size * seq_length * vocab_size;
        if (tid >= total_elements) {
            return;
        }

        const auto* embedding_ptr = static_cast<const DATA_TYPE*>(embedding);
        auto* out_ptr = static_cast<DATA_TYPE*>(out);

        auto batch_idx = tid / (seq_length * vocab_size);
        auto residual = tid % (seq_length * vocab_size);
        auto seq_idx = residual / vocab_size;
        auto dim_idx = residual % vocab_size;

        auto word_id = in[batch_idx * seq_length + seq_idx];
        if (word_id < 0 || word_id >= embedding_dim) {
            out_ptr[tid] = static_cast<DATA_TYPE>(0);
            return;
        }

        auto embedding_offset = word_id * vocab_size + dim_idx;
        out_ptr[tid] = embedding_ptr[embedding_offset];
    }

    template<typename DATA_TYPE>
    __global__ void repeat_kv_cuda_impl(std::size_t n_rep, std::int32_t batch_size, std::int32_t n_kv_heads,
        std::int32_t n_rows, std::int32_t n_cols,
        const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = batch_size * n_kv_heads * n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        const auto* in_ptr = static_cast<const DATA_TYPE*>(in);
        auto* out_ptr = static_cast<DATA_TYPE*>(out);

        auto elements_per_3d = n_kv_heads * n_rows * n_cols;
        auto elements_per_2d = n_rows * n_cols;
        auto elements_per_row = n_cols;

        auto batch_idx = tid / elements_per_3d;
        auto residual_3d = tid % elements_per_3d;
        auto kv_head_idx = residual_3d / elements_per_2d;
        auto residual_2d = residual_3d % elements_per_2d;
        auto seq_idx = residual_2d / elements_per_row;
        auto hidden_size_idx = residual_2d % elements_per_row;

        for (std::size_t i = 0; i < n_rep; ++i) {
            auto in_idx = batch_idx * n_kv_heads * n_rows * n_cols +
                    kv_head_idx * n_rows * n_cols +
                    seq_idx * n_cols + hidden_size_idx;
            auto out_idx = batch_idx * n_rep * n_kv_heads * n_rows * n_cols +
                    (kv_head_idx * n_rep + i) * (n_rows * n_cols) +
                    seq_idx * n_cols + hidden_size_idx;
            out_ptr[out_idx] = in_ptr[in_idx];
            
        }
    }

    template<typename DATA_TYPE>
    __global__ void softmax_cuda_impl(std::int32_t batch_size, std::int32_t n_rows, std::int32_t n_cols,
    const float* t, const void* in, void* out) {

        // 每个线程处理一行
        auto row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx >= batch_size * n_rows) {
            return;
        }

        auto batch_idx = row_idx / n_rows;
        auto row_in_batch = row_idx % n_rows;

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        // 计算行内最大值
        float max_val = -INFINITY;
        for (std::int32_t i = 0; i < n_cols; ++i) {
            auto idx = batch_idx * (n_rows * n_cols) + row_in_batch * n_cols + i;
            auto val = static_cast<float>(input_ptr[idx]);
            if (t != nullptr) val /= *t;
            if (val > max_val) max_val = val;
        }

        // 计算指数和
        float exp_sum = 0.0f;
        for (std::int32_t i = 0; i < n_cols; ++i) {
            auto idx = batch_idx * (n_rows * n_cols) + row_in_batch * n_cols + i;
            auto val = static_cast<float>(input_ptr[idx]);
            if (t != nullptr) val /= *t;
            float exp_val = expf(val - max_val);
            exp_sum += exp_val;
        }

        // 归一化
        for (std::int32_t i = 0; i < n_cols; ++i) {
            auto idx = batch_idx * (n_rows * n_cols) + row_in_batch * n_cols + i;
            auto val = static_cast<float>(input_ptr[idx]);
            if (t != nullptr) val /= *t;
            float exp_val = expf(val - max_val);
            output_ptr[idx] = static_cast<DATA_TYPE>(exp_val / exp_sum);
        }
    }


    template<typename DATA_TYPE>
    __global__ void silu_cuda_impl(std::int32_t size, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        if (std::abs(static_cast<float>(input_ptr[tid])) >= 1e-2f) {
            float z = expf(-static_cast<float>(input_ptr[tid]));
            output_ptr[tid] = input_ptr[tid] * static_cast<DATA_TYPE>(1.0f / (1.0f + z));
        } else {
            float z = expf(static_cast<float>(input_ptr[tid]));
            output_ptr[tid] = input_ptr[tid] * static_cast<DATA_TYPE>(z / (1.0f + z));
        }
    }

    template<typename DATA_TYPE>
    __global__ void rms_norm_cuda_impl(std::int32_t batch_size, std::int32_t n_rows, std::int32_t n_cols,
        const void* in, void* out, float epsilon) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto total_elements = batch_size * n_rows * n_cols;
        if (tid >= total_elements) {
            return;
        }

        auto batch_idx = tid / (n_rows * n_cols);
        auto tid_in_batch = tid % (n_rows * n_cols);
        auto row_idx = tid_in_batch / n_cols;

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        // 每行的平方和平均数
        float squared_sum = 0.0f;
        for (auto i = 0; i < n_cols; ++i) {
            auto idx = batch_idx * (n_rows * n_cols) + row_idx * n_cols + i;
            squared_sum += static_cast<float>(input_ptr[idx]) * static_cast<float>(input_ptr[idx]);
        }
        float mean_squared = squared_sum / static_cast<float>(n_cols);
        float rms = sqrtf(mean_squared + epsilon);

        // 计算RMS Norm
        output_ptr[tid] = static_cast<DATA_TYPE>(static_cast<float>(input_ptr[tid]) / rms);
    }

    template<typename DATA_TYPE>
    __global__ void vec_or_matrix_argmax_cuda_impl(std::int32_t n_rows, std::int32_t n_cols,
        const void* in, std::int32_t* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n_rows) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);

        auto row_start = tid * n_cols;
        std::int32_t max_idx = 0;
        auto max_val = input_ptr[row_start];

        for (std::int32_t i = 1; i < n_cols; ++i) {
            auto cur_idx = row_start + i;
            if (input_ptr[cur_idx] > max_val) {
                max_val = input_ptr[cur_idx];
                max_idx = i;
            }
        }

        out[tid] = max_idx;
    }

    template<typename DATA_TYPE>
    __global__ void rotate_half_cuda_impl(std::int32_t n_rows, std::int32_t n_cols, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n_rows * n_cols) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto half_cols = n_cols / 2;

        auto row_idx = tid / n_cols;
        auto col_idx = tid % n_cols;

        if (col_idx < half_cols) {
            // output左半边 = -input 右半边
            auto offset = col_idx + half_cols;
            output_ptr[row_idx * n_cols + col_idx] = -input_ptr[row_idx * n_cols + offset];
        } else {
            // output右半边 = input 左半边
            auto offset = col_idx - half_cols;
            output_ptr[row_idx * n_cols + col_idx] = input_ptr[row_idx * n_cols + offset];
        }
    }

    template<typename DATA_TYPE>
    __global__ void cosine_cuda_impl(std::int32_t size, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = static_cast<DATA_TYPE>(cosf(static_cast<float>(input_ptr[tid])));
    }

    template<typename DATA_TYPE>
    __global__ void sine_cuda_impl(std::int32_t size, const void* in, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        const auto* input_ptr = static_cast<const DATA_TYPE*>(in);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        output_ptr[tid] = static_cast<DATA_TYPE>(sinf(static_cast<float>(input_ptr[tid])));
    }

    template<typename DATA_TYPE>
    __global__ void concat_by_col_wise_cuda_impl(std::int32_t n_rows, std::int32_t n_cols_1, std::int32_t n_cols_2,
        const void* in1, const void* in2, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n_rows * (n_cols_1 + n_cols_2)) {
            return;
        }

        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto out_row_idx = tid / (n_cols_1 + n_cols_2);
        auto out_col_idx = tid % (n_cols_1 + n_cols_2);

        if (out_col_idx < n_cols_1) {
            output_ptr[tid] = input1_ptr[out_row_idx * n_cols_1 + out_col_idx];
        } else {
            output_ptr[tid] = input2_ptr[out_row_idx * n_cols_2 + (out_col_idx - n_cols_1)];
        }
    }

    template<typename DATA_TYPE>
    __global__ void concat_by_row_wise_cuda_impl(std::int32_t batch_size,
        std::int32_t n_rows_1, std::int32_t n_rows_2, std::int32_t n_cols,
        const void* in1, const void* in2, void* out) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= batch_size * (n_rows_1 + n_rows_2) * n_cols) {
            return;
        }

        const auto* input1_ptr = static_cast<const DATA_TYPE*>(in1);
        const auto* input2_ptr = static_cast<const DATA_TYPE*>(in2);
        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto elements_per_batch = (n_rows_1 + n_rows_2) * n_cols;
        auto batch_idx = tid / elements_per_batch;
        auto residual = tid % elements_per_batch;
        auto out_row_idx = residual / n_cols;
        auto out_col_idx = residual % n_cols;

        if (out_row_idx < n_rows_1) {
            output_ptr[tid] = input1_ptr[batch_idx * n_rows_1 * n_cols + out_row_idx * n_cols + out_col_idx];
        } else {
            output_ptr[tid] = input2_ptr[batch_idx * n_rows_2 * n_cols + (out_row_idx - n_rows_1) * n_cols + out_col_idx];
        }
    }

    template<typename DATA_TYPE>
    __global__ void causal_mask_cuda_impl(std::int32_t l, std::int32_t s, void* out) {
        static_assert(std::is_same_v<DATA_TYPE, __nv_bfloat16> || std::is_same_v<DATA_TYPE, float> ||
            std::is_same_v<DATA_TYPE, std::int32_t> || std::is_same_v<DATA_TYPE, std::int8_t> ||
            std::is_same_v<DATA_TYPE, std::uint8_t>);

        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= l * s) {
            return;
        }

        auto* output_ptr = static_cast<DATA_TYPE*>(out);

        auto row_idx = tid / s;
        auto col_idx = tid % s;

        // 上三角均为-inf，下三角均为0
        if (col_idx <= row_idx) {
            output_ptr[tid] = static_cast<DATA_TYPE>(0);
        } else {
            if constexpr (std::is_same_v<DATA_TYPE, __nv_bfloat16>) {
                output_ptr[tid] = __float2bfloat16(-INFINITY);
            } else if constexpr (std::is_same_v<DATA_TYPE, float>) {
                output_ptr[tid] = -INFINITY;
            } else if constexpr (std::is_same_v<DATA_TYPE, std::int32_t>) {
                output_ptr[tid] = INT32_MIN;
            }
        }
    }

    template <typename DATA_TYPE>
    __global__ void multinomial_cuda_impl(const void* x,
        std::int32_t* output,
        const int* row_end_pos,  // 每行的有效结束位置，nullptr表示使用num_cols
        std::size_t num_rows,
        std::size_t num_cols,
        std::size_t num_samples,
        unsigned long long seed) {

        // 每个线程块处理一行，每个线程处理该行的一个样本
        std::size_t row = blockIdx.x;
        std::size_t sample_idx = threadIdx.x;

        if (row >= num_rows || sample_idx >= num_samples) return;

        // 为每个线程初始化随机数生成器
        curandState state;
        curand_init(seed, row * num_samples + sample_idx, 0, &state);

        // 确定当前行的有效结束位置
        std::size_t end_pos = num_cols;
        if (row_end_pos != nullptr) {
            end_pos = row_end_pos[row];
        }

        // 如果没有有效token，填充-1
        if (end_pos == 0) {
            output[row * num_samples + sample_idx] = -1;
            return;
        }

        // 使用共享内存存储累积概率
        extern __shared__ float s_cumulative_probs[];
        float* cumulative_probs = s_cumulative_probs;

        const auto* x_ptr = static_cast<const DATA_TYPE*>(x);

        // 线程0负责计算累积概率
        if (threadIdx.x == 0) {
            // 提取当前行的概率并计算总和
            float sum = 0.0f;
            for (std::size_t j = 0; j < end_pos; ++j) {
                sum += static_cast<float>(x_ptr[row * num_cols + j]);
            }

            // 归一化概率
            if (sum <= 0.0f) {
                // 如果概率和为0或负数，使用均匀分布
                float uniform_prob = 1.0f / static_cast<float>(end_pos);
                float cumulative = 0.0f;
                for (std::size_t i = 0; i < end_pos; ++i) {
                    cumulative += uniform_prob;
                    cumulative_probs[i] = cumulative;
                }
            } else {
                float cumulative = 0.0f;
                for (std::size_t i = 0; i < end_pos; ++i) {
                    cumulative += static_cast<float>(x_ptr[row * num_cols + i]) / sum;
                    cumulative_probs[i] = cumulative;
                }
            }

            // 确保最后一个累积概率为1.0
            cumulative_probs[end_pos - 1] = 1.0f;
        }

        __syncthreads();  // 等待累积概率计算完成

        // 生成随机数并进行采样
        float r = curand_uniform(&state);
        std::size_t selected = 0;

        // 找到第一个累积概率大于r的索引
        while (selected < end_pos - 1 && r > cumulative_probs[selected]) {
            ++selected;
        }

        output[row * num_samples + sample_idx] = static_cast<int32_t>(selected);
    }
}

#endif //FG42_KERNEL_IMPL_H