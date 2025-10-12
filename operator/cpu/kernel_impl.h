//
// Created by B777B2056-2 on 2025/9/16.
//

#ifndef FG42_KERNEL_IMPL_H
#define FG42_KERNEL_IMPL_H
#include <optional>
#include "Eigen/Core"
#include "unsupported/Eigen/MatrixFunctions"
#include "unsupported/Eigen/CXX11/Tensor"
#include "operator/cpu/kernel.h"

namespace fg42::kernel {
    template <typename DATA_TYPE>
    void add_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        RowVector input_vec1(static_cast<DATA_TYPE*>(input1.raw_ptr()), input1.size());
        RowVector input_vec2(static_cast<DATA_TYPE*>(input2.raw_ptr()), input2.size());
        RowVector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec1 + input_vec2;
    }

    template <typename DATA_TYPE>
    void boardcast_add_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;

        const auto& shape1 = input1.shape();
        const auto& shape2 = input2.shape();

        // 获取矩阵维度
        std::size_t col_size = shape1[shape1.size() - 1];

        // 计算主批量大小
        std::size_t batch_size = 1;
        for (std::size_t i = 0; i < shape1.size() - 1; ++i) {
            batch_size *= shape1[i];
        }

        // 检查rhs批量大小
        std::size_t rhs_batch_size = 1;
        for (std::size_t i = 0; i < shape2.size() - 1; ++i) {
            rhs_batch_size *= shape2[i];
        }

        auto* lhs_data = static_cast<DATA_TYPE*>(input1.raw_ptr());
        auto* rhs_data = static_cast<DATA_TYPE*>(input2.raw_ptr());
        auto* out_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        std::size_t rhs_batch_idx = 0;
        for (std::size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx, ++rhs_batch_idx) {
            if (rhs_batch_idx == rhs_batch_size) {
                rhs_batch_idx = 0;
            }
            RowVector lhs_map(lhs_data + batch_idx * col_size, col_size);
            RowVector rhs_map(rhs_data + rhs_batch_idx * col_size, col_size);
            RowVector out_map(out_data + batch_idx * col_size, col_size);
            out_map = lhs_map + rhs_map;
        }
    }

    template <typename DATA_TYPE>
    void negated_kernel_cpu_impl(const Tensor& input, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        RowVector input_vec(static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());
        RowVector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = -input_vec;
    }

    template <typename DATA_TYPE>
    void vec_outer_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using Vector = Eigen::Map<Eigen::Vector<DATA_TYPE, Eigen::Dynamic>>;
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Vector input_vec1(static_cast<DATA_TYPE*>(input1.raw_ptr()), input1.size());
        RowVector input_vec2(static_cast<DATA_TYPE*>(input2.raw_ptr()), input2.size());
        Matrix output_m(static_cast<DATA_TYPE*>(output.raw_ptr()), output.shape().at(0), output.shape().at(1));
        output_m = input_vec1 * input_vec2;
    }

    template <typename DATA_TYPE>
    void mul_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        RowVector input_vec1(static_cast<DATA_TYPE*>(input1.raw_ptr()), input1.size());
        RowVector input_vec2(static_cast<DATA_TYPE*>(input2.raw_ptr()), input2.size());
        RowVector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec1.array() * input_vec2.array();
    }

    template <typename DATA_TYPE>
    void boardcast_mul_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;

        const auto& shape1 = input1.shape();
        const auto& shape2 = input2.shape();

        // 获取矩阵维度
        std::size_t vector_size = shape1.back();

        // 计算主批量大小
        std::size_t batch_size = 1;
        for (std::size_t i = 0; i < shape1.size() - 1; ++i) {
            batch_size *= shape1[i];
        }

        // 检查rhs批量大小
        std::size_t rhs_batch_size = 1;
        for (std::size_t i = 0; i < shape2.size() - 1; ++i) {
            rhs_batch_size *= shape2[i];
        }

        auto* lhs_data = static_cast<DATA_TYPE*>(input1.raw_ptr());
        auto* rhs_data = static_cast<DATA_TYPE*>(input2.raw_ptr());
        auto* out_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        std::size_t rhs_batch_idx = 0;
        for (std::size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx, ++rhs_batch_idx) {
            if (rhs_batch_idx == rhs_batch_size) {
                rhs_batch_idx = 0;
            }
            RowVector lhs_map(lhs_data + batch_idx * vector_size, vector_size);
            RowVector rhs_map(rhs_data + rhs_batch_idx * vector_size, vector_size);
            RowVector out_map(out_data + batch_idx * vector_size, vector_size);
            out_map = lhs_map.array() * rhs_map.array();
        }
    }

    template <typename DATA_TYPE>
    void mul_with_constant_kernel_cpu_impl(const Tensor& input2, Tensor& output, float value) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        RowVector input_vec2(static_cast<DATA_TYPE*>(input2.raw_ptr()), input2.size());
        RowVector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec2.array() * static_cast<DATA_TYPE>(value);
    }

    template <typename DATA_TYPE>
    void matmul_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output,
                                std::size_t n_rows_1, std::size_t n_cols_1, std::size_t n_cols_2) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Matrix input_m1(static_cast<DATA_TYPE*>(input1.raw_ptr()), n_rows_1, n_cols_1);
        Matrix input_m2(static_cast<DATA_TYPE*>(input2.raw_ptr()), n_cols_1, n_cols_2);
        Matrix output_m(static_cast<DATA_TYPE*>(output.raw_ptr()), n_rows_1, n_cols_2);
        output_m = input_m1 * input_m2;
    }

    template <typename DATA_TYPE>
    void batch_matmul_kernel_cpu_impl(const Tensor& input1, const Tensor& input2, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        const auto& shape1 = input1.shape();
        const auto& shape2 = input2.shape();

        // 获取矩阵维度
        std::size_t m = shape1[shape1.size() - 2];
        std::size_t k = shape1[shape1.size() - 1];
        std::size_t n = shape2[shape2.size() - 1];

        // 计算主批量大小
        std::size_t batch_size = 1;
        for (std::size_t i = 0; i < shape1.size() - 2; ++i) {
            batch_size *= shape1[i];
        }

        // 检查rhs批量大小
        std::size_t rhs_batch_size = 1;
        for (std::size_t i = 0; i < shape2.size() - 2; ++i) {
            rhs_batch_size *= shape2[i];
        }

        auto* lhs_data = static_cast<DATA_TYPE*>(input1.raw_ptr());
        auto* rhs_data = static_cast<DATA_TYPE*>(input2.raw_ptr());
        auto* out_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        // 每个矩阵的大小
        std::size_t lhs_matrix_size = m * k;
        std::size_t rhs_matrix_size = k * n;
        std::size_t out_matrix_size = m * n;

        std::size_t rhs_batch_idx = 0;
        for (std::size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx, ++rhs_batch_idx) {
            if (rhs_batch_idx == rhs_batch_size) {
                rhs_batch_idx = 0;
            }
            Matrix lhs_map(lhs_data + batch_idx * lhs_matrix_size, m, k);
            Matrix rhs_map(rhs_data + rhs_batch_idx * rhs_matrix_size, k, n);
            Matrix out_map(out_data + batch_idx * out_matrix_size, m, n);
            out_map = lhs_map * rhs_map;
        }
    }

    template <typename DATA_TYPE>
    void matrix_transpose_kernel_cpu_impl(const Tensor& input, Tensor& output,
        std::size_t n_rows, std::size_t n_cols) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Matrix input_m(static_cast<DATA_TYPE*>(input.raw_ptr()), n_rows, n_cols);
        Matrix output_m(static_cast<DATA_TYPE*>(output.raw_ptr()), n_cols, n_rows);
        output_m = input_m.transpose();
    }

    // 高维张量转置实现
    template <typename DATA_TYPE, int N>
    void transpose_xd_tensor_kernel_cpu_impl(const Tensor& input, Tensor& output, std::size_t dim0, std::size_t dim1) {
        Eigen::array<Eigen::Index, N> input_dims;
        Eigen::array<Eigen::Index, N> output_dims;

        for (std::size_t i = 0; i < N; ++i) {
            input_dims[i] = static_cast<Eigen::Index>(input.shape()[i]);
            output_dims[i] = static_cast<Eigen::Index>(output.shape()[i]);
        }

        auto* input_data = static_cast<DATA_TYPE*>(input.raw_ptr());
        auto* output_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        Eigen::TensorMap<const Eigen::Tensor<DATA_TYPE, N, Eigen::RowMajor>>
            input_tensor(input_data, input_dims);

        Eigen::TensorMap<Eigen::Tensor<DATA_TYPE, N, Eigen::RowMajor>>
            output_tensor(output_data, output_dims);

        // 构建转置维度映射
        Eigen::array<Eigen::Index, N> shuffle;
        for (int i = 0; i < N; ++i) shuffle[i] = i;
        std::swap(shuffle[dim0], shuffle[dim1]);

        output_tensor = input_tensor.shuffle(shuffle);
    }

    // 3D 张量转置
    template <typename DATA_TYPE>
    void transpose_3d_tensor_kernel_cpu_impl(const Tensor& input, Tensor& output, std::size_t dim0, std::size_t dim1) {
        transpose_xd_tensor_kernel_cpu_impl<DATA_TYPE, 3>(input, output, dim0, dim1);
    }

    // 4D 张量转置
    template <typename DATA_TYPE>
    void transpose_4d_tensor_kernel_cpu_impl(const Tensor& input, Tensor& output,
                            std::size_t dim0, std::size_t dim1) {
        transpose_xd_tensor_kernel_cpu_impl<DATA_TYPE, 4>(input, output, dim0, dim1);
    }

    // 5D 张量转置
    template <typename DATA_TYPE>
    void transpose_5d_tensor_kernel_cpu_impl(const Tensor& input, Tensor& output,
                            std::size_t dim0, std::size_t dim1) {
        transpose_xd_tensor_kernel_cpu_impl<DATA_TYPE, 5>(input, output, dim0, dim1);
    }

    template <typename DATA_TYPE>
    void embedding_kernel_cpu_impl(const Tensor* weight_tensor, const Tensor& input_tensor, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;

        const auto embedding_dim = weight_tensor->shape().at(1);
        const auto vocab_size = weight_tensor->shape().at(0);

        std::size_t batch_size = input_tensor.shape().at(0);
        std::size_t seq_length = input_tensor.shape().at(1);

        auto* weight_data = static_cast<DATA_TYPE*>(weight_tensor->raw_ptr());
        auto* input_data = static_cast<const std::int32_t*>(input_tensor.raw_ptr());
        auto* output_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        for (std::size_t i = 0; i < batch_size * seq_length; ++i) {
            std::int32_t word_id = input_data[i];

            if (word_id >= 0 && word_id < vocab_size) {
                // 使用 Eigen 的 Map 进行向量拷贝
                RowVector dst_vec(output_data + i * embedding_dim, embedding_dim);
                RowVector src_vec(weight_data + word_id * embedding_dim, embedding_dim);
                dst_vec = src_vec;
            } else {
                // OOV 处理
                RowVector dst_vec(output_data + i * embedding_dim, embedding_dim);
                dst_vec.setZero();
            }
        }
    }

    template <typename DATA_TYPE>
    void repeat_kv_cpu_impl(const Tensor& x, Tensor& output, std::size_t n_rep) {
        auto batch = x.shape().at(0);
        auto num_kv_heads = x.shape().at(1);
        auto seq_len = x.shape().at(2);
        auto head_dim = x.shape().at(3);

        std::size_t head_size = seq_len * head_dim;
        std::size_t input_batch_stride = num_kv_heads * head_size;
        std::size_t output_batch_stride = num_kv_heads * n_rep * head_size;

        auto* x_data = static_cast<DATA_TYPE*>(x.raw_ptr());
        auto* out_data = static_cast<DATA_TYPE*>(output.raw_ptr());

        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        for (std::size_t b = 0; b < batch; ++b) {
            // 将输入重塑为3D张量: [num_kv_heads_, seq_len, head_dim_]
            Matrix input_matrix(x_data + b * input_batch_stride, num_kv_heads, head_size);
            Matrix output_matrix( out_data + b * output_batch_stride, num_kv_heads * n_rep, head_size);

            for (std::size_t k = 0; k < num_kv_heads; ++k) {
                // 获取输入的第k行（对应第k个head）
                auto input_row = input_matrix.row(k);

                // 复制到输出的n_rep个位置
                for (std::size_t r = 0; r < n_rep; ++r) {
                    output_matrix.row(k * n_rep + r) = input_row;
                }
            }
        }
    }

    template <typename DATA_TYPE>
    void vec_softmax_kernel_cpu_impl(const Tensor& input, Tensor& output, std::optional<float> t) {
        using Vector = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1>>;
        Vector input_vec(static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());
        Vector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());

        // 如果传入系数，需除以该系数
        if (t.has_value()) {
            input_vec = input_vec / static_cast<DATA_TYPE>(t.value());
        }

        // 计算输入向量的最大值
        DATA_TYPE max_val = input_vec.maxCoeff();

        // 减去最大值后进行指数运算
        auto exp_vec = (input_vec.array() - max_val).exp();

        // 计算指数和
        DATA_TYPE exp_sum = exp_vec.sum();

        // 归一化得到softmax概率
        output_vec = exp_vec / exp_sum;
    }

    template <typename DATA_TYPE>
    void matrix_softmax_kernel_cpu_impl(const Tensor& input, Tensor& output, std::size_t n_rows, std::size_t n_cols,
        std::optional<float> t) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Matrix input_matrix(static_cast<DATA_TYPE*>(input.raw_ptr()), n_rows, n_cols);
        Matrix output_matrix(static_cast<DATA_TYPE*>(output.raw_ptr()), n_rows, n_cols);

        // 如果传入系数，需除以该系数
        if (t.has_value() && t.value() != 0) {
            input_matrix = input_matrix / static_cast<DATA_TYPE>(t.value());
        }

        // 计算每行的最大值
        Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1> max_vals = input_matrix.rowwise().maxCoeff();

        // 减去最大值
        auto shifted = input_matrix.array().colwise() - max_vals.array();

        // 计算指数
        auto exp_matrix = shifted.exp();

        // 计算每行的指数和
        Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1> exp_sums = exp_matrix.rowwise().sum();

        output_matrix = exp_matrix.array().colwise() / exp_sums.array();
    }

    template <typename DATA_TYPE>
    void vec_silu_kernel_cpu_impl(const Tensor& input, Tensor& output) {
        using RowVector = Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>>;
        RowVector input_vec(static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());
        RowVector output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec.unaryExpr([](DATA_TYPE x) -> DATA_TYPE {
            if (x >= 0) {
                float z = std::exp(-float(x));
                return x * static_cast<DATA_TYPE>(1.0f / (1.0f + z));
            } else {
                float z = std::exp(float(x));
                return x * static_cast<DATA_TYPE>(z / (1.0f + z));
            }
        });
    }

    template <typename DATA_TYPE>
    void rme_norm_kernel_cpu_impl(const Tensor& input, Tensor& output, float eps) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        // 获取最后一个维度的大小（向量长度）
        std::size_t vector_size = input.shape().back();

        // 计算需要处理的行向量数量
        std::size_t num_vectors = input.size() / vector_size;

        Matrix input_matrix(static_cast<DATA_TYPE*>(input.raw_ptr()), num_vectors, vector_size);
        Matrix output_matrix(static_cast<DATA_TYPE*>(output.raw_ptr()), num_vectors, vector_size);

        Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1> variance =
            input_matrix.array().square().rowwise().mean();

        Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1> rsqrt_variance =
            (variance.array() + DATA_TYPE(eps)).rsqrt();

        for (std::size_t i = 0; i < num_vectors; ++i) {
            output_matrix.row(i) = input_matrix.row(i) * rsqrt_variance(i);
        }
    }

    template <typename DATA_TYPE>
    std::vector<std::pair<DATA_TYPE, std::int32_t>> sort_indexes_by_row(const Tensor& input) {
        std::size_t vector_size = input.shape().at(1);

        Eigen::Map<Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>> input_vec(
            static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());

        std::vector<std::pair<DATA_TYPE, std::int32_t>> pairs;
        for (std::size_t j = 0; j < vector_size; ++j) {
            pairs.emplace_back(input_vec(j), static_cast<std::int32_t>(j));
        }

        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        return pairs;
    }

    template <typename DATA_TYPE>
    void find_max_k_idx_in_row(const Tensor& input, Tensor& output) {
        std::size_t n = output.shape().at(1);

        Eigen::Map<Eigen::RowVector<std::int32_t, Eigen::Dynamic>> output_vec(
            static_cast<std::int32_t*>(output.raw_ptr()), output.size());

        // 对于单个行，按从大到小排序，找到排序后前n个元素的索引
        auto pairs = sort_indexes_by_row<DATA_TYPE>(input);
        for (std::size_t j = 0; j < n; ++j) {
            output_vec(j) = pairs[j].second;
        }
    }

    template <typename DATA_TYPE>
    void rotate_half_impl(const Tensor& input, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        Matrix x(static_cast<DATA_TYPE*>(input.raw_ptr()), input.shape()[0], input.shape()[1]);

        int cols = x.cols();
        int half_cols = cols / 2;

        Matrix result(static_cast<DATA_TYPE*>(output.raw_ptr()), x.rows(), cols);

        result.leftCols(half_cols) = -x.rightCols(half_cols);
        result.rightCols(half_cols) = x.leftCols(half_cols);
    }

    template <typename DATA_TYPE>
    void cosine_impl(const Tensor& input, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1>>;
        Matrix input_vec(static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());
        Matrix output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec.array().cos();
    }

    template <typename DATA_TYPE>
    void sine_impl(const Tensor& input, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, 1>>;
        Matrix input_vec(static_cast<DATA_TYPE*>(input.raw_ptr()), input.size());
        Matrix output_vec(static_cast<DATA_TYPE*>(output.raw_ptr()), output.size());
        output_vec = input_vec.array().sin();
    }

    // 按列拼接，即最后一个维度拼接
    template <typename DATA_TYPE>
    void concat_by_col_wise(const Tensor& x1, const Tensor& x2, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < x1.shape().size() - 1; ++i) {
            total_batch *= x1.shape().at(i);
        }

        Matrix m1(static_cast<DATA_TYPE*>(x1.raw_ptr()), total_batch, x1.shape().back());
        Matrix m2(static_cast<DATA_TYPE*>(x2.raw_ptr()), total_batch, x2.shape().back());
        Matrix output_m(static_cast<DATA_TYPE*>(output.raw_ptr()), total_batch, output.shape().back());
        output_m << m1, m2;
    }

    // 按行拼接，即最后两维拼接
    template <typename DATA_TYPE>
    void concat_by_row_wise(const Tensor& x1, const Tensor& x2, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        // 计算除最后两维外的批次大小
        std::size_t total_batch = 1;
        for (std::size_t i = 0; i < x1.shape().size() - 2; ++i) {
            total_batch *= x1.shape().at(i);
        }

        // 获取最后两维的大小
        std::size_t x1_rows = x1.shape().at(x1.shape().size() - 2);
        std::size_t x1_cols = x1.shape().back();
        std::size_t x2_rows = x2.shape().at(x2.shape().size() - 2);
        std::size_t x2_cols = x2.shape().back();

        // 按行拼接：将最后两维视为矩阵，在行方向拼接
        std::size_t x1_elements_per_matrix = x1_rows * x1_cols;
        std::size_t x2_elements_per_matrix = x2_rows * x2_cols;
        std::size_t output_elements_per_matrix = (x1_rows + x2_rows) * x1_cols;

        for (std::size_t batch = 0; batch < total_batch; ++batch) {
            DATA_TYPE* x1_ptr = static_cast<DATA_TYPE*>(x1.raw_ptr()) + batch * x1_elements_per_matrix;
            DATA_TYPE* x2_ptr = static_cast<DATA_TYPE*>(x2.raw_ptr()) + batch * x2_elements_per_matrix;
            DATA_TYPE* output_ptr = static_cast<DATA_TYPE*>(output.raw_ptr()) + batch * output_elements_per_matrix;

            Matrix m1(x1_ptr, x1_rows, x1_cols);
            Matrix m2(x2_ptr, x2_rows, x2_cols);
            Matrix output_m(output_ptr, x1_rows + x2_rows, x1_cols);

            output_m << m1, m2;
        }
    }

    template <typename DATA_TYPE>
    void causal_mask_impl(std::size_t l, std::size_t s, Tensor& output) {
        using Matrix = Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Matrix output_matrix(static_cast<DATA_TYPE*>(output.raw_ptr()), l, s);
        output_matrix.setConstant(-std::numeric_limits<DATA_TYPE>::infinity());

        output_matrix.template triangularView<Eigen::Lower>() =
            Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(l, s);
    }

    template <typename DATA_TYPE>
    void multinomial_impl(const Tensor& x, Tensor& output,
        const std::function<std::size_t(std::size_t)>& row_end_pos) {
        std::size_t num_rows = x.shape().at(0);
        std::size_t num_cols = x.shape().at(1);
        std::size_t num_samples = output.shape().back();

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        Eigen::Map<Eigen::Matrix<DATA_TYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> probs_matrix(
            static_cast<DATA_TYPE*>(x.raw_ptr()), num_rows, num_cols);
        Eigen::Map<Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> samples(
            static_cast<std::int32_t*>(output.raw_ptr()), num_rows, num_samples);

        for (std::size_t row = 0; row < num_rows; ++row) {
            // 使用row_end_pos确定当前行的有效结束位置
            std::size_t end_pos = num_cols;
            if (row_end_pos) {
                end_pos = row_end_pos(row);
            }

            if (end_pos == 0) {
                // 如果没有有效token，填充-1或其他标记值
                for (int s = 0; s < num_samples; ++s) {
                    samples(row, s) = -1;
                }
                continue;
            }

            // 提取当前行的有效概率（前end_pos个元素）
            Eigen::RowVector<DATA_TYPE, Eigen::Dynamic> row_probs(end_pos);
            for (std::size_t j = 0; j < end_pos; ++j) {
                row_probs(j) = probs_matrix(row, j);
            }

            // 归一化概率
            float sum = row_probs.sum();
            if (std::islessequal(sum, 0.0f)) {
                // 如果概率和为0或负数，使用均匀分布
                row_probs = Eigen::RowVector<DATA_TYPE, Eigen::Dynamic>::Ones(end_pos) / static_cast<DATA_TYPE>(end_pos);
            } else {
                row_probs /= static_cast<DATA_TYPE>(sum);
            }

            // 计算累积概率
            Eigen::RowVector<float, Eigen::Dynamic> cumulative_probs(end_pos);
            float cumulative = 0.0;
            for (std::size_t i = 0; i < end_pos; ++i) {
                cumulative += row_probs(i);
                cumulative_probs(i) = static_cast<float>(cumulative);
            }

            // 确保最后一个累积概率为1.0（避免浮点误差）
            cumulative_probs(end_pos - 1) = 1.0f;

            // 采样
            for (std::size_t s = 0; s < num_samples; ++s) {
                float r = dis(gen);
                std::size_t selected = 0;

                // 找到第一个累积概率大于r的索引
                while (selected < end_pos - 1 && std::isgreater(r, cumulative_probs(selected))) {
                    ++selected;
                }
                samples(row, s) = static_cast<std::int32_t>(selected);
            }
        }
    }

    #define DATA_TYPE_SWITCH(ENUM_DATA_TYPE, TMPL_HANDLER, ...) \
    {   \
        switch (ENUM_DATA_TYPE) {    \
        case DataType::Int8:    \
            TMPL_HANDLER<std::int8_t>(__VA_ARGS__);   \
            break;  \
        case DataType::UInt8:   \
            TMPL_HANDLER<std::uint8_t>(__VA_ARGS__);   \
            break;  \
        case DataType::Int32:   \
            TMPL_HANDLER<std::int32_t>(__VA_ARGS__);   \
            break;  \
        case DataType::BF16:    \
            TMPL_HANDLER<Eigen::bfloat16>(__VA_ARGS__);   \
            break;  \
        case DataType::FP32:    \
            TMPL_HANDLER<float>(__VA_ARGS__);   \
            break;  \
        default:    \
            throw std::runtime_error("unsupported data type");  \
        }   \
    }
}

#endif //FG42_KERNEL_IMPL_H