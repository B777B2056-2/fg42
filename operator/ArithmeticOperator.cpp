//
// Created by B777B2056-2 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"
#include "operator/ArithmeticOperator.h"

namespace fg42::kernel {
    Tensor AddOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return vec_add_kernel_func(*input_tensors[0], *input_tensors[1], stream);
    }

    void AddOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        if (input_tensors[0]->empty() || input_tensors[1]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        if (input_tensors[0]->shape().size() < input_tensors[1]->shape().size()) {
            throw std::invalid_argument("Input tensor 1 shape size should equal or large than tensor 2");
        }
    }

    Tensor NegateOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return negate_kernel_func(*input_tensors[0], stream);
    }

    void NegateOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }
    }

    Tensor VecOuterOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return vec_outer_kernel_func(*input_tensors[0], *input_tensors[1], stream);
    }

    void VecOuterOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        if (input_tensors[0]->empty() || input_tensors[1]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        // 两个输入都必须是一维向量
        if (input_tensors[0]->shape().size() != 1 || input_tensors[1]->shape().size() != 1) {
            throw std::invalid_argument("Input tensor should be vector");
        }
    }

    Tensor MulOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return mul_kernel_func(*input_tensors[0], *input_tensors[1], stream);
    }

    void MulOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        if (input_tensors[0]->empty() || input_tensors[1]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        if (input_tensors[0]->shape().size() < input_tensors[1]->shape().size()) {
            throw std::invalid_argument("Input tensor 1 shape size should equal or large than tensor 2");
        }

        if (input_tensors[0]->shape().back() != input_tensors[1]->shape().back()) {
            throw std::invalid_argument("Input tensor 1 last dim should equal to tensor 2 last dim");
        }
    }

    MulWithConstantValueOperator::MulWithConstantValueOperator(float c) : c_(c) {}

    Tensor MulWithConstantValueOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return mul_with_constant_value_kernel_func(c_, *input_tensors[0], stream);
    }

    void MulWithConstantValueOperator::check(const std::vector<const Tensor*>& input_tensors) {
        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }

        if (input_tensors[0]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }
    }

    Tensor MatmulOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        const Tensor* x = input_tensors[0];
        const Tensor* y = input_tensors[1];

        return matrix_matmul_kernel_func(*x, *y, stream);
    }

    void MatmulOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        if (input_tensors[0]->empty() || input_tensors[1]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        const auto& shape1 = input_tensors[0]->shape();
        const auto& shape2 = input_tensors[1]->shape();
        if (shape1.size() < 2 || shape2.size() < 2) {
            throw std::invalid_argument("Input tensor should have at least 2 dims");
        }

        if (shape1.at(shape1.size()-1) != shape2.at(shape2.size()-2)) {
            throw std::invalid_argument("Input tensors' last two dims cannot be matched");
        }

        // 前n-2维度不同时，需要确保input2为矩阵
        if (!Tensor::shape_equal(
            std::vector<std::size_t>(shape1.begin(), shape1.end() - 2),
            std::vector<std::size_t>(shape2.begin(), shape2.end() - 2))) {
            if (shape2.size() != 2) {
                throw std::invalid_argument("Input 2 should be a matrix");
            }
        }
    }

    TransposeOperator::TransposeOperator(std::size_t dim0, std::size_t dim1) : dim0_(dim0), dim1_(dim1) {}

    Tensor TransposeOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);
        return transpose_kernel_func(*input_tensors[0], dim0_, dim1_, stream);
    }

    void TransposeOperator::check(const std::vector<const Tensor *> &input_tensors) {
        BaseOperator::check(input_tensors);
        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }

        if (input_tensors[0]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        if (this->dim0_ >= input_tensors[0]->shape().size()) {
            throw std::invalid_argument("Invalid dim0");
        }

        if (this->dim1_ >= input_tensors[0]->shape().size()) {
            throw std::invalid_argument("Invalid dim1");
        }
    }

    VecOrMatrixArgmaxOperator::VecOrMatrixArgmaxOperator(std::size_t n) : n_(n) {}

    Tensor VecOrMatrixArgmaxOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        return vec_or_matrix_argmax_kernel_func(*input_tensors[0], n_, stream);
    }

    void VecOrMatrixArgmaxOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (n_ == 0) {
            throw std::invalid_argument("n should not be 0");
        }

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }

        if (input_tensors[0]->empty()) {
            throw std::invalid_argument("Input tensor should not be empty");
        }

        if (input_tensors[0]->shape().size() != 1 && input_tensors[0]->shape().size() != 2) {
            throw std::invalid_argument("Input tensor should have shape 1 or 2");
        }

        if (n_ > input_tensors[0]->size()) {
            throw std::invalid_argument("n should not be larger than input tensor size");
        }
    }

    Tensor RotateHalfOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);

        return rotate_half_kernel_func(*input_tensors[0], stream);
    }

    void RotateHalfOperator::check(const std::vector<const Tensor *> &input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }

        // 维度必须大于等于2
        if (input_tensors[0]->shape().size() < 2) {
            throw std::invalid_argument("Input tensor should have at least 2 dims");
        }

        // 列必须能被2整除
        if (input_tensors[0]->shape().back() % 2 != 0) {
            throw std::invalid_argument("Input tensor's last dim should be even");
        }
    }

    Tensor CosineOperator::forward(const std::vector<const Tensor*>& input_tensors, void *stream) {
        this->check(input_tensors);

        return cos_kernel_func(*input_tensors[0], stream);
    }

    void CosineOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }
    }

    Tensor SineOperator::forward(const std::vector<const Tensor*>& input_tensors, void *stream) {
        this->check(input_tensors);

        return sin_kernel_func(*input_tensors[0], stream);
    }

    void SineOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }
    }

    Tensor ConcatByColWiseOperator::forward(const std::vector<const Tensor*>& input_tensors, void *stream) {
        this->check(input_tensors);

        return concat_by_col_wise_kernel_func(*input_tensors[0], *input_tensors[1], stream);
    }

    void ConcatByColWiseOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        // 前n-1个维度需要相同
        auto shape1 = input_tensors[0]->shape();
        shape1.pop_back();

        auto shape2 = input_tensors[1]->shape();
        shape2.pop_back();

        if (!Tensor::shape_equal(shape1, shape2)) {
            throw std::invalid_argument("Input tensor n - 1 shape should be same");
        }
    }

    Tensor ConcatByRowWiseOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);

        return concat_by_row_wise_kernel_func(*input_tensors[0], *input_tensors[1], stream);
    }

    void ConcatByRowWiseOperator::check(const std::vector<const Tensor *> &input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 tensors");
        }

        // 最后一个维度需要相同
        if (input_tensors[0]->shape().back() != input_tensors[1]->shape().back()) {
            throw std::invalid_argument("Input tensor last shape should be same");
        }

        // 前n-2个维度需要相同
        auto shape1 = input_tensors[0]->shape();
        shape1.pop_back(); shape1.pop_back();

        auto shape2 = input_tensors[1]->shape();
        shape2.pop_back(); shape2.pop_back();

        if (!Tensor::shape_equal(shape1, shape2)) {
            throw std::invalid_argument("Input tensor n - 2 shape should be same");
        }
    }

    CausalMaskOperator::CausalMaskOperator(DataType data_type, DeviceType device_type, std::size_t l, std::size_t s)
    : BaseOperator(), data_type_(data_type), device_type_(device_type), l_(l), s_(s) {}

    Tensor CausalMaskOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);

        return causal_mask_kernel_func(data_type_, device_type_, l_, s_, stream);
    }

    void CausalMaskOperator::check(const std::vector<const Tensor *> &input_tensors) {
        if (!input_tensors.empty()) {
            throw std::invalid_argument("Input tensor should have 0 tensor");
        }
    }

    MultinomialOperator::MultinomialOperator(std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos)
    : BaseOperator(), num_samples_(num_samples), row_end_pos_(row_end_pos) {}

    Tensor MultinomialOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);

        return multinomial_kernel_func(*input_tensors[0], num_samples_, row_end_pos_, stream);
    }

    void MultinomialOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should have 1 tensor");
        }

        if (input_tensors[0]->shape().size() != 2) {
            throw std::invalid_argument("Input tensor should have 2 dims");
        }

        if (num_samples_ == 0) {
            throw std::invalid_argument("Input tensor should have at least one sample");
        }
    }
}
