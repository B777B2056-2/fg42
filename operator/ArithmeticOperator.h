//
// Created by B777B2056-2 on 2025/9/3.
//

#ifndef FG42_VECADDOPERATOR_H
#define FG42_VECADDOPERATOR_H
#include "operator/BaseOperator.h"
#include <tuple>

namespace fg42::kernel {
    class AddOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~AddOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class NegateOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~NegateOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class VecOuterOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~VecOuterOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class MulOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~MulOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class MulWithConstantValueOperator final : public BaseOperator {
    public:
        explicit MulWithConstantValueOperator(float c);
        ~MulWithConstantValueOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        float c_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class MatmulOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~MatmulOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class TransposeOperator final : public BaseOperator {
    public:
        TransposeOperator(std::size_t dim0, std::size_t dim1);
        ~TransposeOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;
    private:
        std::size_t dim0_;
        std::size_t dim1_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class VecOrMatrixArgmaxOperator final : public BaseOperator {
    public:
        explicit VecOrMatrixArgmaxOperator(std::size_t n);
        ~VecOrMatrixArgmaxOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;
    private:
        std::size_t n_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class RotateHalfOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~RotateHalfOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class CosineOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~CosineOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class SineOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~SineOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class ConcatByColWiseOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~ConcatByColWiseOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class ConcatByRowWiseOperator final : public BaseOperator {
    public:
        using BaseOperator::BaseOperator;
        ~ConcatByRowWiseOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class CausalMaskOperator final : public BaseOperator {
    public:
        CausalMaskOperator(DataType data_type, DeviceType device_type, std::size_t l, std::size_t s);
        ~CausalMaskOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        DataType data_type_;
        DeviceType device_type_;
        std::size_t l_;
        std::size_t s_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class MultinomialOperator final : public BaseOperator {
    public:
        explicit MultinomialOperator(std::size_t num_samples = 1,
            const std::function<std::size_t(std::size_t)>& row_end_pos = nullptr);
        ~MultinomialOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        std::size_t num_samples_;
        std::function<std::size_t(std::size_t)> row_end_pos_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };
}

#endif //FG42_VECADDOPERATOR_H
