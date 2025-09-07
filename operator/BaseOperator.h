//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_BASEOPERATOR_H
#define FG42_BASEOPERATOR_H
#include <string>
#include "tensor/Tensor.h"

namespace fg42::kernel {
    class BaseOperator {
    protected:
        DeviceType device_type_;
        std::string name_;

    public:
        explicit BaseOperator(DeviceType device_type, std::string name = "");

        virtual ~BaseOperator() = default;

        [[nodiscard]] DeviceType device_type() const;
        [[nodiscard]] const std::string& name() const;

        virtual Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) = 0;

    protected:
        virtual bool check(const std::vector<const Tensor*>& input_tensors) const = 0;
    };

    class BaseOperatorWithWeight : public BaseOperator {
    protected:
        const Tensor* weight_tensor_;

    public:
        explicit BaseOperatorWithWeight(const Tensor& weight_tensor, const std::string& name = "");

    protected:
        virtual bool check_weights(const Tensor& weight_tensors) const = 0;
    };
} // kernel
// fg42

#endif //FG42_BASEOPERATOR_H