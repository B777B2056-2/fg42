//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_VECADDOPERATOR_H
#define FG42_VECADDOPERATOR_H
#include "operator/BaseOperator.h"

namespace fg42::kernel {
    class VecAddOperator final : public BaseOperator {
    public:
        explicit VecAddOperator(DeviceType device_type, std::string name = "");
        ~VecAddOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        bool check(const std::vector<const Tensor*>& input_tensors) const override;
    };
}

#endif //FG42_VECADDOPERATOR_H
