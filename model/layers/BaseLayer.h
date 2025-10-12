//
// Created by B777B2056-2 on 2025/9/18.
//

#ifndef FG42_BASELAYER_H
#define FG42_BASELAYER_H
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor/Tensor.h"

namespace fg42 {
    struct ModelConfig;
    using StateDict = std::unordered_map<std::string, Tensor>;

    class BaseLayer {
    public:
        explicit BaseLayer(const ModelConfig& model_config, std::string layer_name);
        BaseLayer(const BaseLayer&) = delete;
        BaseLayer& operator=(const BaseLayer&) = delete;
        BaseLayer(BaseLayer&&) = delete;
        BaseLayer& operator=(BaseLayer&&) = delete;
        virtual ~BaseLayer() = default;

        virtual void init_weights(const StateDict& state_dict) = 0;
        virtual Tensor forward(const std::vector<const Tensor*>& input) = 0;
        [[nodiscard]] const std::string& layer_name() const;

    protected:
        std::string layer_name_;
        const ModelConfig* model_config_;
    };

    const Tensor* get_weight_from_state_dict(const StateDict& state_dict,
        const std::string& name, bool need_exception = false);
    const Tensor* get_tensor_from_vector(const std::vector<const Tensor*>& tensors,
        std::size_t index, bool need_exception = false);

    std::string build_weight_tensor_name(const std::string& layer_name, int index=-1);
    std::string build_bias_tensor_name(const std::string& layer_name, int index=-1);
} // fg42

#endif //FG42_BASELAYER_H