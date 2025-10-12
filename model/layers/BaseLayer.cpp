//
// Created by B777B2056-2 on 2025/9/18.
//
#include "model/BaseModel.h"
#include "model/layers/BaseLayer.h"

#include <utility>

namespace fg42 {
    BaseLayer::BaseLayer(const ModelConfig& model_config, std::string layer_name)
        : layer_name_(std::move(layer_name)), model_config_(&model_config) {}

    const std::string& BaseLayer::layer_name() const {
        return layer_name_;
    }

    const Tensor* get_weight_from_state_dict(const StateDict& state_dict, const std::string& name, bool need_exception) {
        if (state_dict.find(name) != state_dict.end()) {
            return &state_dict.at(name);
        }
        if (need_exception) {
            throw std::runtime_error("Weight not found in state_dict: " + name);
        }
        return nullptr;
    }

    const Tensor* get_tensor_from_vector(
        const std::vector<const Tensor*>& tensors, std::size_t index, bool need_exception) {
        if (index >= tensors.size()) {
            if (need_exception) {
                throw std::runtime_error("Tensor index out of range.");
            }
            return nullptr;
        }
        return tensors.at(index);
    }

    std::string build_weight_tensor_name(const std::string& layer_name, int index) {
        if (index == -1) {
            return "model." + layer_name + ".weight";
        }
        return "model.layers." + std::to_string(index) + "." + layer_name + ".weight";
    }

    std::string build_bias_tensor_name(const std::string& layer_name, int index) {
        if (index == -1) {
            return "model." + layer_name + ".bias";
        }
        return "model.layers." + std::to_string(index) + "." + layer_name + ".bias";
    }
} // fg42