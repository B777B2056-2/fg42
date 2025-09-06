//
// Created by 19373 on 2025/9/4.
//

#ifndef FG42_MODELLOADER_H
#define FG42_MODELLOADER_H
#include <string>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "tensor/Tensor.h"

namespace fg42 {
    typedef std::unordered_map<std::string, Tensor> StateDict;

    class BaseModel {
    protected:
        DeviceType device_type_;
        nlohmann::json model_config_;
        StateDict state_dict_;

    public:
        BaseModel() = delete;
        explicit BaseModel(const std::string& dir_path, DeviceType device_type);
        BaseModel(const BaseModel&) = delete;
        BaseModel& operator=(const BaseModel&) = delete;
        BaseModel(BaseModel&&) = delete;
        BaseModel& operator=(BaseModel&&) = delete;
        virtual ~BaseModel() = default;

        virtual Tensor forward(const Tensor& input) = 0;

    protected:
        void load_model_config(const std::string& dir_path);
        void load_model_weights(const std::string& dir_path);
    };
} // fg42

#endif //FG42_MODELLOADER_H