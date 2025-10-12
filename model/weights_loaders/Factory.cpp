//
// Created by B777B2056-2 on 2025/10/10.
//
#include "model/weights_loaders/Factory.h"
#include "model/weights_loaders/SafeTensorsLoader.h"
#include "util/util.h"

namespace fg42 {
    BaseWeightsLoader* model_weights_loader_factory(const std::filesystem::path& dir_path) {
        static SafeTensorsLoader safetensors_weights_loader;
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (util::ends_with(entry.path().string(), ".safetensors")) {
                return &safetensors_weights_loader;
            }
        }
        return nullptr;
    }
}   // namespace fg42