//
// Created by B777B2056-2 on 2025/10/10.
//

#ifndef FG42_FACTORY_H
#define FG42_FACTORY_H
#include <filesystem>
#include "BaseWeightsLoader.h"

namespace fg42 {
    BaseWeightsLoader* model_weights_loader_factory(const std::filesystem::path& dir_path);
}   // namespace fg42

#endif //FG42_FACTORY_H