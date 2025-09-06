//
// Created by 19373 on 2025/9/6.
//
#include <algorithm>
#include "util/util.h"

namespace fg42::util {
    std::string to_lower(const std::string& str) {
        std::string copy(str);
        std::transform(copy.begin(), copy.end(), copy.begin(),
            [](unsigned char c){ return std::tolower(c); });
        return copy;
    }
}