//
// Created by B777B2056-2 on 2025/9/6.
//

#ifndef FG42_UTIL_H
#define FG42_UTIL_H
#include <string>
#include "util/enum.h"

// 前置声明
namespace Eigen {
    struct bfloat16;
}

namespace fg42::util {
    std::string to_lower(const std::string& str);

    // bfloat16转float
    float bfloat16_to_float(DeviceType device_type, const void* bfloat16_val_ptr);

    // 字符串后缀判断
    bool ends_with(const std::string& str, const std::string& suffix);
}

#endif //FG42_UTIL_H