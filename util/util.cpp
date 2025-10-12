//
// Created by B777B2056-2 on 2025/9/6.
//
#include <algorithm>
#include "Eigen/Core"
#include "util/util.h"
#include "memory/Common.h"
#ifdef HAVE_CUDA
#include <cuda_bf16.h>
#endif

namespace fg42::util {
    std::string to_lower(const std::string& str) {
        std::string copy(str);
        std::transform(copy.begin(), copy.end(), copy.begin(),
            [](unsigned char c){ return std::tolower(c); });
        return copy;
    }

    float bfloat16_to_float(DeviceType device_type, const void* bfloat16_val_ptr) {
        switch (device_type) {
            case DeviceType::CPU:
                return *static_cast<const Eigen::bfloat16*>(bfloat16_val_ptr);
            case DeviceType::NvidiaGPU: {
#ifdef HAVE_CUDA
                const __nv_bfloat16* val = static_cast<const __nv_bfloat16*>(bfloat16_val_ptr);
                return float(*val);
#endif
            }
            default:
                throw std::runtime_error("unsupported device type");
        }
    }

    bool ends_with(const std::string& str, const std::string& suffix) {
        if (suffix.length() > str.length()) { return false; }
        return (str.rfind(suffix) == (str.length() - suffix.length()));
    }
}
