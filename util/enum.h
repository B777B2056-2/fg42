//
// Created by B777B2056-2 on 2025/8/31.
//

#ifndef FG42_ENUM_H
#define FG42_ENUM_H
#include <stdexcept>

namespace fg42 {
    enum class DeviceType : std::uint8_t {
        Unknown = 0,
        CPU = 1,
        NvidiaGPU = 2,
    };

    enum class DataType : std::uint8_t {
        Unknown = 0,
        Int8 = 10,
        UInt8 = 11,
        Int32 = 12,
        BF16 = 20,
        FP32 = 30,
    };

    inline std::size_t data_type_size(DataType data_type) {
        switch (data_type) {
        case DataType::Int8:
            return sizeof(std::int8_t);
        case DataType::UInt8:
            return sizeof(std::uint8_t);
        case DataType::Int32:
            return sizeof(std::int32_t);
        case DataType::BF16:
            return sizeof(std::uint16_t);
        case DataType::FP32:
            return sizeof(float);
        default:
            return 0;
        }
    }
}

#endif //FG42_ENUM_H