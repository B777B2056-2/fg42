//
// Created by 19373 on 2025/8/31.
//

#ifndef FG42_COMMON_H
#define FG42_COMMON_H
#include <memory>
#include "memory/BaseAllocator.h"

namespace fg42 {
    BaseAllocator* allocator_factory(DeviceType device_type);

    std::shared_ptr<void> make_shared_ptr_on_device(const PtrDeviceWrapper& ptr);

    struct MemcpyOptions {
        void* stream = nullptr;
        bool need_sync = false;
    };

    void memcpy(PtrDeviceWrapper& dst, const PtrDeviceWrapper& src,
                std::size_t count, MemcpyOptions* options = nullptr);

    // 为每个字节赋值
    void memset(PtrDeviceWrapper& p, int val, std::size_t n);
}

#endif //FG42_COMMON_H