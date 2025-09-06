//
// Created by 19373 on 2025/9/4.
//
#include <cstdlib>
#include "memory/CPUMemoryAllocator.h"

namespace fg42 {
    CPUMemoryAllocator::CPUMemoryAllocator() : BaseAllocator(DeviceType::CPU) {}

    PtrDeviceWrapper CPUMemoryAllocator::allocate(std::size_t n) {
        if (n == 0) {
            return PtrDeviceWrapper(this->device_type_, nullptr);
        }
        void* raw_ptr = std::malloc(n);
        return PtrDeviceWrapper(this->device_type_, raw_ptr);
    }

    void CPUMemoryAllocator::deallocate(PtrDeviceWrapper&& p) {
        std::free(p.raw_ptr());
        p.reset();
    }
}