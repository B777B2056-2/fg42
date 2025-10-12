//
// Created by B777B2056-2 on 2025/9/4.
//
#ifdef HAVE_CUDA
#include "memory/NvidiaGPUMemoryAllocator.h"

namespace fg42 {
    NvidiaGPUMemoryError::NvidiaGPUMemoryError(cudaError_t error)
            : std::runtime_error(
                "CUDA memory alloc failed, error code is "
                +std::string(cudaGetErrorString(error))) {}

    NvidiaGPUMemoryAllocator::NvidiaGPUMemoryAllocator() : BaseAllocator(DeviceType::NvidiaGPU) {};

    PtrDeviceWrapper NvidiaGPUMemoryAllocator::allocate(std::size_t n) {
        if (n == 0) {
            return PtrDeviceWrapper(this->device_type_, nullptr);
        }

        void* object_ptr = nullptr;
        auto err = cudaMalloc((void**)&object_ptr, n);
        if (err != cudaSuccess) {
            throw NvidiaGPUMemoryError(err);
        }
        return PtrDeviceWrapper(this->device_type_, object_ptr);
    }

    void NvidiaGPUMemoryAllocator::deallocate(PtrDeviceWrapper&& p) {
        cudaFree(p.raw_ptr());
        p.reset();
    }
}
#endif