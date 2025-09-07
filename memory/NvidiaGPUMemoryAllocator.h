//
// Created by 19373 on 2025/8/31.
//

#ifndef FG42_NVIDIAGPUMEMORYALLOCATOR_H
#define FG42_NVIDIAGPUMEMORYALLOCATOR_H
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include "memory/BaseAllocator.h"

namespace fg42 {
    class  NvidiaGPUMemoryError : public std::runtime_error {
    public:
        explicit  NvidiaGPUMemoryError(cudaError_t error);
    };

    class NvidiaGPUMemoryAllocator : public BaseAllocator {
    public:
        NvidiaGPUMemoryAllocator();
        ~NvidiaGPUMemoryAllocator() override = default;

        PtrDeviceWrapper allocate(std::size_t n) override;
        void deallocate(PtrDeviceWrapper&& p) override;
    };
} // fg42
#endif

#endif //FG42_NVIDIAGPUMEMORYALLOCATOR_H