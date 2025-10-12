//
// Created by B777B2056-2 on 2025/8/31.
//

#ifndef FG42_CPUMEMORYALLOCATOR_H
#define FG42_CPUMEMORYALLOCATOR_H
#include "memory/BaseAllocator.h"

namespace fg42 {
    class CPUMemoryAllocator : public BaseAllocator {
    public:
        CPUMemoryAllocator();
        ~CPUMemoryAllocator() override = default;

        PtrDeviceWrapper allocate(std::size_t n) override;
        void deallocate(PtrDeviceWrapper&& p) override;
    };
} // fg42

#endif //FG42_CPUMEMORYALLOCATOR_H