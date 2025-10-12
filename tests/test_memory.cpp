//
// Created by B777B2056-2 on 2025/9/6.
//
#ifdef HAVE_CUDA
#include "memory/NvidiaGPUMemoryAllocator.h"
#include "memory/Common.h"
#include <gtest/gtest.h>


TEST(MemoryAlloclTest, Cuda) {
    // 分配 1MB 的显存 (1MB = 1024 * 1024 bytes)
    const size_t size = (1024 * 1024) / sizeof(int); // 1MB

    fg42::NvidiaGPUMemoryAllocator allocator;
    auto d_memory = allocator.allocate(size);

    // 释放显存
    allocator.deallocate(std::move(d_memory));

    // 检查是否有任何CUDA错误
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}
#endif
