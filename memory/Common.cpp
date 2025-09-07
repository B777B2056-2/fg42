//
// Created by 19373 on 2025/9/4.
//
#include <memory>
#include <stdexcept>
#ifdef HAVE_CUDA
    #include <cuda_runtime_api.h>
    #include <cuda_runtime.h>
    #include "memory/NvidiaGPUMemoryAllocator.h"
#endif

#include "memory/Common.h"
#include "memory/CPUMemoryAllocator.h"

namespace fg42 {
    BaseAllocator* allocator_factory(DeviceType device_type) {
        static CPUMemoryAllocator cpu_allocator;
#ifdef HAVE_CUDA
        static NvidiaGPUMemoryAllocator cuda_allocator;
#endif
        switch (device_type) {
        case DeviceType::CPU:
            return &cpu_allocator;
#ifdef HAVE_CUDA
        case DeviceType::NvidiaGPU:
            return &cuda_allocator;
#endif
        default:
            throw std::runtime_error("Unknown device type");
        }
    }

    std::shared_ptr<void> make_shared_ptr_on_device(const PtrDeviceWrapper& ptr) {
        auto device_type = ptr.device_type();

        auto deleter = [device_type](void* p) {
            auto allocator = allocator_factory(device_type);
            allocator->deallocate(PtrDeviceWrapper(device_type, p));
        };
        return {ptr.raw_ptr(), deleter};
    }

    void memcpy(PtrDeviceWrapper& dst, const PtrDeviceWrapper& src,
                std::size_t count, MemcpyOptions* options) {
        // 均为cpu类型
        if (dst.device_type() == DeviceType::CPU && src.device_type() == DeviceType::CPU) {
            std::memcpy(dst.raw_ptr(), src.raw_ptr(), count);
            return;
        }

#ifdef HAVE_CUDA
        // 存在cuda类型
        if (dst.device_type() == DeviceType::NvidiaGPU || src.device_type() == DeviceType::NvidiaGPU) {
            auto cudaMemcpyWrapper = [&dst, &src, count, options](cudaMemcpyKind kind) {
                // 存在gpu类型，且存在拷贝选项
                cudaStream_t stream = nullptr;
                if (options != nullptr && options->stream != nullptr) {
                    stream = static_cast<cudaStream_t>(options->stream);
                }

                if (stream == nullptr) {
                    cudaMemcpy(dst.raw_ptr(), src.raw_ptr(),  count, kind);
                } else {
                    cudaMemcpyAsync(dst.raw_ptr(), src.raw_ptr(), count, kind, stream);
                }

                if (options != nullptr && options->need_sync) {
                    cudaDeviceSynchronize();
                }
            };

            // 均为gpu类型
            if (dst.device_type() == DeviceType::NvidiaGPU && src.device_type() == DeviceType::NvidiaGPU) {
                cudaMemcpyWrapper(cudaMemcpyDeviceToDevice);
                return;
            }
            // gpu -> cpu
            if (dst.device_type() == DeviceType::CPU && src.device_type() == DeviceType::NvidiaGPU) {
                cudaMemcpyWrapper(cudaMemcpyDeviceToHost);
                return;
            }
            // cpu -> gpu
            if (dst.device_type() == DeviceType::NvidiaGPU && src.device_type() == DeviceType::CPU) {
                cudaMemcpyWrapper(cudaMemcpyHostToDevice);
                return;
            }
        }
#endif
    }

    // 为每个字节赋值
    void memset(PtrDeviceWrapper& p, int val, std::size_t n) {
        // cpu
        if (p.device_type() == DeviceType::CPU) {
            std::memset(p.raw_ptr(), val, n);
            return;
        }
#ifdef HAVE_CUDA
        // gpu
        if (p.device_type() == DeviceType::NvidiaGPU) {
            cudaMemset(p.raw_ptr(), val, n);
            return;
        }
#endif
    }
}