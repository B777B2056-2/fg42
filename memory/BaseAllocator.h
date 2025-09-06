//
// Created by 19373 on 2025/8/31.
//

#ifndef FG42_BASEALLOCATOR_H
#define FG42_BASEALLOCATOR_H
#include <cstddef>
#include "util/enum.h"

namespace fg42 {
    class PtrDeviceWrapper {
    private:
        DeviceType device_type_;
        void* ptr_;

    public:
        PtrDeviceWrapper();
        explicit PtrDeviceWrapper( DeviceType device_type, void* ptr=nullptr);
        PtrDeviceWrapper(PtrDeviceWrapper&& other) noexcept;
        PtrDeviceWrapper& operator=(PtrDeviceWrapper&& other) noexcept;
        ~PtrDeviceWrapper() = default;

        // 判断指针是否非空
        explicit operator bool() const noexcept;

        // 访问原始指针
        [[nodiscard]] void* raw_ptr() const noexcept;
        void* operator->() const noexcept;

        // 重置指针
        void reset() noexcept;

        // 获取设备类型
        [[nodiscard]] DeviceType device_type() const noexcept;

        // 与nullptr比较
        bool operator==(std::nullptr_t) const noexcept;
        bool operator!=(std::nullptr_t) const noexcept;

    private:
        PtrDeviceWrapper(const PtrDeviceWrapper& other) = default;
        PtrDeviceWrapper& operator=(const PtrDeviceWrapper& other) = default;
    };

    class BaseAllocator {
    protected:
        DeviceType device_type_;

    public:
        explicit BaseAllocator(DeviceType device_type) : device_type_(device_type) {};
        virtual ~BaseAllocator() = default;

        virtual PtrDeviceWrapper allocate(std::size_t n) = 0;
        virtual void deallocate(PtrDeviceWrapper&& p) = 0;
    };
} // fg42

#endif //FG42_BASEALLOCATOR_H