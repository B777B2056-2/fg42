//
// Created by B777B2056-2 on 2025/9/4.
//
#include "memory/BaseAllocator.h"

namespace fg42 {
    PtrDeviceWrapper::PtrDeviceWrapper() :  device_type_(DeviceType::Unknown), ptr_(nullptr) {}

    PtrDeviceWrapper::PtrDeviceWrapper( DeviceType device_type, void* ptr)
        : device_type_(device_type), ptr_(ptr) {}

    PtrDeviceWrapper::PtrDeviceWrapper(PtrDeviceWrapper&& other) noexcept
            : device_type_(other.device_type_), ptr_(other.ptr_) {
        other.reset();
    }

    PtrDeviceWrapper& PtrDeviceWrapper::operator=(PtrDeviceWrapper&& other)  noexcept {
        if (this != &other) {
            this->device_type_ = other.device_type_;
            this->ptr_ = other.ptr_;
            other.reset();
        }
        return *this;
    }

    PtrDeviceWrapper::operator bool() const noexcept {
        return this->ptr_ != nullptr;
    }

    void* PtrDeviceWrapper::raw_ptr() const noexcept { return this->ptr_; }
    void* PtrDeviceWrapper::operator->() const noexcept { return this->ptr_; }

    void PtrDeviceWrapper::reset() noexcept {
        this->device_type_ = DeviceType::Unknown;
        this->ptr_ = nullptr;
    }

    DeviceType PtrDeviceWrapper::device_type() const noexcept { return this->device_type_; }

    bool PtrDeviceWrapper::operator==(std::nullptr_t) const noexcept { return this->ptr_ == nullptr; }
    bool PtrDeviceWrapper::operator!=(std::nullptr_t) const noexcept { return this->ptr_ != nullptr; }
}