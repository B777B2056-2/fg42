//
// Created by B777B2056-2 on 2025/10/10.
//
#include "model/weights_loaders/BaseWeightsLoader.h"
#if defined(__linux__) || defined(__APPLE__)
#include <cstring>
extern "C" {
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
}
#endif
#include "memory/Common.h"

namespace fg42 {
#if defined(__linux__) || defined(__APPLE__)
    LinuxWeightsFileReaderImpl::LinuxWeightsFileReaderImpl(std::filesystem::path path)
    : current_pos_(0), path_(std::move(path)) {
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ == -1) {
            return;
        }
        file_bytes_ = lseek(fd_, 0, SEEK_END);
        if (file_bytes_ == -1) {
            return;
        }
        // 内存映射
        lseek(fd_, 0, SEEK_SET);
        mapped_mem_ = static_cast<char*>(mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0));
    }

    LinuxWeightsFileReaderImpl::~LinuxWeightsFileReaderImpl() {
        if (mapped_mem_ != MAP_FAILED) {
            munmap(mapped_mem_, file_bytes_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    bool LinuxWeightsFileReaderImpl::is_open() const {
        return (fd_ != -1) && (file_bytes_ != -1) && (mapped_mem_ != MAP_FAILED);
    }

    void LinuxWeightsFileReaderImpl::read_into_tensor(Tensor& tensor) {
        std::size_t size = tensor.bytes_size();
        PtrDeviceWrapper dst(tensor.device_type(), tensor.raw_ptr());
        PtrDeviceWrapper src(DeviceType::CPU, mapped_mem_ + current_pos_);
        memcpy_between_device(dst, src, size);
        current_pos_ += size;
    }

    void LinuxWeightsFileReaderImpl::read(void* buffer, std::size_t size) {
        std::memcpy(buffer, mapped_mem_ + current_pos_, size);
        current_pos_ += size;
    }

    std::size_t LinuxWeightsFileReaderImpl::offset() {
        return current_pos_;
    }

    void LinuxWeightsFileReaderImpl::set_offset(std::size_t pos) {
        current_pos_ = pos;
    }
#elif _WIN32
    WindowsWeightsFileReaderImpl::WindowsWeightsFileReaderImpl(std::filesystem::path path)
    : ifs_(path, std::ios::in | std::ios::binary) {}

    bool WindowsWeightsFileReaderImpl::is_open() const { return ifs_.is_open(); }

    void WindowsWeightsFileReaderImpl::read_into_tensor(Tensor& tensor) {
        std::size_t size = tensor.bytes_size();

        if (tensor.device_type() == DeviceType::CPU) {
            ifs_.read(static_cast<char*>(tensor.raw_ptr()), size);
        } else {
            char* buffer = new char[size];
            ifs_.read(static_cast<char*>(buffer), size);

            PtrDeviceWrapper dst(tensor.device_type(), tensor.raw_ptr());
            PtrDeviceWrapper src(DeviceType::CPU, buffer);
            memcpy_between_device(dst, src, size);

            delete[] buffer;
        }
    }

    void WindowsWeightsFileReaderImpl::read(void* buffer, std::size_t size) {
        ifs_.read(static_cast<char*>(buffer), size);
    }

    std::size_t WindowsWeightsFileReaderImpl::offset() { return ifs_.tellg(); }

    void WindowsWeightsFileReaderImpl::set_offset(std::size_t pos) { ifs_.seekg(pos, std::ios::beg); }
#endif

    WeightsFileReader::WeightsFileReader(std::filesystem::path path) : pimpl_(std::make_unique<Impl>(path)) {}

    bool WeightsFileReader::is_open() const { return pimpl_->is_open(); }

    void WeightsFileReader::read_into_tensor(Tensor& tensor) { pimpl_->read_into_tensor(tensor); }

    void WeightsFileReader::read(void* buffer, std::size_t size) { pimpl_->read(buffer, size); }

    std::size_t WeightsFileReader::offset() { return pimpl_->offset(); }

    void WeightsFileReader::set_offset(std::size_t pos) { pimpl_->set_offset(pos); }
}   // fg42