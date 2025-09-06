//
// Created by 19373 on 2025/9/4.
//
#include <stdexcept>
#include "memory/Common.h"
#include "tensor/Tensor.h"

namespace fg42 {
    Tensor::Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape)
                : data_type_(data_type), device_type_(device_type), shape_(shape), ptr_(nullptr) {
        // 分配底层内存
        auto allocator = allocator_factory(device_type);
        PtrDeviceWrapper dst = allocator->allocate(this->bytes_size());
        // 构造智能指针
        this->ptr_ = make_shared_ptr_on_device(dst);
    }

    Tensor::Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape, std::shared_ptr<void> ptr)
            : data_type_(data_type), device_type_(device_type), shape_(shape), ptr_(std::move(ptr)) {}

    Tensor::Tensor(const Tensor& tensor) : data_type_(tensor.data_type_), device_type_(tensor.device_type_), shape_(tensor.shape_), ptr_(nullptr) {
        // 为目标设备指针分配底层内存
        auto allocator = allocator_factory(tensor.device_type_);
        PtrDeviceWrapper dst = allocator->allocate(tensor.size() * data_type_size(tensor.data_type()));
        // 构造带设备信息的源指针
        PtrDeviceWrapper src(tensor.device_type(), tensor.raw_ptr());
        // 拷贝底层内存（同步拷贝）
        fg42::memcpy(dst, src, tensor.size() * data_type_size(tensor.data_type()));
        // 构造智能指针
        this->ptr_ = make_shared_ptr_on_device(dst);
    }

    Tensor& Tensor::operator=(const Tensor& tensor) {
        if (this != &tensor) {
            // 为目标设备指针分配底层内存
            auto allocator = allocator_factory(this->device_type());
            PtrDeviceWrapper dst = allocator->allocate(tensor.size() * data_type_size(tensor.data_type()));
            // 构造带设备信息的源指针
            PtrDeviceWrapper src(tensor.device_type(), tensor.raw_ptr());
            // 拷贝底层内存（同步拷贝）
            fg42::memcpy(dst, src, tensor.size() * data_type_size(tensor.data_type()));
            // 构造智能指针
            this->ptr_ = make_shared_ptr_on_device(dst);
            // 切换数据类型
            this->data_type_ = tensor.data_type();
            // 切换硬件
            this->device_type_ = tensor.device_type();
            // 更新形状
            this->shape_ = tensor.shape();
        }
        return *this;
    }

    Tensor::Tensor(Tensor&& tensor) noexcept : data_type_(DataType::Unknown), device_type_(DeviceType::Unknown), shape_(), ptr_(nullptr) {
        *this = std::move(tensor);
    }

    Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
        if (this != &tensor) {
            this->data_type_ = tensor.data_type();
            this->device_type_ = tensor.device_type();
            this->shape_ = tensor.shape();
            this->ptr_ = std::move(tensor.ptr_);
        }
        return *this;
    }

    // 获取原生指针
    void* Tensor::raw_ptr() noexcept { return this->ptr_.get(); }
    void* Tensor::raw_ptr() const noexcept { return this->ptr_.get(); }

    // 获取底层引用计数
    long Tensor::use_count() const noexcept { return this->ptr_.use_count(); }

    // 获取形状
    const std::vector<std::size_t>& Tensor::shape() const noexcept { return this->shape_; }

    // 修改形状
    std::optional<Tensor> Tensor::reshape(const std::vector<std::size_t>& new_shape, bool copy) {
        auto new_size = Tensor::calc_size(new_shape);
        if (new_size != this->size()) {
            throw std::invalid_argument("New shape must have the same number of elements");
        }

        if (copy) {
            return Tensor(this->data_type(), this->device_type(), new_shape);
        }
        this->shape_ = new_shape;
        return std::nullopt;
    }

    // 获取当前数据类型
    DataType Tensor::data_type() const noexcept { return this->data_type_; }

    // 获取当前设备
    DeviceType Tensor::device_type() const noexcept { return this->device_type_; }

    // 判断是否为空tensor
    bool Tensor::empty() const noexcept {
        return this->data_type() == DataType::Unknown ||
            this->device_type() == DeviceType::Unknown ||
                this->shape().empty() ||
                    this->ptr_ == nullptr;
    }

    // 获取大小
    std::size_t Tensor::size() const noexcept {
        if (this->shape_.empty()) {
            return 0;
        }
        return Tensor::calc_size(this->shape());
    }

    std::size_t Tensor::bytes_size() const noexcept {
        return this->size() * data_type_size(this->data_type());
    }

    // 获取步长
    std::vector<std::size_t> Tensor::strides() const noexcept {
        std::vector<std::size_t> res;
        if (this->shape_.empty()) {
            return res;
        }

        for (std::size_t cur_dim = 0; cur_dim < this->shape_.size(); cur_dim++) {
            std::size_t stride = 1;
            for (std::size_t next_dim = cur_dim + 1; next_dim < this->shape_.size(); next_dim++) {
                stride *= this->shape_.at(next_dim);
            }
            res.push_back(stride);
        }
        return res;
    }

    // 按索引取值
    /*
      A[i0][i1][i2]...[in-1] = A_internal[
            stride_offset
            + i0 * A.strides[0]
            + i1 * A.strides[1]
            + i2 * A.strides[2]
            + ...
            + in-1 * A.strides[n-1]
        ]
     */
    void* Tensor::data(const std::vector<std::size_t>& indexes) const {
        // 按步长获取元素
        auto strides = this->strides();
        // 校验
        if (indexes.size() > this->size()) {
            throw std::invalid_argument("indexes must below the number of elements");
        }
        // 计算
        std::size_t offset = 0;
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            auto it = indexes.begin() + i;
            offset += (*it) * strides.at(i);
        }
        // 返回
        if (offset >= this->size()) {
            throw std::invalid_argument("indexes out of range");
        }
        offset *= data_type_size(this->data_type());
        return static_cast<char*>(this->raw_ptr()) + offset;
    }

    // 按索引赋值
    void Tensor::index_fill(const std::vector<std::size_t>& indexes, void* val) {
        // 获取元素
        void* p = this->data(indexes);
        // 赋值
        PtrDeviceWrapper dst(this->device_type(), p);
        PtrDeviceWrapper src(DeviceType::CPU, val);
        fg42::memcpy(dst, src, 1 * data_type_size(this->data_type()));
    }

    // 切换底层硬件
    void Tensor::to_device(DeviceType device_type, MemcpyOptions* options) {
        if (device_type == this->device_type()) {
            return;
        }

        // 为目标设备分配底层内存
        auto allocator = allocator_factory(device_type);
        PtrDeviceWrapper dst = allocator->allocate(this->bytes_size());
        // 构造带设备信息的源指针
        PtrDeviceWrapper src(this->device_type(), this->raw_ptr());
        // 拷贝底层内存
        fg42::memcpy(dst, src, this->bytes_size(), options);
        // 构造智能指针
        this->ptr_ = make_shared_ptr_on_device(dst);
        // 切换设备
        this->device_type_ = device_type;
    }

    // 拷贝张量
    Tensor Tensor::clone(DeviceType device_type, MemcpyOptions* options) const {
        if (device_type == this->device_type()) {
            return *this;
        }

        Tensor t(*this);
        t.to_device(device_type, options);
        return t;
    }

    // 检测两张量shape是否一致
    bool Tensor::shape_equal(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    std::size_t Tensor::calc_size(const std::vector<std::size_t>& shape) noexcept {
        std::size_t size = 1;
        for (auto it = shape.begin(); it != shape.end(); ++it) {
            size *= *it;
        }
        return size;
    }
}
