//
// Created by B777B2056-2 on 2025/9/4.
//
#include <stdexcept>
#include "memory/Common.h"
#include "operator/ArithmeticOperator.h"
#include "tensor/Tensor.h"
#ifdef HAVE_CUDA
#include <cuda_bf16.h>
#endif
#include "Eigen/Core"
#include "util/util.h"

namespace fg42 {
    static void memcpy_with_data_type(DeviceType device_type, void* dst, DataType data_type, float val) {
        PtrDeviceWrapper dst_wrapper(device_type, dst);
        std::size_t byte_size = fg42::data_type_size(data_type);
        switch (data_type) {
            case DataType::Int32: {
                auto src_val = static_cast<std::int32_t>(val);
                PtrDeviceWrapper src_wrapper(DeviceType::CPU, &src_val);
                fg42::memcpy_between_device(dst_wrapper, src_wrapper, byte_size);
            }
                break;
            case DataType::BF16: {
                if (device_type == DeviceType::NvidiaGPU) {
#ifdef HAVE_CUDA
                    auto src_val = static_cast<__nv_bfloat16>(val);
                    PtrDeviceWrapper src_wrapper(DeviceType::CPU, &src_val);
                    fg42::memcpy_between_device(dst_wrapper, src_wrapper, byte_size);
#endif
                } else if (device_type == DeviceType::CPU) {
                    auto src_val = static_cast<Eigen::bfloat16>(val);
                    PtrDeviceWrapper src_wrapper(DeviceType::CPU, &src_val);
                    fg42::memcpy_between_device(dst_wrapper, src_wrapper, byte_size);
                }
            }
                break;
            case DataType::FP32:{
                auto src_val = static_cast<float>(val);
                PtrDeviceWrapper src_wrapper(DeviceType::CPU, &src_val);
                fg42::memcpy_between_device(dst_wrapper, src_wrapper, byte_size);
            }
                break;
            default:
                throw std::runtime_error("unsupported data type");
        }
    }

    Tensor::Tensor()
        : data_type_(DataType::Unknown), device_type_(DeviceType::Unknown),
          shape_(), ptr_(nullptr), view_ptr_(nullptr) {}

    Tensor::Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape)
        : data_type_(data_type), device_type_(device_type),
            shape_(shape), strides_(Tensor::calc_strides(shape)), ptr_(nullptr), view_ptr_(nullptr) {
        // 分配底层内存
        auto allocator = allocator_factory(device_type);
        PtrDeviceWrapper dst = allocator->allocate(this->bytes_size());
        // 构造智能指针
        this->ptr_ = make_shared_ptr_on_device(dst);
        this->view_ptr_ = this->ptr_.get();
    }

    Tensor::Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t> &shape, std::shared_ptr<void> ptr)
        : Tensor(data_type, device_type, shape, ptr, nullptr){}

    Tensor::Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape, std::shared_ptr<void> ptr, void* view_ptr)
            : data_type_(data_type), device_type_(device_type),
                shape_(shape), strides_(Tensor::calc_strides(shape)), ptr_(std::move(ptr)), view_ptr_(nullptr) {
        if (view_ptr != nullptr) {
            this->view_ptr_ = view_ptr;
        } else {
            this->view_ptr_ = this->ptr_.get();
        }
    }

    Tensor& Tensor::operator=(const Tensor& tensor) {
        if (this != &tensor) {
            this->ptr_ = tensor.ptr_;
            this->view_ptr_ = tensor.view_ptr_;
            this->data_type_ = tensor.data_type();
            this->device_type_ = tensor.device_type();
            this->shape_ = tensor.shape();
            this->strides_ = tensor.strides();
        }
        return *this;
    }

    Tensor::Tensor(Tensor&& tensor) noexcept
        : data_type_(DataType::Unknown), device_type_(DeviceType::Unknown),
        shape_(), strides_(), ptr_(nullptr), view_ptr_(nullptr) {
        *this = std::move(tensor);
    }

    Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
        if (this != &tensor) {
            this->data_type_ = tensor.data_type();
            this->device_type_ = tensor.device_type();
            this->shape_ = tensor.shape();
            this->strides_ = tensor.strides();
            this->ptr_ = std::move(tensor.ptr_);
            this->view_ptr_ = tensor.view_ptr_;
            tensor.view_ptr_ = nullptr;
        }
        return *this;
    }

    // 获取原生指针
    void* Tensor::raw_ptr() noexcept { return this->view_ptr_; }
    void* Tensor::raw_ptr() const noexcept { return this->view_ptr_; }

    std::shared_ptr<void> Tensor::shared_ptr() const noexcept { return this->ptr_; }

    // 获取底层引用计数
    long Tensor::use_count() const noexcept { return this->ptr_.use_count(); }

    // 获取形状
    const std::vector<std::size_t>& Tensor::shape() const noexcept { return this->shape_; }

    void Tensor::reshape(const std::vector<std::size_t>& shape) {
        if (this->size() != Tensor::calc_size(shape)) {
            throw std::invalid_argument("Tensor::reshape(): total elements not equal.");
        }
        this->shape_ = shape;
        this->strides_ = Tensor::calc_strides(shape);
    }

    // 获取当前数据类型
    DataType Tensor::data_type() const noexcept { return this->data_type_; }

    // 获取当前设备
    DeviceType Tensor::device_type() const noexcept { return this->device_type_; }

    // 判断是否为空tensor
    bool Tensor::empty() const noexcept {
        return this->data_type() == DataType::Unknown ||
            this->device_type() == DeviceType::Unknown ||
                this->size() == 0 ||
                this->shape().empty() ||
                    this->ptr_ == nullptr || this->view_ptr_ == nullptr;
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

    // 根据形状计算步长
    std::vector<std::size_t> Tensor::calc_strides(const std::vector<std::size_t>& shape) noexcept {
        std::vector<std::size_t> res;
        if (shape.empty()) {
            return res;
        }

        for (std::size_t cur_dim = 0; cur_dim < shape.size(); ++cur_dim) {
            std::size_t stride = 1;
            for (std::size_t next_dim = cur_dim + 1; next_dim < shape.size(); ++next_dim) {
                stride *= shape.at(next_dim);
            }
            res.push_back(stride);
        }
        return res;
    }

    Tensor Tensor::concat(const Tensor& x1, const Tensor& x2, ConcatDim concat_dim) {
        if (x1.empty()) {
            return x2;
        }
        if (x2.empty()) {
            return x1;
        }

        switch (concat_dim) {
            case ConcatDim::eRowWise:
                return kernel::ConcatByRowWiseOperator().forward({&x1, &x2}, nullptr);
            case ConcatDim::eColWise:
                return kernel::ConcatByColWiseOperator().forward({&x1, &x2}, nullptr);
            default:
                throw std::invalid_argument("Tensor::concat(): unsupported concat_dim");
        }
    }

    // 获取步长
    const std::vector<std::size_t>& Tensor::strides() const noexcept { return this->strides_; }

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
        // 校验
        if (Tensor::calc_size(indexes) > this->size()) {
            throw std::invalid_argument("indexes must below the number of elements");
        }
        // 计算
        std::size_t offset = 0;
        for (std::size_t i = 0; i < indexes.size(); ++i) {
            auto it = indexes.begin() + i;
            offset += (*it) * this->strides().at(i);
        }
        // 返回
        if (offset >= this->size()) {
            throw std::invalid_argument("indexes out of range");
        }
        offset *= data_type_size(this->data_type());
        return static_cast<char*>(this->raw_ptr()) + offset;
    }

    void Tensor::index_fill(const std::vector<std::size_t>& indexes, float val) {
        // 获取目标元素位置
        void* p = this->data(indexes);
        // 内存拷贝
        memcpy_with_data_type(this->device_type(), p, this->data_type(), val);
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
        fg42::memcpy_between_device(dst, src, this->bytes_size(), options);
        // 构造智能指针
        this->ptr_ = make_shared_ptr_on_device(dst);
        // 构建视图指针
        this->view_ptr_ = this->ptr_.get();
        // 切换设备
        this->device_type_ = device_type;
    }

    // 拷贝张量
    Tensor Tensor::clone(DeviceType device_type, MemcpyOptions* options) const {
        if (device_type == this->device_type()) {
            return *this;
        }

        Tensor t(this->data_type(), device_type, this->shape());
        PtrDeviceWrapper dst(t.device_type(), t.raw_ptr());
        PtrDeviceWrapper src(this->device_type(), this->raw_ptr());
        fg42::memcpy_between_device(dst, src, this->bytes_size(), options);
        return t;
    }

    // 张量运算
    Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
        std::vector<const Tensor*> input_tensors{&lhs, &rhs};

        kernel::AddOperator op;
        return op.forward(input_tensors, nullptr);
    }

    Tensor& Tensor::operator+=(const Tensor& tensor) {
        std::vector<const Tensor*> input_tensors{this, &tensor};

        kernel::AddOperator op;
        *this = op.forward(input_tensors, nullptr);
        return *this;
    }

    Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
        std::vector<const Tensor*> input_tensors{&lhs, &rhs};

        kernel::VecOuterOperator op;
        return op.forward(input_tensors, nullptr);
    }

    Tensor& Tensor::operator-=(const Tensor& tensor) {
        std::vector<const Tensor*> input_tensors{this, &tensor};

        kernel::VecOuterOperator op;
        *this = op.forward(input_tensors, nullptr);
        return *this;
    }

    Tensor Tensor::matmul(const Tensor& tensor) const {
        kernel::MatmulOperator op;
        return op.forward({this, &tensor}, nullptr);
    }

    Tensor Tensor::mul(const Tensor& tensor) const {
        std::vector<const Tensor*> input_tensors{this, &tensor};

        kernel::MulOperator op;
        return op.forward(input_tensors, nullptr);
    }

    Tensor Tensor::mul(float c) const {
        kernel::MulWithConstantValueOperator op(c);
        return op.forward({this}, nullptr);
    }

    Tensor Tensor::transpose() const {
        auto shape = this->shape();
        // 1. 行向量转置，dim均传0，会自动转置
        if (shape.size() == 1) {
            return this->transpose(0, 0);
        }
        // 2. 其余情况，默认转置后两维度
        return this->transpose(shape.size() - 2, shape.size() - 1);
    }

    Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
        kernel::TransposeOperator op(dim0, dim1);
        return op.forward({this}, nullptr);
    }

    Tensor Tensor::view() const {
        return this->view(this->shape(), this->raw_ptr());
    }

    Tensor Tensor::view(const std::vector<std::size_t> &shape) const {
        return this->view(shape, this->raw_ptr());
    }

    Tensor Tensor::view(const std::vector<std::size_t>& shape, void* data) const {
        if (shape.empty()) {
            throw std::invalid_argument("shape is empty");
        }

        std::size_t new_total_elements = Tensor::calc_size(shape);
        if (new_total_elements > this->size()) {
            throw std::invalid_argument("shape size too large");
        }

        Tensor t(this->data_type(), this->device_type(), shape, this->ptr_, data);
        return t;
    }

    void Tensor::copy_from(const Tensor& vec_tensor, std::size_t target_row) {
        if (this->shape().size() != 2) {
            throw std::invalid_argument("shape is not 2-dimensional");
        }

        if (vec_tensor.shape().size() != 1) {
            throw std::invalid_argument("input tensor is not vector");
        }

        PtrDeviceWrapper dst(this->device_type(), this->data({target_row, 0}));
        PtrDeviceWrapper src(vec_tensor.device_type(), vec_tensor.raw_ptr());
        fg42::memcpy_between_device(dst, src, vec_tensor.bytes_size());
    }

    // 检测两张量shape是否一致
    bool Tensor::shape_equal(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        if (a.empty()) {
            return true;
        }
        for (std::size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    Tensor Tensor::zeros(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape) {
        return Tensor::full(data_type, device_type, shape, 0.f);
    }
    Tensor Tensor::ones(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape) {
        return Tensor::full(data_type, device_type, shape, 1.f);
    }

    Tensor Tensor::full(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape, float val) {
        // 计算总元素数与单个元素所占字节数
        std::size_t total_elements = Tensor::calc_size(shape);
        std::size_t item_bytes = data_type_size(data_type);
        // 分配内存
        auto allocator = allocator_factory(device_type);
        PtrDeviceWrapper ptr = allocator->allocate(total_elements * item_bytes);
        // 赋值
        void* elem_ptr = ptr.raw_ptr();
        for (std::size_t i = 0; i < total_elements; i++) {
            memcpy_with_data_type(device_type, elem_ptr, data_type, val);
            elem_ptr = reinterpret_cast<void*>(static_cast<char*>(elem_ptr) + item_bytes);
        }
        // 构建tensor
        auto data = make_shared_ptr_on_device(ptr);
        return {data_type, device_type, shape, data};
    }

    std::size_t Tensor::calc_size(const std::vector<std::size_t>& shape) noexcept {
        std::size_t size = 1;
        for (std::size_t n : shape) {
            size *= n;
        }
        return size;
    }
}
