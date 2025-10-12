//
// Created by B777B2056-2 on 2025/9/1.
//

#ifndef FG42_TENSOR_H
#define FG42_TENSOR_H
#include <vector>
#include <memory>
#include <functional>
#include "util/enum.h"

namespace fg42 {
    struct MemcpyOptions;

    // 张量（按行存储）
    class Tensor {
    private:
        DataType data_type_;
        DeviceType device_type_;            // 设备类型
        std::vector<std::size_t> shape_;    // tensor形状
        std::vector<std::size_t> strides_;  // tensor步长（根据shape计算）
        std::shared_ptr<void> ptr_;         // 底层内存指针
        void* view_ptr_;

    public:
        enum class ConcatDim {
            eRowWise = 0,
            eColWise = 1,
        };

        Tensor();

        // 需自动分配内存的构造
        Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape);

        // 直接管理已分配内存的构造
        Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape,
            std::shared_ptr<void> ptr);

        // 拷贝
        Tensor(const Tensor& tensor) = default;
        Tensor& operator=(const Tensor& tensor);

        // 移动
        Tensor(Tensor&& tensor) noexcept;
        Tensor& operator=(Tensor&& tensor) noexcept;

        // 析构
        ~Tensor() = default;

        // 获取原生指针
        [[nodiscard]] void* raw_ptr() noexcept;
        [[nodiscard]] void* raw_ptr() const noexcept;

        // 获取底层智能指针
        [[nodiscard]] std::shared_ptr<void> shared_ptr() const noexcept;

        // 获取底层引用计数
        [[nodiscard]] long use_count() const noexcept;

        // 获取形状
        [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;

        // 重构形状
        void reshape(const std::vector<std::size_t>& shape);

        // 获取当前数据类型
        [[nodiscard]] DataType data_type() const noexcept;

        // 获取当前设备
        [[nodiscard]] DeviceType device_type() const noexcept;

        // 判断是否为空tensor
        [[nodiscard]] bool empty() const noexcept;

        // 获取大小
        [[nodiscard]] std::size_t size() const noexcept;

        // 获取字节长度
        [[nodiscard]] std::size_t bytes_size() const noexcept;

        // 获取步长
        [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept;

        // 按索引取值
        [[nodiscard]] void* data(const std::vector<std::size_t>& indexes) const;

        // 按索引赋值
        void index_fill(const std::vector<std::size_t>& indexes, float val);

        // 切换底层硬件
        void to_device(DeviceType device_type, MemcpyOptions* options = nullptr);

        // 拷贝张量
        [[nodiscard]] Tensor clone(DeviceType device_type, MemcpyOptions* options = nullptr) const;

        // 张量运算
        Tensor& operator+=(const Tensor& tensor);  // 矩阵相加
        Tensor& operator-=(const Tensor& tensor);  // 矩阵相减
        [[nodiscard]] Tensor matmul(const Tensor& tensor) const;        // 矩阵相乘
        [[nodiscard]] Tensor mul(const Tensor& tensor) const;           // 元素逐个相乘
        [[nodiscard]] Tensor mul(float c) const;                        // 元素逐个乘以常数
        [[nodiscard]] Tensor transpose() const;                         // 转置，多维只交换最后两个维度的数据，等价于.transpose(-1,-2)
        [[nodiscard]] Tensor transpose(std::size_t dim0, std::size_t dim1) const;   // 交换两个维度

        // 视图
        [[nodiscard]] Tensor view() const;
        [[nodiscard]] Tensor view(const std::vector<std::size_t>& shape) const;
        [[nodiscard]] Tensor view(const std::vector<std::size_t>& shape, void* data) const;

        // 按行拷贝（仅限二维张量）
        void copy_from(const Tensor& vec_tensor, std::size_t target_row);

        // 转换为FP32
        [[nodiscard]] Tensor to_float() const;

        // 转换为BF16
        [[nodiscard]] Tensor to_bf16() const;

        // 检测两张量shape是否一致
        static bool shape_equal(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b);

        // 生成全0张量
        static Tensor zeros(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape);

        // 生成全1张量
        static Tensor ones(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape);

        // 生成同值张量
        static Tensor full(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape, float val);

        // 计算指定维度的张量元素个数
        static std::size_t calc_size(const std::vector<std::size_t>& shape) noexcept;

        // 根据形状计算步长
        static std::vector<std::size_t> calc_strides(const std::vector<std::size_t>& shape) noexcept;

        // 拼接
        static Tensor concat(const Tensor& x1, const Tensor& x2, ConcatDim concat_dim);

    private:
        // 视图构造
        Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape,
            std::shared_ptr<void> ptr, void* view_ptr);
    };

    Tensor operator+(const Tensor& lhs, const Tensor& rhs);   // 矩阵相加
    Tensor operator-(const Tensor& lhs, const Tensor& rhs);   // 矩阵相减
} // fg42

#endif //FG42_TENSOR_H