//
// Created by 19373 on 2025/9/1.
//

#ifndef FG42_TENSOR_H
#define FG42_TENSOR_H
#include <vector>
#include <memory>
#include <optional>
#include "util/enum.h"

namespace fg42 {
    struct MemcpyOptions;

    // 张量（按行存储）
    class Tensor {
    private:
        DataType data_type_;
        DeviceType device_type_;            // 设备类型
        std::vector<std::size_t> shape_;    // tensor形状
        std::shared_ptr<void> ptr_;            // 底层内存指针

    public:
        // 需自动分配内存的构造
        Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape);

        // 直接管理已分配内存的构造
        Tensor(DataType data_type, DeviceType device_type, const std::vector<std::size_t>& shape, std::shared_ptr<void> ptr);

        // 拷贝
        Tensor(const Tensor& tensor);
        Tensor& operator=(const Tensor& tensor);

        // 移动
        Tensor(Tensor&& tensor) noexcept;
        Tensor& operator=(Tensor&& tensor) noexcept;

        // 析构
        ~Tensor() = default;

        // 获取原生指针
        [[nodiscard]] void* raw_ptr() noexcept;
        [[nodiscard]] void* raw_ptr() const noexcept;

        // 获取底层引用计数
        [[nodiscard]] long use_count() const noexcept;

        // 获取形状
        [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;

        // 修改形状
        std::optional<Tensor> reshape(const std::vector<std::size_t>& new_shape, bool copy = false);

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
        [[nodiscard]] std::vector<std::size_t> strides() const noexcept;

        // 按索引取值
        void* data(const std::vector<std::size_t>& indexes) const;

        // 按索引赋值
        void index_fill(const std::vector<std::size_t>& indexes, void* val);

        // 切换底层硬件
        void to_device(DeviceType device_type, MemcpyOptions* options = nullptr);

        // 拷贝张量
        Tensor clone(DeviceType device_type, MemcpyOptions* options = nullptr) const;

        // 检测两张量shape是否一致
        static bool shape_equal(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b);
    private:
        static std::size_t calc_size(const std::vector<std::size_t>& shape) noexcept;
    };
} // fg42

#endif //FG42_TENSOR_H