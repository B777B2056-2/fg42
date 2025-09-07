//
// Created by 19373 on 2025/9/7.
//
#include "operator/cpu/embedding.h"

#include "memory/Common.h"

namespace fg42::kernel {
    Tensor embedding_kernel_cpu(const Tensor* weight_tensor, const Tensor& input_tensor) {
        // 获取权重张量的信息
        const auto embedding_dim = weight_tensor->shape().at(1);
        // 获取输入张量信息
        std::size_t batch_size = input_tensor.shape().at(0);
        std::size_t seq_length = input_tensor.shape().at(1);
        // 设置输出张量维度
        Tensor output(weight_tensor->data_type(), weight_tensor->device_type(),
            {batch_size, seq_length, embedding_dim});
        // 对于批量中的每个样本
        for (std::size_t b = 0; b < batch_size; ++b) {
            // 每一句拷贝embedding矩阵里的对应行向量
            for (std::size_t s = 0; s < seq_length; ++s) {
                // 获取当前batch的当前句子的当前input_id
                auto input_id = static_cast<std::size_t>(
                    *static_cast<const std::int32_t*>(input_tensor.data({b, s})));
                // 以input_id为索引，查找embedding矩阵
                void* weight_data = weight_tensor->data({input_id});
                // 拷贝到输出
                void* out_ptr = output.data({b, s});
                PtrDeviceWrapper dst(output.device_type(), out_ptr);
                PtrDeviceWrapper src(weight_tensor->device_type(), weight_data);
                fg42::memcpy(dst, src, embedding_dim * data_type_size(output.data_type()));
            }
        }
        return output;
    }
}
