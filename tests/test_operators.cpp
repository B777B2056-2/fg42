//
// Created by 19373 on 2025/9/6.
//
#include "Eigen/Core"
#ifdef HAVE_CUDA
#include "memory/NvidiaGPUMemoryAllocator.h"
#endif
#include "memory/Common.h"
#include "operator/VecAddOperator.h"
#include "operator/EmbeddingOperator.h"
#include <gtest/gtest.h>


void test_add_operator_int(fg42::DeviceType device_type) {
    std::int8_t na[6]{1, 2, 3, 4, 5, 6};
    fg42::Tensor a(fg42::DataType::Int8, device_type, {2, 3});
    a.index_fill({0, 0}, &na[0]);
    a.index_fill({0, 1}, &na[1]);
    a.index_fill({0, 2}, &na[2]);
    a.index_fill({1, 0}, &na[3]);
    a.index_fill({1, 1}, &na[4]);
    a.index_fill({1, 2}, &na[5]);

    std::int8_t nb[6] = {1*2, 2*2, 3*2, 4*2, 5*2, 6*2};
    fg42::Tensor b(fg42::DataType::Int8, device_type, {2, 3});
    b.index_fill({0, 0}, &nb[0]);
    b.index_fill({0, 1}, &nb[1]);
    b.index_fill({0, 2}, &nb[2]);
    b.index_fill({1, 0}, &nb[3]);
    b.index_fill({1, 1}, &nb[4]);
    b.index_fill({1, 2}, &nb[5]);

    std::vector<const fg42::Tensor*> input_tensors({&a, &b});

    fg42::kernel::VecAddOperator op(device_type);
    auto output = op.forward(input_tensors, nullptr);

    if (output.device_type() != fg42::DeviceType::CPU) {
        output.to_device(fg42::DeviceType::CPU);
    }

    std::int8_t nc[2][3] = {{1*3, 2*3, 3*3}, {4*3, 5*3, 6*3}};
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            auto val = *static_cast<std::int8_t*>(output.data({i, j}));
            EXPECT_EQ(val, nc[i][j]);
        }
    }
}

bool is_float_equal(float a, float b) {
    return std::abs(a - b) < std::numeric_limits<float>::epsilon();
}

void test_add_operator_float(fg42::DeviceType device_type) {
    float na[6]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fg42::Tensor a(fg42::DataType::FP32, device_type, {2, 3});
    a.index_fill({0, 0}, &na[0]);
    a.index_fill({0, 1}, &na[1]);
    a.index_fill({0, 2}, &na[2]);
    a.index_fill({1, 0}, &na[3]);
    a.index_fill({1, 1}, &na[4]);
    a.index_fill({1, 2}, &na[5]);

    float nb[6]{1.0f*2, 2.0f*2, 3.0f*2, 4.0f*2, 5.0f*2, 6.0f*2};
    fg42::Tensor b(fg42::DataType::FP32, device_type, {2, 3});
    b.index_fill({0, 0}, &nb[0]);
    b.index_fill({0, 1}, &nb[1]);
    b.index_fill({0, 2}, &nb[2]);
    b.index_fill({1, 0}, &nb[3]);
    b.index_fill({1, 1}, &nb[4]);
    b.index_fill({1, 2}, &nb[5]);

    std::vector<const fg42::Tensor*> input_tensors({&a, &b});

    fg42::kernel::VecAddOperator op(device_type);
    auto output = op.forward(input_tensors, nullptr);

    if (output.device_type() != fg42::DeviceType::CPU) {
        output.to_device(fg42::DeviceType::CPU);
    }

    float nc[2][3] = {{1.0f*3, 2.0f*3, 3.0f*3}, {4.0f*3, 5.0f*3, 6.0f*3}};
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            float val = *static_cast<float*>(output.data({i, j}));
            EXPECT_TRUE(is_float_equal(val, nc[i][j]));
        }
    }
}

void test_add_operator_bfloat16(fg42::DeviceType device_type) {
    using bfloat16 = Eigen::bfloat16;
    bfloat16 na[6] {
        bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f),
        bfloat16(4.0f), bfloat16(5.0f), bfloat16(6.0f)
    };
    fg42::Tensor a(fg42::DataType::BF16, device_type, {2, 3});
    a.index_fill({0, 0}, &na[0]);
    a.index_fill({0, 1}, &na[1]);
    a.index_fill({0, 2}, &na[2]);
    a.index_fill({1, 0}, &na[3]);
    a.index_fill({1, 1}, &na[4]);
    a.index_fill({1, 2}, &na[5]);

    bfloat16 nb[6] {
        bfloat16(1.0f*2), bfloat16(2.0f*2), bfloat16(3.0f*2),
        bfloat16(4.0f*2), bfloat16(5.0f*2), bfloat16(6.0f*2)
    };
    fg42::Tensor b(fg42::DataType::BF16, device_type, {2, 3});
    b.index_fill({0, 0}, &nb[0]);
    b.index_fill({0, 1}, &nb[1]);
    b.index_fill({0, 2}, &nb[2]);
    b.index_fill({1, 0}, &nb[3]);
    b.index_fill({1, 1}, &nb[4]);
    b.index_fill({1, 2}, &nb[5]);

    std::vector<const fg42::Tensor*> input_tensors({&a, &b});

    fg42::kernel::VecAddOperator op(device_type);
    auto output = op.forward(input_tensors, nullptr);

    if (output.device_type() != fg42::DeviceType::CPU) {
        output.to_device(fg42::DeviceType::CPU);
    }

    bfloat16 nc[2][3] = {
        {
            bfloat16(1.0f*3), bfloat16(2.0f*3), bfloat16(3.0f*3)
        }, {
            bfloat16(4.0f*3), bfloat16(5.0f*3), bfloat16(6.0f*3)
        }
    };
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            auto val = static_cast<float>(*static_cast<bfloat16*>(output.data({i, j})));
            EXPECT_TRUE(is_float_equal(val, nc[i][j]));
        }
    }
}

TEST(VecAddOperatorTest, CPU) {
    test_add_operator_int(fg42::DeviceType::CPU);
    test_add_operator_float(fg42::DeviceType::CPU);
    test_add_operator_bfloat16(fg42::DeviceType::CPU);
}

#ifdef HAVE_CUDA
TEST(VecAddOperatorTest, CUDA) {
    test_add_operator_int(fg42::DeviceType::NvidiaGPU);
    test_add_operator_float(fg42::DeviceType::NvidiaGPU);
    test_add_operator_bfloat16(fg42::DeviceType::NvidiaGPU);
}
#endif

void test_embedding_operator_bfloat16(fg42::DeviceType device_type) {
    using bfloat16 = Eigen::bfloat16;

    // embedding权重赋值，只有0和1两个索引位
    bfloat16 na[2][3] {
        {bfloat16(11.0f), bfloat16(21.0f), bfloat16(31.0f)},
        {bfloat16(41.0f), bfloat16(51.0f), bfloat16(61.0f)},
    };
    fg42::Tensor embedding_tensor(fg42::DataType::BF16, device_type, {2, 3});
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            embedding_tensor.index_fill({i, j}, &na[i][j]);
        }
    }

    // 输入向量赋值，input_id只能为0或1（因为embedding权重只有0和1两个索引位）
    std::int32_t nb[4][6] {
        {1%2, 2%2, 3%2, 4%2, 5%2, 6%2},         // 0,0,1,0,1,0
        {7%2, 8%2, 9%2, 10%2, 11%2, 12%2},      // 1,0,1,0,1,0
        {13%2, 14%2, 15%2, 16%2, 17%2, 18%2},   // 1,0,1,0,1,0
        {19%2, 20%2, 21%2, 22%2, 23%2, 24%2},   // 1,0,1,0,1,0
    };
    fg42::Tensor input_tensor(fg42::DataType::Int32, fg42::DeviceType::CPU, {4, 6});
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            input_tensor.index_fill({i, j}, &nb[i][j]);
        }
    }

    // embedding运算
    fg42::kernel::EmbeddingOperator op(embedding_tensor);
    auto output = op.forward({&input_tensor}, nullptr);

    // 检查输出张量的维度
    ASSERT_EQ(output.shape().at(0), input_tensor.shape().at(0));
    ASSERT_EQ(output.shape().at(1), input_tensor.shape().at(1));
    ASSERT_EQ(output.shape().at(2), embedding_tensor.shape().at(1));

    // 检查输出张量数值
    for (std::size_t b = 0; b < 4; ++b) {
        for (std::size_t s = 0; s < 6; ++s) {
            for (std::size_t t = 0; t < 3; ++t) {
                float expect = na[nb[b][s]][t];
                auto val = static_cast<float>(*static_cast<bfloat16*>(output.data({b, s, t})));
                EXPECT_TRUE(is_float_equal(val, expect));
            }
        }
    }
}

TEST(EmbeddingOperatorTest, CPU) {
    test_embedding_operator_bfloat16(fg42::DeviceType::CPU);
}