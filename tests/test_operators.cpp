//
// Created by B777B2056-2 on 2025/10/4.
//
#include "tests/test_operators_impl.h"
#include <string>
#include <filesystem>

std::filesystem::path build_test_cases_path() {
    std::string current_src_file_path = __FILE__;
    std::filesystem::path dir_path = std::filesystem::path(current_src_file_path).parent_path() / "test_cases";
    return dir_path;
}

static OperatorTesterManager cpu_tester_manager = OperatorTesterManager(
    build_test_cases_path(), fg42::DeviceType::CPU);

TEST(AddOperator, CPU) {
    constexpr const char* tester_name = "add_with_board_cast";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(NegOperator, CPU) {
    constexpr const char* tester_name = "negation";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(OuterOperator, CPU) {
    constexpr const char* tester_name = "outer";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(MulOperator, CPU) {
    constexpr const char* tester_name = "mul";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(BatchMatmulOperator, CPU) {
    constexpr const char* tester_name = "matmul";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(MatrixTransposeOperator, CPU) {
    constexpr const char* tester_name = "transpose_last_two_dims";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(TensorTransposeOperator, CPU) {
    constexpr const char* tester_name = "transpose_first_two_dims";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(ArgmaxOperator, CPU) {
    constexpr const char* tester_name = "argmax";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(RotateHalfOperator, CPU) {
    constexpr const char* tester_name = "rotate_half";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(CosineOperator, CPU) {
    constexpr const char* tester_name = "cosine";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(SineOperator, CPU) {
    constexpr const char* tester_name = "sine";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(ConcatByRowWiseOperator, CPU) {
    constexpr const char* tester_name = "concat_by_row";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(ConcatByColWiseOperator, CPU) {
    constexpr const char* tester_name = "concat_by_col";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(CausalMaskOperator, CPU) {
    constexpr const char* tester_name = "bf16_causal_mask";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));

    constexpr const char* tester_name2 = "fp32_causal_mask";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name2));
}

TEST(SoftmaxOperator, CPU) {
    constexpr const char* tester_name = "softmax";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(SiLUOperator, CPU) {
    constexpr const char* tester_name = "silu";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(RMSNormOperator, CPU) {
    constexpr const char* tester_name = "rms_norm";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(EmbeddingOperator, CPU) {
    constexpr const char* tester_name = "embedding";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(AttentionOperator, CPU) {
    constexpr const char* tester_name = "sdpa";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(MultiQueryAttentionOperator, CPU) {
    constexpr const char* tester_name = "mqa";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

TEST(MultiQueryAttentionOperatorWithRoPE, CPU) {
    constexpr const char* tester_name = "mqa_with_rope";
    EXPECT_TRUE(cpu_tester_manager.run_test(tester_name));
}

#ifdef HAVE_CUDA

static OperatorTesterManager cuda_tester_manager = OperatorTesterManager(
    build_test_cases_path(), fg42::DeviceType::NvidiaGPU);

TEST(AddOperator, CUDA) {
    constexpr const char* tester_name = "add_with_board_cast";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(NegOperator, CUDA) {
    constexpr const char* tester_name = "negation";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(OuterOperator, CUDA) {
    constexpr const char* tester_name = "outer";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(MulOperator, CUDA) {
    constexpr const char* tester_name = "mul";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(BatchMatmulOperator, CUDA) {
    constexpr const char* tester_name = "matmul";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(MatrixTransposeOperator, CUDA) {
    constexpr const char* tester_name = "transpose_last_two_dims";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(TensorTransposeOperator, CUDA) {
    constexpr const char* tester_name = "transpose_first_two_dims";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(ArgmaxOperator, CUDA) {
    constexpr const char* tester_name = "argmax";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(RotateHalfOperator, CUDA) {
    constexpr const char* tester_name = "rotate_half";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(CosineOperator, CUDA) {
    constexpr const char* tester_name = "cosine";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(SineOperator, CUDA) {
    constexpr const char* tester_name = "sine";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(ConcatByRowWiseOperator, CUDA) {
    constexpr const char* tester_name = "concat_by_row";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(ConcatByColWiseOperator, CUDA) {
    constexpr const char* tester_name = "concat_by_col";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(CausalMaskOperator, CUDA) {
    constexpr const char* tester_name = "bf16_causal_mask";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));

    constexpr const char* tester_name2 = "fp32_causal_mask";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name2));
}

TEST(SoftmaxOperator, CUDA) {
    constexpr const char* tester_name = "softmax";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(SigmoidOperator, CUDA) {
    constexpr const char* tester_name = "sigmoid";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(SiLUOperator, CUDA) {
    constexpr const char* tester_name = "silu";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(RMSNormOperator, CUDA) {
    constexpr const char* tester_name = "rms_norm";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(EmbeddingOperator, CUDA) {
    constexpr const char* tester_name = "embedding";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(AttentionOperator, CUDA) {
    constexpr const char* tester_name = "sdpa";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(MultiQueryAttentionOperator, CUDA) {
    constexpr const char* tester_name = "mqa";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

TEST(MultiQueryAttentionOperatorWithRoPE, CUDA) {
    constexpr const char* tester_name = "mqa_with_rope";
    EXPECT_TRUE(cuda_tester_manager.run_test(tester_name));
}

#endif