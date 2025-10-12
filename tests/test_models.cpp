//
// Created by B777B2056-2 on 2025/9/6.
//
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <string>
#include <gtest/gtest.h>
#include "model/Qwen2.h"
#include "tokenizer/AutoTokenizer.h"
#include "util/util.h"

// 固定prompts
static std::vector<std::string> prompts = {
    "What is the radius of the Earth in kilometers?",
    "What is the capital of China?"
};

// 对于prompts，预期会输出的关键词
static std::vector<std::vector<std::string>> expected_keywords = {
    {"kilometers"},
    {"capital"}
};

static void model_test(fg42::AutoTokenizer& tokenizer, fg42::BaseModel* model) {
    std::vector<std::vector<std::int32_t>> input_ids(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        input_ids[i] = tokenizer.encode(prompts[i]);
    }

    fg42::SamplerConfig sampler_config{};
    sampler_config.method = fg42::SamplingMethod::GreedySampling;

    constexpr std::size_t max_len = 20;
    auto output_tokens = model->generate(sampler_config, input_ids, max_len);

    // 检查关键词
    for (std::size_t batch_idx = 0; batch_idx < output_tokens.size(); ++batch_idx) {
        auto answer = fg42::util::to_lower(tokenizer.decode(output_tokens[batch_idx]));
        auto expected_keyword = expected_keywords[batch_idx];
        for (const auto& keyword : expected_keyword) {
            EXPECT_NE(answer.find(fg42::util::to_lower(keyword)), std::string::npos)
            << "Expected keyword " << keyword << " not found in answer: " << answer
            << " for question: " << prompts[batch_idx] << std::endl;
        }
    }
}

TEST(Qwen2Test, CPU) {
    std::string model_dir = std::getenv("QWEN2_MODEL_DIR");

    fg42::AutoTokenizer tokenizer(model_dir);
    fg42::Qwen2ForCausalLM model(model_dir, fg42::DeviceType::CPU,
        tokenizer.padding_idx(), fg42::DataType::FP32);

    model_test(tokenizer, &model);
}