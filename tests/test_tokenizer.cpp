//
// Created by 19373 on 2025/9/6.
//
#include <gtest/gtest.h>
#include "tokenizer/AutoTokenizer.h"

TEST(TokenizerTest, EncodeDecode) {
    fg42::AutoTokenizer tokenizer(R"(C:\Users\19373\Downloads\tokenizer.json)");

    std::string prompt = "What is the capital of Canada?";
    auto ids = tokenizer.encode(prompt);
    std::string decoded_prompt = tokenizer.decode(ids);

    EXPECT_TRUE(prompt == decoded_prompt);
}