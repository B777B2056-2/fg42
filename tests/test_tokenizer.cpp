//
// Created by B777B2056-2 on 2025/9/6.
//

#include <gtest/gtest.h>
#include <unordered_map>
#include "tokenizer/AutoTokenizer.h"

TEST(TokenizerTest, EncodeDecode) {
    fg42::AutoTokenizer tokenizer(R"(C:\Users\19373\Downloads)");

    std::string prompt = "What is the capital of Canada?";
    auto ids = tokenizer.encode(prompt);
    std::string decoded_prompt = tokenizer.decode(ids);

    EXPECT_TRUE(prompt == decoded_prompt);
}

TEST(TokenizerTest, ApplyChatTemplate) {
    fg42::AutoTokenizer tokenizer(R"(C:\Users\19373\Downloads)");

    std::vector<std::unordered_map<std::string, std::string>> messages = {
    {{"role", "system"}, {"content", "You are a friendly chatbot who always responds in the style of a pirate"},},
    {{"role", "user"}, {"content", "How many helicopters can a human eat in one sitting?"}},
     };
    auto messages_with_chat_template = tokenizer.apply_chat_template(messages);
    const std::string expected = R"(<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>

)";

    EXPECT_TRUE(messages_with_chat_template == expected);
}
