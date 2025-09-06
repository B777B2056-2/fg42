//
// Created by 19373 on 2025/9/6.
//
#include <gtest/gtest.h>

int main(int argc, char **argv) {
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}