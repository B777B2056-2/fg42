cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# 设置项目名称和支持的语言
project(FG42_TESTS LANGUAGES CXX)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(HAVE_CUDA)
    # 启用CUDA语言支持
    enable_language(CUDA)

    # 设置CUDA标准
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

endif ()

# 添加google test库
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW ON
)
FetchContent_MakeAvailable(googletest)

# 递归查找所有.cpp文件
file(GLOB_RECURSE CPP_TEST_SOURCES "test_*.cpp")

message(STATUS "CPP_TEST_SOURCES: ${CPP_TEST_SOURCES}")

# 包含目录
include_directories(${PROJECT_NAME} PRIVATE
        ..
        ${EIGEN3_INCLUDE_DIRS}
        ${JSON_INCLUDE_DIRS}
        ${JINJA2CPP_INCLUDE_DIRS}
)

if (DEFINED HAVE_CUDA)
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif ()

# 添加可执行文件
add_executable(${PROJECT_NAME} ${CPP_TEST_SOURCES} ${CPP_SOURCES} ${CPP_TOKENIZER_SOURCES} ${CUDA_SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        tokenizers_cpp
        jinja2cpp
        GTest::gtest_main
)

if (DEFINED HAVE_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::cudart
    )
endif ()