cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# 设置项目名称和支持的语言
project(FG42 LANGUAGES CXX CUDA)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 下载Eigen3.4库
include(FetchContent)
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen)
set(EIGEN3_INCLUDE_DIRS ${FETCHCONTENT_BASE_DIR}/eigen-src/)

# 下载json库
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)
set(JSON_INCLUDE_DIRS ${FETCHCONTENT_BASE_DIR}/json-src/include/)

# 添加tokenizers-cpp库
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
add_subdirectory("external/tokenizers-cpp")
install(TARGETS tokenizers_cpp)

# 递归查找所有.cpp文件
file(GLOB_RECURSE CPP_SOURCES
    "memory/*.cpp"
    "operator/*.cpp"
    "tensor/*.cpp"
    "model/*.cpp"
    "tokenizer/*.cpp"
    "util/*.cpp"
)

# 递归查找所有.cu文件（CUDA源文件）
file(GLOB_RECURSE CUDA_SOURCES "operator/cuda/*.cu")

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp ${CPP_SOURCES} ${CUDA_SOURCES})

# 包含目录
include_directories(${PROJECT_NAME} PRIVATE
        "."
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${EIGEN3_INCLUDE_DIRS}
        ${JSON_INCLUDE_DIRS}
)

message(STATUS "EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
        CUDA::cudart
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        tokenizers_cpp
)

# 设置CUDA架构
set(CUDA_ARCHITECTURES "50;61;75;80;86")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 包含单元测试
add_subdirectory("tests")
