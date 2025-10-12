cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# 设置项目名称和支持的语言
project(FG42 LANGUAGES CXX)  # 移除了CUDA语言依赖

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows下特殊编译命令
if (MSVC)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options(/bigobj)
endif()

# 查找CUDA工具包（改为可选查找）
find_package(CUDAToolkit QUIET)

set(CUDAToolkit_FOUND false)

# 根据CUDA是否可用设置宏和变量
if(CUDAToolkit_FOUND)
    # 启用CUDA语言支持
    enable_language(CUDA)

    # 设置CUDA标准
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # 定义HAVE_CUDA宏
    add_definitions(-DHAVE_CUDA)

    # 递归查找所有.cu文件（CUDA源文件）
    file(GLOB_RECURSE CUDA_SOURCES "operator/cuda/*.cu")

    # 设置CUDA架构
    set(CUDA_ARCHITECTURES "50;61;75;80;86")
else()
    # 没有CUDA时清空CUDA源文件列表
    set(CUDA_SOURCES "")
endif()

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

# 添加Jinja2cpp库
set(JINJA2CPP_BUILD_TESTS OFF)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/bin")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_DATAROOTDIR "${CMAKE_INSTALL_PREFIX}/data")
set(JINJA2CPP_DEPS_MODE internal)
set(JINJA2CPP_INSTALL ON)
add_subdirectory("external/Jinja2Cpp")
install(TARGETS)
set(JINJA2CPP_INCLUDE_DIRS "external/Jinja2Cpp/include")

file(GLOB_RECURSE CPP_TOKENIZER_SOURCES "tokenizer/*.cpp")

# 递归查找所有.cpp文件
file(GLOB_RECURSE CPP_SOURCES
        "cache/*.cpp"
        "memory/*.cpp"
        "operator/*.cpp"
        "tensor/*.cpp"
        "model/*.cpp"
        "sampler/*.cpp"
        "util/*.cpp"
)

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp ${CPP_SOURCES} ${CPP_TOKENIZER_SOURCES} ${CUDA_SOURCES})

# 包含目录
include_directories(${PROJECT_NAME} PRIVATE
        "."
        ${EIGEN3_INCLUDE_DIRS}
        ${JSON_INCLUDE_DIRS}
        ${JINJA2CPP_INCLUDE_DIRS}
)

# 条件包含CUDA目录
if(CUDAToolkit_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif()

link_directories()

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        tokenizers_cpp
        jinja2cpp
)

# 条件链接CUDA库
if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::cudart
    )

    # 设置CUDA相关属性
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# 包含单元测试
add_subdirectory("tests")