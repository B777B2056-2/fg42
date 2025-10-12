//
// Created by B777B2056-2 on 2025/10/10.
//

#ifndef FG42_BASEWEIGHTSLOADER_H
#define FG42_BASEWEIGHTSLOADER_H
#include <filesystem>
#include <functional>
#if defined(__linux__) || defined(__APPLE__)
extern "C" {
#include <unistd.h>
}
#elif _WIN32
#include <fstream>
#endif
#include "model/Common.h"
#include "tensor/Tensor.h"

namespace fg42 {
    class BaseWeightsLoader {
    public:
        virtual ~BaseWeightsLoader() = default;

        virtual void load(const std::filesystem::path& path,
            StateDict& state_dict, DeviceType device_type, DataType data_type) = 0;

        void set_weight_need_transpose_func(std::function<bool(const std::string&)> func) {
            weight_need_transpose_func_ = std::move(func);
        }

    protected:
        std::function<bool(const std::string&)> weight_need_transpose_func_;
    };

    // 文件只读的平台相关实现（linux或macos: mmap，windows: std::ifstream）
#if defined(__linux__) || defined(__APPLE__)
    class LinuxWeightsFileReaderImpl {
    public:
        explicit LinuxWeightsFileReaderImpl(std::filesystem::path path);
        ~LinuxWeightsFileReaderImpl();

        [[nodiscard]] bool is_open() const;
        void read_into_tensor(Tensor& tensor);
        void read(void* buffer, std::size_t size);
        std::size_t offset();
        void set_offset(std::size_t pos);

    private:
        int fd_;
        off_t file_bytes_;
        char* mapped_mem_;

        std::size_t current_pos_;
        std::filesystem::path path_;
    };
#elif _WIN32
    class WindowsWeightsFileReaderImpl {
    public:
        explicit WindowsWeightsFileReaderImpl(std::filesystem::path path);
        ~WindowsWeightsFileReaderImpl() = default;

        [[nodiscard]] bool is_open() const;
        void read_into_tensor(Tensor& tensor);
        void read(void* buffer, std::size_t size);
        std::size_t offset();
        void set_offset(std::size_t pos);

    private:
        std::ifstream ifs_;
    };
#endif

    class WeightsFileReader {
    public:
        explicit WeightsFileReader(std::filesystem::path path);
        ~WeightsFileReader() = default;

        [[nodiscard]] bool is_open() const;
        void read_into_tensor(Tensor& tensor);
        void read(void* buffer, std::size_t size);
        std::size_t offset();
        void set_offset(std::size_t pos);

    private:
#if defined(__linux__) || defined(__APPLE__)
        using Impl = LinuxWeightsFileReaderImpl;
#elif _WIN32
        using Impl = WindowsWeightsFileReaderImpl;
#endif
        std::unique_ptr<Impl> pimpl_;
    };
} // fg42

#endif //FG42_BASEWEIGHTSLOADER_H