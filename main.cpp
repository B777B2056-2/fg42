//
// Created by B777B2056-2 on 2025/8/31.
//
#include <fstream>
#include "external/cmdline/cmdline.h"
#include "model/AutoModel.h"
#include "tokenizer/AutoTokenizer.h"
#include "util/util.h"


std::tuple<std::vector<std::string>, std::vector<std::vector<int32_t>>> get_input_ids(fg42::AutoTokenizer& tokenizer, cmdline::parser& a) {
    auto prompt = a.get<std::string>("prompt");
    auto prompts_file_path = a.get<std::string>("prompts_file_path");
    if (prompt.empty() && prompts_file_path.empty()) {
        throw std::invalid_argument("You must provide a valid prompt or prompts file path");
    }

    std::vector<std::string> prompts;
    std::vector<std::vector<int32_t>> input_ids;
    if (!prompt.empty()) {
        prompts.push_back(prompt);
        input_ids.emplace_back(tokenizer.encode(prompt));
    } else {
        std::ifstream file(prompts_file_path, std::ios::in);
        if(!file.is_open()) {
            throw std::invalid_argument("Failed to open file " + prompts_file_path);
        }

        std::string line;
        while(getline(file,line)) {
            if(line.empty())
                continue;
            prompts.emplace_back(line);
            input_ids.emplace_back(tokenizer.encode(line));
        }
    }
    return std::make_tuple(prompts, input_ids);
}


fg42::DeviceType get_device_type(const std::string& device_name) {
    if (device_name.empty()) {
        throw std::runtime_error("Device name is empty.");
    } else if (device_name == "cpu") {
        return fg42::DeviceType::CPU;
    } else if (device_name == "nvidia") {
        return fg42::DeviceType::NvidiaGPU;
    } else {
        throw std::runtime_error("Unknown device type.");
    }
}

fg42::SamplerConfig get_sampling_method(const std::string& sampling_method_name, float temperature) {
    fg42::SamplerConfig sampler_config{};
    if (sampling_method_name == "greedy") {
        sampler_config.method = fg42::SamplingMethod::GreedySampling;
    } else if (sampling_method_name == "temperature") {
        sampler_config.method = fg42::SamplingMethod::TemperatureSampling;
        sampler_config.temperature = temperature;
    } else {
        throw std::runtime_error("Unsupport sampling method.");
    }
    return sampler_config;
}


int main(int argc, char** argv) {
    cmdline::parser a;
    a.add<int>("port", 'p', "port number", false, 80, cmdline::range(1, 65535));
    a.add<std::string>("model_dir", 'm', "model directory", true, "");
    a.add<std::string>("device_type", 'd', "device type", true, "");
    a.add<std::string>("prompt", 'p', "prompt", false, "");
    a.add<std::string>("prompts_file_path", 'f', "prompts file path", false, "");
    a.add<std::size_t>("max_length", 'l', "max length", false, 20);
    a.add<bool>("enable_stream", 'e', "enable stream", false, false);
    a.add<std::string>("sampling_method", 's', "sampling method", true, "");
    a.add<float>("temperature", 't', "temperature", false, 1.0f, cmdline::range<float>(0.0f, 2.0f));
    a.parse_check(argc, argv);

    auto model_dir = a.get<std::string>("model_dir");
    auto device_type = get_device_type(fg42::util::to_lower(a.get<std::string>("device_type")));
    auto max_length = a.get<std::size_t>("max_length");
    auto enable_stream = a.get<bool>("enable_stream");
    // 1. 初始化tokenizer
    fg42::AutoTokenizer tokenizer(model_dir);

    // 2. 加载模型
    fg42::AutoModel model(model_dir, device_type, tokenizer.padding_idx());

    // 3. 获取采样器配置
    auto sampler_config = get_sampling_method(
        fg42::util::to_lower(a.get<std::string>("sampling_method")), 1.0f);

    // 4. 生成回答
    auto [prompts, input_ids] = get_input_ids(tokenizer, a);
    if (enable_stream) {
        fg42::BaseModel::StreamHandler stream_handler = [&tokenizer](std::size_t batch_idx, std::int32_t token) {
            auto answer = tokenizer.decode({token});
            std::cout << "Batch idx: " << batch_idx << ", answer: " << answer << std::endl;
        };
        model.generate(sampler_config, input_ids, max_length, stream_handler);
    } else {
        auto output_tokens = model.generate(sampler_config, input_ids, max_length);
        for (std::size_t batch_idx = 0; batch_idx < output_tokens.size(); ++batch_idx) {
            auto answer = tokenizer.decode(output_tokens[batch_idx]);
            std::cout << "Question: " << prompts[batch_idx] << std::endl;
            std::cout << "Answer: " << answer << std::endl;
            std::cout << std::endl;
        }
    }
    return 0;
}