//
// Created by B777B2056-2 on 2025/9/30.
//

#ifndef FG42_TEST_OPERATORS_H
#define FG42_TEST_OPERATORS_H
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include "operator/BaseOperator.h"
#include "tensor/Tensor.h"
#include "util/util.h"
#include "nlohmann/json.hpp"

class OperatorTester final {
public:
    OperatorTester(const std::string& global_data_type,
        fg42::DeviceType device_type, const nlohmann::json& config);
    OperatorTester(const OperatorTester&) = delete;
    OperatorTester& operator=(const OperatorTester&) = delete;
    OperatorTester(OperatorTester&&) = default;
    OperatorTester& operator=(OperatorTester&&) = default;
    ~OperatorTester() = default;

    [[nodiscard]] std::string error_message() const;

    bool test();

private:
    std::string name_;
    fg42::DeviceType device_type_;
    std::vector<std::pair<std::string, fg42::Tensor>> weights_;
    std::vector<std::pair<std::string, fg42::Tensor>> biases_;
    std::vector<std::pair<std::string, fg42::Tensor>> inputs_;
    fg42::Tensor expected_output_;
    std::unique_ptr<fg42::kernel::BaseOperator> op_;

    void set_operator(const nlohmann::json& op_json);

    void add_input(const std::string& global_data_type, const nlohmann::json& input_json);
    void add_weight(const std::string& global_data_type, const nlohmann::json& weight_json);
    void add_bias(const std::string& global_data_type, const nlohmann::json& bias_json);

    [[nodiscard]] const fg42::Tensor& get_input_by_name(const std::string& name) const;
    [[nodiscard]] const fg42::Tensor& get_weight_by_name(const std::string& name) const;
    [[nodiscard]] const fg42::Tensor& get_bias_by_name(const std::string& name) const;

    void set_expected_output(const std::string& global_data_type, const nlohmann::json& expected_output_json);
};

class OperatorTesterManager {
public:
    OperatorTesterManager(const std::filesystem::path& test_cases_path, fg42::DeviceType device_type);
    OperatorTesterManager(const OperatorTesterManager&) = delete;
    OperatorTesterManager(OperatorTesterManager&&) = delete;
    OperatorTesterManager& operator=(const OperatorTesterManager&) = delete;
    OperatorTesterManager& operator=(OperatorTesterManager&&) = delete;
    ~OperatorTesterManager() = default;

    testing::AssertionResult run_test(const std::string& tester_name);

private:
    std::unordered_map<std::string, std::vector<std::unique_ptr<OperatorTester>>> testers_map_;
};

#endif //FG42_TEST_OPERATORS_H