//
// Created by 19373 on 2025/10/20.
//

#ifndef FG42_AUTOMODEL_H
#define FG42_AUTOMODEL_H
#include <string>
#include <vector>
#include <memory>

#include "BaseModel.h"
#include "cache/KVCache.h"
#include "sampler/Sampler.h"
#include "util/enum.h"

namespace fg42 {
    class AutoModel final {
    public:
        AutoModel() = delete;
        AutoModel(const std::string& dir_path, DeviceType device_type,
            std::int32_t padding_idx, KVCacheImpl kv_cache_impl=KVCacheImpl::Dynamic);
        AutoModel(const AutoModel&) = delete;
        AutoModel& operator=(const AutoModel&) = delete;
        AutoModel(AutoModel&&) = delete;
        AutoModel& operator=(AutoModel&&) = delete;
        ~AutoModel() = default;

        std::vector<std::vector<std::int32_t>> generate(SamplerConfig sampler_config,
            const std::vector<std::vector<std::int32_t>>& input_ids, std::size_t max_length,
            BaseModel::StreamHandler stream_handler=nullptr);

    private:
        std::unique_ptr<BaseModel> model_;
    };
} // fg42

#endif //FG42_AUTOMODEL_H