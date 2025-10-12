//
// Created by B777B2056-2 on 2025/10/11.
//

#ifndef FG42_KVCACHE_H
#define FG42_KVCACHE_H
#include <vector>
#include <memory>
#include <unordered_map>
#include "tensor/Tensor.h"
#include "model/Common.h"

namespace fg42 {
    enum class KVCacheImpl : std::int8_t {
        None = 0,
        Dynamic = 1,
    };

    class BaseKVCache {
    public:
        explicit BaseKVCache(const ModelConfig& model_config);
        BaseKVCache(const BaseKVCache&) = delete;
        BaseKVCache(BaseKVCache&&) = delete;
        BaseKVCache& operator=(const BaseKVCache&) = delete;
        BaseKVCache& operator=(BaseKVCache&&) = delete;
        virtual ~BaseKVCache() = default;

        [[nodiscard]] virtual Tensor apply_attention(std::size_t layer_idx,
            const Tensor& query, const Tensor& last_key, const Tensor& last_value, void* stream) = 0;

        virtual void clear() = 0;

    protected:
        ModelConfig model_config_;
    };

    class DynamicKVCache final : public BaseKVCache {
    public:
        explicit DynamicKVCache(const ModelConfig& model_config);
        ~DynamicKVCache() override = default;

        Tensor apply_attention(std::size_t layer_idx,
            const Tensor& query, const Tensor& last_key, const Tensor& last_value, void* stream) override;

        void clear() override;

    private:
        std::vector<Tensor> k_cached_;
        std::vector<Tensor> v_cached_;

        [[nodiscard]] const Tensor& k_cached(std::size_t layer_idx) const;
        [[nodiscard]] const Tensor& v_cached(std::size_t layer_idx) const;

        const Tensor& append_last_k(std::size_t layer_idx, const Tensor& last_k);
        const Tensor& append_last_v(std::size_t layer_idx, const Tensor& last_v);
    };

    std::unique_ptr<BaseKVCache> kv_cache_factory(KVCacheImpl impl,
        DeviceType device_type, const ModelConfig& model_config);
} // fg42

#endif //FG42_KVCACHE_H