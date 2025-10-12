//
// Created by B777B2056-2 on 2025/10/11.
//

#include "cache/KVCache.h"

#include "memory/BaseAllocator.h"
#include "memory/Common.h"
#include "operator/Attention.h"

namespace fg42 {
    BaseKVCache::BaseKVCache(const ModelConfig& model_config) : model_config_(model_config) {}

    DynamicKVCache::DynamicKVCache(const ModelConfig& model_config)
    : BaseKVCache(model_config),
    k_cached_(model_config.num_hidden_layers), v_cached_(model_config.num_hidden_layers) {}

    const Tensor& DynamicKVCache::k_cached(std::size_t layer_idx) const {
        if (layer_idx >= k_cached_.size()) {
            throw std::runtime_error("layer_idx out of range.");
        }

        return k_cached_.at(layer_idx);
    }

    const Tensor& DynamicKVCache::v_cached(std::size_t layer_idx) const {
        if (layer_idx >= v_cached_.size()) {
            throw std::runtime_error("layer_idx out of range.");
        }

        return v_cached_.at(layer_idx);
    }

    Tensor DynamicKVCache::apply_attention(std::size_t layer_idx, const Tensor& query, const Tensor& last_key,
        const Tensor& last_value, void* stream) {
        std::size_t num_heads = model_config_.num_attention_heads;
        std::size_t num_kv_heads = model_config_.num_key_value_heads;

        auto key = this->append_last_k(layer_idx, last_key);
        auto value = this->append_last_v(layer_idx, last_value);

        kernel::ScaledDotProductAttention sdpa(num_heads, num_kv_heads);
        return sdpa.forward({&query, &key, &value}, stream);
    }

    const Tensor& DynamicKVCache::append_last_k(std::size_t layer_idx, const Tensor& last_k) {
        if (last_k.empty()) {
            throw std::runtime_error("last_k must not be empty.");
        }

        if (layer_idx >= k_cached_.size()) {
            throw std::runtime_error("layer_idx out of range.");
        }

        if (k_cached_[layer_idx].empty()) {    // prefill 阶段
            k_cached_[layer_idx] = last_k;
        } else {    // decode 阶段
            k_cached_[layer_idx] = Tensor::concat(k_cached_[layer_idx], last_k, Tensor::ConcatDim::eRowWise);
        }
        return k_cached_[layer_idx];
    }

    const Tensor& DynamicKVCache::append_last_v(std::size_t layer_idx, const Tensor& last_v) {
        if (last_v.empty()) {
            throw std::runtime_error("last_v must not be empty.");
        }

        if (layer_idx >= v_cached_.size()) {
            throw std::runtime_error("layer_idx out of range.");
        }

        if (v_cached_[layer_idx].empty()) {    // prefill 阶段
            v_cached_[layer_idx] = last_v;
        } else {    // decode 阶段
            v_cached_[layer_idx] = Tensor::concat(v_cached_[layer_idx], last_v, Tensor::ConcatDim::eRowWise);
        }
        return v_cached_[layer_idx];
    }

    void DynamicKVCache::clear() {
        this->k_cached_.clear();
        this->v_cached_.clear();
    }

    std::unique_ptr<BaseKVCache> kv_cache_factory(KVCacheImpl impl,
        DeviceType device_type, const ModelConfig& model_config) {
        switch (impl) {
            case KVCacheImpl::Dynamic:
                return std::make_unique<DynamicKVCache>(model_config);
            default:
                throw std::runtime_error("Unknown KVCacheImpl type.");
        }
    }
} // fg42