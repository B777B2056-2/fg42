//
// Created by 19373 on 2025/9/7.
//

#ifndef FG42_EMBEDDINGOPERATOR_H
#define FG42_EMBEDDINGOPERATOR_H
#include <optional>
#include "operator/BaseOperator.h"

namespace fg42 {
    namespace kernel {
        struct EmbeddingOperatorOptions {
            std::size_t num_embeddings;
            std::size_t embedding_dim;
            std::optional<float> padding_idx=std::nullopt;
            std::optional<float> max_norm=std::nullopt;
            float norm_type=2.0;
            bool scale_grad_by_freq=false;
            bool sparse=false;
        };

        class EmbeddingOperator : public BaseOperatorWithWeight {
        private:
            std::optional<EmbeddingOperatorOptions> options_;

        public:
            explicit EmbeddingOperator(const Tensor& weight_tensor,
                const std::string& name = "", std::optional<EmbeddingOperatorOptions> options = std::nullopt);
            ~EmbeddingOperator() override = default;

            Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

        private:
            // explicit EmbeddingOperator(const std::vector<const Tensor*>& weight_tensors, const std::string& name = "");
            bool check(const std::vector<const Tensor*>& input_tensors) const override;

            bool check_weights(const Tensor& weight_tensor) const override;
        };
    } // kernel
} // fg42

#endif //FG42_EMBEDDINGOPERATOR_H