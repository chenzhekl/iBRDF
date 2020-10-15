//
// Created by Zhe Chen on 2019/08/27.
//

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>

#include "piecewise_quadratic.h"

namespace ibrdf {
enum class TransformType
{
  LINEAR,
  QUADRATIC,
};

class IBRDFImpl : public torch::nn::Module
{
public:
  IBRDFImpl(std::size_t numLayers,
             std::size_t inFeatures,
             std::size_t numEmbedDims = 0,
             std::size_t numPieces = 32,
             std::size_t numUNetBins = 1,
             TransformType transform = TransformType::QUADRATIC);

  std::tuple<torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& embedCode = std::nullopt,
    bool reverse = false);

  torch::Tensor sample(
    std::size_t numSamples,
    const std::optional<torch::Tensor>& embedCode = std::nullopt);

  torch::Tensor logPdf(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& embedCode = std::nullopt)
  {
    [[maybe_unused]] auto [_, logDet] = forward(x, embedCode);
    (void)_; // unused

    return logDet;
  }

private:
  std::size_t mInFeatures;
  torch::nn::ModuleList mLayers;
};

TORCH_MODULE(IBRDF);
} // namespace ibrdf
