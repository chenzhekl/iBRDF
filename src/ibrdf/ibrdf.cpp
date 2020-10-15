//
// Created by Zhe Chen on 2019/08/27.
//

#include "ibrdf.h"

#include <utility>

namespace ibrdf {
IBRDFImpl::IBRDFImpl(std::size_t numLayers,
                       std::size_t inFeatures,
                       std::size_t numEmbedDims,
                       std::size_t numPieces,
                       std::size_t numUNetBins,
                       TransformType transform)
  : mInFeatures(inFeatures)
{
  mLayers = register_module("layers", torch::nn::ModuleList());

  for (size_t i = 0; i < numLayers; ++i) {
    auto options = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor mask = torch::ones(inFeatures, options);
    mask[i % inFeatures] = 0;
    switch (transform) {
      case TransformType ::QUADRATIC:
        mLayers->push_back(PiecewiseQuadraticCoupling(
          inFeatures, mask, numPieces, numUNetBins, numEmbedDims));
        break;
      case TransformType ::LINEAR:
        throw std::runtime_error("Not implemented!");
        break;
      default:
        break;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
IBRDFImpl::forward(const torch::Tensor& x,
                    const std::optional<torch::Tensor>& embedCode,
                    bool reverse)
{
  torch::Tensor result = x;
  torch::Tensor logDet = torch::zeros(x.size(0), x.device());
  if (reverse) {
    for (std::int64_t i = mLayers->size() - 1; i >= 0; --i) {
      auto [x_, logDet_] =
        mLayers[i]->as<CouplingLayer>()->forward(result, embedCode, reverse);
      result = std::move(x_);
      logDet += logDet_;
    }
  } else {
    for (auto const& layer : *mLayers) {
      auto [x_, logDet_] =
        layer->as<CouplingLayer>()->forward(result, embedCode, reverse);
      result = std::move(x_);
      logDet += logDet_;
    }
  }

  return { result, logDet };
}
torch::Tensor
IBRDFImpl::sample(std::size_t numSamples,
                   const std::optional<torch::Tensor>& embedCode)
{
  torch::Device device = parameters()[0].device();
  torch::Tensor z = torch::rand(
    { static_cast<long>(numSamples), static_cast<long>(mInFeatures) }, device);

  [[maybe_unused]] auto [x, _] = forward(z, embedCode, true);
  (void)_; // unused

  return x;
}
} // namespace ibrdf
