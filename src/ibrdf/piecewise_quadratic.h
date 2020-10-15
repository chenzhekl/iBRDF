//
// Created by Zhe Chen on 2019/08/27.
//

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include <torch/torch.h>

#include "coupling_layer.h"
#include "unet.h"

namespace ibrdf {
class PiecewiseQuadraticCouplingImpl : public CouplingLayerImpl
{
public:
  PiecewiseQuadraticCouplingImpl(std::size_t inFeatures,
                                 const torch::Tensor& mask,
                                 std::size_t k = 32,
                                 std::size_t uNetBins = 1,
                                 std::size_t numEmbedDim = 0)
    : CouplingLayerImpl(mask)
    , mK(k)
  {
    std::size_t uNetIn = mask.sum().item().toLong() + numEmbedDim;
    std::size_t uNetOut =
      (inFeatures - mask.sum().item().toLong()) * (k + k + 1);
    // std::size_t uNetOut = (inFeatures - mask.sum().item().toLong()) * (k + 1);
    mNetM = register_module("netM", UNet(uNetIn, uNetOut, uNetBins));
  }

protected:
  std::tuple<torch::Tensor, torch::Tensor> couple(
    const torch::Tensor& xa,
    const torch::Tensor& xb,
    const std::optional<torch::Tensor>& embedCode,
    bool reverse) override
  {
    if (reverse) {
      return coupleInverse(xa, xb, embedCode);
    } else {
      return coupleForward(xa, xb, embedCode);
    }
  }

private:
  std::tuple<torch::Tensor, torch::Tensor> coupleForward(
    const torch::Tensor& xa,
    const torch::Tensor& xb,
    const std::optional<torch::Tensor>& embedCode);
  std::tuple<torch::Tensor, torch::Tensor> coupleInverse(
    const torch::Tensor& ya,
    const torch::Tensor& yb,
    const std::optional<torch::Tensor>& embedCode);
  std::tuple<torch::Tensor, torch::Tensor> getParams(
    const torch::Tensor& xa,
    const torch::Tensor& xb,
    const std::optional<torch::Tensor>& embedCode);

  std::size_t mK = 32;
  UNet mNetM{ nullptr };
};

TORCH_MODULE(PiecewiseQuadraticCoupling);

} // namespace ibrdf
