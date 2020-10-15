//
// Created by Zhe Chen on 2019/08/27.
//

#pragma once

#include <optional>

#include <torch/torch.h>

namespace ibrdf {
class CouplingLayerImpl : public torch::nn::Module
{
public:
  explicit CouplingLayerImpl(const torch::Tensor& mask)
    : mMask(register_buffer("mask", mask))
    , mMaskInv(register_buffer("maskInv", torch::logical_not(mask)))
  {}

  std::tuple<torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& embedCode,
    bool reverse = false)
  {
    torch::Tensor xa = x.masked_select(mMask).view({ x.size(0), -1 });
    torch::Tensor xb = x.masked_select(mMaskInv).view({ x.size(0), -1 });
    auto [yb, det] = couple(xa, xb, embedCode, reverse);
    torch::Tensor y = x.masked_scatter(mMaskInv, yb);

    return { y, det };
  }

protected:
  virtual std::tuple<torch::Tensor, torch::Tensor> couple(
    const torch::Tensor& xa,
    const torch::Tensor& xb,
    const std::optional<torch::Tensor>& embedCode,
    bool reverse) = 0;

private:
  torch::Tensor mMask;
  torch::Tensor mMaskInv;
};

TORCH_MODULE(CouplingLayer);
} // namespace ibrdf
