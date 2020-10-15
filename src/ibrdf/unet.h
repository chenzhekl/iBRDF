//
// Created by Zhe Chen on 2019/08/27.
//

#pragma once

#include <cstddef>

#include <torch/torch.h>

#include "util.h"

namespace ibrdf {
class UNetImpl : public torch::nn::Module
{
public:
  UNetImpl(std::size_t inFeatures,
           std::size_t outFeatures,
           std::size_t numBins);

  torch::Tensor forward(torch::Tensor x);

private:
  std::size_t mNumBins{ 1 };
  torch::nn::Linear mPre{ nullptr };
  torch::nn::Linear mE1{ nullptr }, mE2{ nullptr }, mE3{ nullptr },
    mE4{ nullptr };
  torch::nn::Linear mD1{ nullptr }, mD2{ nullptr }, mD3{ nullptr },
    mD4{ nullptr };
  torch::nn::Linear mPost{ nullptr };
};

TORCH_MODULE(UNet);
} // namespace ibrdf
