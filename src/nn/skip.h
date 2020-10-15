//
// Created by Zhe Chen on 2019/09/09.
//

#pragma once

#include <torch/torch.h>

#include "stack_sequential.h"

class SkipImpl : public torch::nn::Module
{
public:
  SkipImpl(std::int64_t inFeatures,
           std::int64_t outFeatures,
           const std::vector<std::int64_t>& numChannelsDown,
           const std::vector<std::int64_t>& numChannelsUp,
           const std::vector<std::int64_t>& numChannelsSkip,
           std::int64_t filterSizeDown = 3,
           std::int64_t filterSizeUp = 3,
           std::int64_t filterSkipSize = 1,
           bool needBias = true,
           bool need1x1Up = true);
  torch::Tensor forward(const torch::Tensor& x);

private:
  StackSequentail mModel = nullptr;
  torch::Tensor mScale = torch::tensor(0.1f);
};

TORCH_MODULE(Skip);
