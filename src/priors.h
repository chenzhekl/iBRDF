#pragma once

#include <torch/torch.h>

torch::Tensor
EntropyPrior(const torch::Tensor& img, std::int64_t bincount, float sigma);

torch::Tensor
EntropyPriorD(const torch::Tensor& img,
              const torch::Tensor& hist,
              std::int64_t bincount,
              float sigma);

inline torch::Tensor
GrayWorld(const torch::Tensor& x)
{
  torch::Tensor mean = x.view({ -1, 3 }).mean(0);
  return (mean[0] - mean[1]) * (mean[0] - mean[1]) +
         (mean[0] - mean[2]) * (mean[0] - mean[2]) +
         (mean[1] - mean[2]) * (mean[1] - mean[2]);
}
