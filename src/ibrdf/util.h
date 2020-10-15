//
// Created by Zhe Chen on 2019/08/27.
//

#pragma once

#include <cmath>
#include <cstddef>

#include <torch/torch.h>

namespace ibrdf {
inline torch::Tensor
oneBlob(const torch::Tensor& x, std::size_t numBins)
{
  torch::Tensor grid = torch::linspace(0.0, 1.0, numBins, x.options());

  float sigma = 1.0f / numBins;
  float normalizer = 1.0f / std::sqrt(2.0f * M_PIf32 * sigma * sigma);

  torch::Tensor mean =
    x.unsqueeze(-1).repeat({ 1, 1, static_cast<long>(numBins) });

  return normalizer * torch::exp(-(grid - mean).pow(2) / (2.0f * sigma * sigma))
                        .view({ x.size(0), -1 });
}
} // namespace ibrdf
