#pragma once

#include <torch/torch.h>

torch::Tensor
SSIM(const torch::Tensor& x,
     const torch::Tensor& y,
     std::size_t windowSize = 11,
     bool sizeAverage = true);
