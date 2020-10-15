#pragma once

#include <torch/torch.h>

class DownSamplerImpl : public torch::nn::Module
{
public:
  DownSamplerImpl();

  torch::Tensor forward(const torch::Tensor x);

private:
  torch::nn::Conv2d mConv2d = nullptr;
  torch::nn::ReplicationPad2d mPad = nullptr;
};

TORCH_MODULE(DownSampler);

torch::Tensor
GaussianKernel2D(std::int64_t size,
                 float std = 2.0f,
                 std::int64_t channels = 3);
