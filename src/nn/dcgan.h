//
// Created by Zhe Chen on 2019/09/10.
//

#pragma once

#include <cmath>
#include <cstddef>

#include <torch/torch.h>

class DCGANImpl : public torch::nn::Module
{
public:
  explicit DCGANImpl(std::size_t numNoiseChannels,
                     std::size_t numLayers = 7,
                     std::size_t ngf = 64)
    : mLayers(torch::nn::Sequential())
  {
    std::size_t factor = std::pow(2, numLayers - 2);

    mLayers->push_back(
      torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
                                   numNoiseChannels, ngf * factor, { 4, 8 })
                                   .bias(false)));
    mLayers->push_back(torch::nn::BatchNorm2d(ngf * factor));
    mLayers->push_back(torch::nn::Functional(torch::leaky_relu, 0.2));

    for (std::size_t i = 1; i < numLayers - 1; ++i) {
      factor /= 2;
      mLayers->push_back(torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(ngf * factor * 2, ngf * factor, 4)
          .stride(2)
          .padding(1)
          .bias(false)));
      mLayers->push_back(torch::nn::BatchNorm2d(ngf * factor));
      mLayers->push_back(torch::nn::Functional(torch::leaky_relu, 0.2));
    }

    mLayers->push_back(torch::nn::ConvTranspose2d(
      torch::nn::ConvTranspose2dOptions(ngf * factor, 3, 4)
        .stride(2)
        .padding(1)
        .bias(false)));
    // mLayers->push_back(torch::nn::Functional(torch::leaky_relu, 0.2));

    register_module("layers", mLayers);
  }

  torch::Tensor forward(const torch::Tensor& x)
  {
    torch::Tensor y = mLayers->forward(x).exp();
    return y;
  }

private:
  torch::nn::Sequential mLayers = nullptr;
};

TORCH_MODULE(DCGAN);
