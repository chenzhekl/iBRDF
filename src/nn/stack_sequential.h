//
// Created by Zhe Chen on 2019/10/30.
//

#pragma once

#include <torch/torch.h>

struct StackSequentailImpl : torch::nn::SequentialImpl
{
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(const torch::Tensor& tensor)
  {
    return SequentialImpl::forward(tensor);
  }
};

TORCH_MODULE(StackSequentail);
