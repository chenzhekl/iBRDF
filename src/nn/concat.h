//
// Created by Zhe Chen on 2019/10/30.
//

#pragma once

#include <torch/torch.h>

#include "stack_sequential.h"

class ConcatImpl : public torch::nn::Module
{
public:
  ConcatImpl(std::size_t dim,
             const StackSequentail& m1,
             const StackSequentail& m2)
    : mDim(dim)
    , mModule1(m1)
    , mModule2(m2)
  {
    register_module("Module1", mModule1);
    register_module("Module2", mModule2);
  }

  torch::Tensor forward(const torch::Tensor& tensor);

private:
  std::size_t mDim = 0;
  StackSequentail mModule1 = nullptr;
  StackSequentail mModule2 = nullptr;
};

TORCH_MODULE(Concat);
