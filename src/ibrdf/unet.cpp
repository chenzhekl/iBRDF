//
// Created by Zhe Chen on 2019/08/27.
//

#include "unet.h"

namespace ibrdf {
UNetImpl::UNetImpl(std::size_t inFeatures,
                   std::size_t outFeatures,
                   std::size_t numBins)
  : mNumBins(numBins)
{
  mPre = register_module("pre", torch::nn::Linear(inFeatures * numBins, 256));

  mE1 = register_module("e1", torch::nn::Linear(256, 128));
  mE2 = register_module("e2", torch::nn::Linear(128, 64));
  mE3 = register_module("e3", torch::nn::Linear(64, 32));
  mE4 = register_module("e4", torch::nn::Linear(32, 16));

  mD1 = register_module("d1", torch::nn::Linear(16, 32));
  mD2 = register_module("d2", torch::nn::Linear(64, 64));
  mD3 = register_module("d3", torch::nn::Linear(128, 128));
  mD4 = register_module("d4", torch::nn::Linear(256, 256));

  mPost = register_module("post", torch::nn::Linear(256, outFeatures));
}

torch::Tensor
UNetImpl::forward(torch::Tensor x)
{
  if (mNumBins > 1) {
    x = oneBlob(x, mNumBins);
  }

  x = mPre->forward(x);

  torch::Tensor h1 = mE1->forward(torch::relu(x));
  torch::Tensor h2 = mE2->forward(torch::relu(h1));
  torch::Tensor h3 = mE3->forward(torch::relu(h2));
  torch::Tensor h4 = mE4->forward(torch::relu(h3));

  torch::Tensor y = mD1->forward(torch::relu(h4));
  y = mD2->forward(torch::relu(torch::cat({ y, h3 }, 1)));
  y = mD3->forward(torch::relu(torch::cat({ y, h2 }, 1)));
  y = mD4->forward(torch::relu(torch::cat({ y, h1 }, 1)));

  y = mPost->forward(torch::relu(y));

  return y;
}
} // namespace ibrdf
