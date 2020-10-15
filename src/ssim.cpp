#include "ssim.h"

namespace {
torch::Tensor
Gaussian(std::size_t windowSize, float sigma)
{
  std::vector<float> gauss_;

  for (std::size_t x = 0; x < windowSize; ++x) {
    gauss_.push_back(std::exp(-(static_cast<float>(x) - windowSize / 2) *
                              (static_cast<float>(x) - windowSize / 2) /
                              (2.0f * sigma * sigma)));
  }

  torch::Tensor gauss = torch::tensor(gauss_, torch::kFloat32);
  return gauss / gauss.sum();
}

torch::Tensor
CreateWindow(std::size_t windowSize, std::size_t channel)
{
  torch::Tensor window1D = Gaussian(windowSize, 1.5f).unsqueeze(1);
  torch::Tensor window2D = window1D.mm(window1D.t()).unsqueeze(0).unsqueeze(0);
  torch::Tensor window = window2D
                           .expand({ static_cast<std::int64_t>(channel),
                                     1,
                                     static_cast<std::int64_t>(windowSize),
                                     static_cast<std::int64_t>(windowSize) })
                           .contiguous();

  return window;
}
}

torch::Tensor
SSIM(const torch::Tensor& x,
     const torch::Tensor& y,
     std::size_t windowSize,
     bool sizeAverage)
{
  torch::Tensor img1 = x.permute({ 2, 0, 1 }).unsqueeze(0);
  torch::Tensor img2 = y.permute({ 2, 0, 1 }).unsqueeze(0);
  std::int64_t channel = img1.size(1);
  torch::Tensor window = CreateWindow(windowSize, channel).to(img1.device());

  torch::Tensor mu1 =
    torch::conv2d(img1, window, {}, 1, windowSize / 2, 1, channel);
  torch::Tensor mu2 =
    torch::conv2d(img2, window, {}, 1, windowSize / 2, 1, channel);

  torch::Tensor mu1Sq = mu1 * mu1;
  torch::Tensor mu2Sq = mu2 * mu2;
  torch::Tensor mu1Mu2 = mu1 * mu2;

  torch::Tensor sigma1Sq =
    torch::conv2d(img1 * img1, window, {}, 1, windowSize / 2, 1, channel) -
    mu1Sq;
  torch::Tensor sigma2Sq =
    torch::conv2d(img2 * img2, window, {}, 1, windowSize / 2, 1, channel) -
    mu2Sq;
  torch::Tensor sigma12 =
    torch::conv2d(img1 * img2, window, {}, 1, windowSize / 2, 1, channel) -
    mu1Mu2;

  float c1 = 0.01f * 0.01f;
  float c2 = 0.03f * 0.03f;

  torch::Tensor ssimMap = ((2.0f * mu1Mu2 + c1) * (2.0f * sigma12 + c2)) /
                          ((mu1Sq + mu2Sq + c1) * (sigma1Sq + sigma2Sq + c2));

  if (sizeAverage) {
    return ssimMap.mean();
  } else {
    return ssimMap.mean(1).mean(1).mean(1);
  }
}
