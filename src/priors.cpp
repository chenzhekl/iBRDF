#include "priors.h"

#include <vector>

#include "math.h"

torch::Tensor
EntropyPrior(const torch::Tensor& img, std::int64_t bincount, float sigma)
{
  torch::NoGradGuard noGradGuard;
  torch::Device device = img.device();

  torch::Tensor x = img.flatten();

  float mult = 1.0f / (x.size(0) * std::sqrt(2.0f * kPI * sigma * sigma));
  float inv2sigmasq = 1.0f / (2.0f * sigma * sigma);
  float xmin = x.min().cpu().item<float>();
  float xmax = x.max().cpu().item<float>();
  float intwidth = xmax - xmin + 6.0f * sigma;

  torch::Tensor histogram = torch::empty({ bincount }, torch::kFloat32);
  torch::TensorAccessor histogramAccessor = histogram.accessor<float, 1>();
  for (std::int64_t i = 0; i < bincount; ++i) {
    float center =
      static_cast<float>(i) / static_cast<float>(bincount) * intwidth + xmin -
      3.0f * sigma;

    torch::Tensor diff = x - center;
    float accum = (-diff * diff * inv2sigmasq).exp().sum().cpu().item<float>();
    histogramAccessor[i] = accum * mult;
  }

  // torch::Tensor entropy = torch::where(
  //   histogram > 0.0f, histogram * histogram.log(), torch::tensor(0.0f));
  // entropy = -entropy.sum() * intwidth / static_cast<float>(bincount);

  return histogram.to(device);
}

torch::Tensor
EntropyPriorD(const torch::Tensor& img,
              const torch::Tensor& hist,
              std::int64_t bincount,
              float sigma)
{
  torch::NoGradGuard noGradGuard;
  torch::Device device = img.device();

  torch::Tensor x = img.flatten();

  float mult = -2.0f / (x.size(0) * std::sqrt(2.0f * kPI * sigma * sigma));
  float inv2sigmasq = 1.0f / (2.0f * sigma * sigma);
  float xmin = x.min().cpu().item<float>();
  float xmax = x.max().cpu().item<float>();
  float intwidth = xmax - xmin + 6.0f * sigma;

  torch::Tensor centers = torch::arange(bincount, torch::kFloat32).to(device) /
                            static_cast<float>(bincount) * intwidth +
                          xmin - 3.0f * sigma;
  std::vector<float> imgD(x.size(0));
  torch::Tensor xCPU = x.cpu();
  torch::TensorAccessor xCPUAccessor = xCPU.accessor<float, 1>();

  torch::Tensor histLog =
    torch::where(hist > 0.0f, hist.log(), torch::tensor(0.0f, hist.device())) +
    1.0f;

  for (std::int64_t i = 0; i < xCPUAccessor.size(0); ++i) {
    torch::Tensor diff = xCPUAccessor[i] - centers;
    float accum = ((-diff * diff * inv2sigmasq).exp() * diff * histLog)
                    .sum()
                    .cpu()
                    .item<float>();

    imgD[i] = mult * accum * inv2sigmasq * intwidth / bincount;
  }

  return torch::tensor(imgD, torch::kFloat32).to(device).view(img.sizes());
}
