#include "downsampler.h"
#include "../math.h"

DownSamplerImpl::DownSamplerImpl()
{
  float support = 2.0f;
  float phase = 0.5f;
  std::int64_t factor = 2;
  std::int64_t kernelWidth = 4 * factor + 1;
  float center = (static_cast<float>(kernelWidth) + 1.0f) / 2.0f;

  torch::Tensor kernel = torch::zeros({ kernelWidth - 1, kernelWidth - 1 });
  torch::TensorAccessor kernelAccessor = kernel.accessor<float, 2>();

  for (std::int64_t i = 1; i < kernelAccessor.size(0) + 1; ++i) {
    for (std::int64_t j = 1; j < kernelAccessor.size(1) + 1; ++j) {
      float di = std::abs(i + 0.5f - center) / factor;
      float dj = std::abs(j + 0.5f - center) / factor;

      float val = 1;
      if (di != 0.0f) {
        val *= support * std::sin(kPI * di) * std::sin(kPI * di / support);
        val /= kPI * kPI * di * di;
      }
      if (dj != 0.0f) {
        val *= support * std::sin(kPI * dj) * std::sin(kPI * dj / support);
        val /= kPI * kPI * dj * dj;
      }

      kernelAccessor[i - 1][j - 1] = val;
    }
  }

  kernel /= kernel.sum();

  mConv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, kernel.sizes())
                                .stride(factor)
                                .padding(0)
                                .bias(false));
  mConv2d->weight.data().fill_(0.0f);
  mConv2d->weight.data()[0][0] = kernel;
  mConv2d->weight.data()[1][1] = kernel;
  mConv2d->weight.data()[2][2] = kernel;

  register_module("Downsampler", mConv2d);

  std::int64_t pad;
  if (kernel.size(0) % 2 == 1) {
    pad = (kernel.size(0) - 1) / 2;
  } else {
    pad = (kernel.size(0) - factor) / 2;
  }

  mPad = torch::nn::ReplicationPad2d(pad);

  register_module("Padder", mPad);
}

torch::Tensor
DownSamplerImpl::forward(const torch::Tensor x)
{
  torch::Tensor y = mPad->forward(x);
  y = mConv2d->forward(y);

  return y;
}

torch::Tensor
GaussianKernel2D(std::int64_t size, float std, std::int64_t channels)
{
  torch::NoGradGuard noGradGuard;

  std::vector<torch::Tensor> meshgrids =
    torch::meshgrid({ torch::arange(size, torch::kFloat32),
                      torch::arange(size, torch::kFloat32) });

  torch::Tensor kernel;

  kernel = 1.0f / (std * std::sqrt(2 * kPI)) *
           (((meshgrids[0] - size) / std).pow(2) * -0.5f).exp();
  kernel *= 1.0f / (std * std::sqrt(2 * kPI)) *
            (((meshgrids[1] - size) / std).pow(2) * -0.5f).exp();

  kernel /= kernel.sum();
  kernel = kernel.view({ 1, 1, size, size });
  kernel = kernel.repeat({ channels, 1, 1, 1 });

  return kernel;
}
