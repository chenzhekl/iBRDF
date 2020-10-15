#pragma once

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace ibrdf {
class PiecewiseConst1D
{
public:
  PiecewiseConst1D() = default;
  explicit PiecewiseConst1D(const torch::Tensor& f);

  [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  sampleContinuous(const torch::Tensor& p) const;
  [[nodiscard]] std::tuple<float, float, std::int64_t> sampleContinuousP(
    float p) const;

  [[nodiscard]] torch::Tensor pdf(const torch::Tensor& p) const;
  [[nodiscard]] float pdfP(float p) const;

  [[nodiscard]] float func(std::size_t index) const {
    return mFuncAccessor[index];
  }[[nodiscard]] float funcInt() const
  {
    return mFuncInt;
  }
  [[nodiscard]] int nx() const { return mNx; }

  private : torch::Tensor mCdf;
  torch::Tensor mFunc = torch::empty({ 1 });
  torch::TensorAccessor<float, 1> mFuncAccessor = mFunc.accessor<float, 1>();
  float mFuncInt = 0.0f;
  std::int64_t mNx = 0;
};

class PiecewiseConst2D
{
public:
  PiecewiseConst2D() = default;
  explicit PiecewiseConst2D(const torch::Tensor& f);

  [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  sampleContinuous(const torch::Tensor& p) const;
  [[nodiscard]] std::tuple<torch::Tensor, float, torch::Tensor>
  sampleContinuousP(const torch::Tensor& p) const;

  [[nodiscard]] torch::Tensor pdf(const torch::Tensor& p) const;
  [[nodiscard]] float pdfP(const torch::Tensor& p) const;

  [[nodiscard]] const std::vector<PiecewiseConst1D>& pyOnX() const {
    return mPyOnx;
  }[[nodiscard]] const PiecewiseConst1D& px() const
  {
    return mPx;
  }
  [[nodiscard]] int nx() const { return mNx; }[[nodiscard]] int ny() const
  {
    return mNy;
  }

private:
  std::vector<PiecewiseConst1D> mPyOnx;
  PiecewiseConst1D mPx;
  std::int64_t mNx = 0;
  std::int64_t mNy = 0;
};

class PiecewiseConst3D
{
public:
  explicit PiecewiseConst3D(const torch::Tensor& f);

  [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  sampleContinuous(const torch::Tensor& p) const;

  [[nodiscard]] torch::Tensor pdf(const torch::Tensor& p) const;

  [[nodiscard]] const std::vector<std::vector<PiecewiseConst1D>>& pzOnxy()
    const { return mPzOnxy; }[[nodiscard]] const PiecewiseConst2D& pxy() const
  {
    return mPxy;
  }
  [[nodiscard]] int nx() const { return mNx; }[[nodiscard]] int ny() const
  {
    return mNy;
  }
  [[nodiscard]] int nz() const { return mNz; }

  private : std::vector<std::vector<PiecewiseConst1D>> mPzOnxy;
  PiecewiseConst2D mPxy;

  int mNx;
  int mNy;
  int mNz;
};

template<typename Predicate>
int
findInterval(int size, const Predicate& pred)
{
  int first = 0, len = size;

  while (len > 0) {
    int half = len >> 1, middle = first + half;
    if (pred(middle)) {
      first = middle + 1;
      len -= half + 1;
    } else {
      len = half;
    }
  }

  int ret = std::clamp(first - 1, 0, size - 2);

  return ret;
}
} // namespace ibrdf
