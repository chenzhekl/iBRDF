//
// Created by Zhe Chen on 2019/08/28.
//

#pragma once

#include <cstddef>
#include <experimental/filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../distributions/piecewise.h"

namespace ibrdf {
struct BRDF
{
  BRDF(const std::string& label, const PiecewiseConst3D& distribution)
    : label(label)
    , distribution(distribution)
  {}
  std::string label;
  PiecewiseConst3D distribution;
};

class Merl
{
public:
  explicit Merl(const std::experimental::filesystem::path& root,
                const std::vector<std::string>* include_list = nullptr);

  [[nodiscard]] torch::Tensor pdf(const torch::Tensor& x,
                                  std::int64_t embedID) const;
  [[nodiscard]] torch::Tensor logPdf(const torch::Tensor& x,
                                     std::int64_t embedID) const;
  [[nodiscard]] torch::Tensor sample(std::int64_t numSamples,
                                     std::int64_t embedID) const;

  [[nodiscard]] std::int64_t numBRDFs() const { return mBRDFs.size(); }
  [[nodiscard]] std::string label(std::int64_t embedID) const
  {
    return mBRDFs[embedID].label;
  }

  [[nodiscard]] static torch::Tensor load(
    const std::experimental::filesystem::path& path);

  static void save(const torch::Tensor& x,
                   const std::experimental::filesystem::path& path,
                   bool normalize = true);

private:
  std::vector<BRDF> mBRDFs;
};
} // namespace ibrdf
