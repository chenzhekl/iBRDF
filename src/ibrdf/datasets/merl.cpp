//
// Created by Zhe Chen on 2019/08/28.
//

#include "merl.h"

#include <cstdio>
#include <stdexcept>

#include "../distributions/piecewise.h"

namespace ibrdf {
namespace fs = std::experimental::filesystem;

static constexpr long BRDF_SAMPLING_RES_THETA_H = 90;
static constexpr long BRDF_SAMPLING_RES_THETA_D = 90;
static constexpr long BRDF_SAMPLING_RES_PHI_D = 180;

static constexpr float RED_SCALE = 1.0f / 1500.0f;
static constexpr float GREEN_SCALE = 1.15f / 1500.0f;
static constexpr float BLUE_SCALE = 1.66f / 1500.0f;

Merl::Merl(const fs::path& root, const std::vector<std::string>* include_list)
{
  std::vector<fs::path> paths;
  if (include_list) {
    for (auto const& name : *include_list) {
      fs::path p(root / (name + ".binary"));
      paths.emplace_back(p);
    }
  } else {
    for (const auto& f : fs::directory_iterator(root)) {
      if (f.path().extension().string() != ".binary") {
        continue;
      }
      paths.emplace_back(f.path());
    }
  }

#pragma omp parallel for default(none) shared(paths, std::clog)
  for (std::size_t i = 0; i < paths.size(); ++i) {
    // std::clog << paths[i] << std::endl;
    torch::Tensor brdf = load(paths[i]).log1p();

#pragma omp critical
    {
      mBRDFs.emplace_back(paths[i].stem().string() + "-red",
                          PiecewiseConst3D(brdf[0]));
      mBRDFs.emplace_back(paths[i].stem().string() + "-green",
                          PiecewiseConst3D(brdf[1]));
      mBRDFs.emplace_back(paths[i].stem().string() + "-blue",
                          PiecewiseConst3D(brdf[2]));

      std::clog << "Processed: " << paths[i].stem().string() << std::endl;
    }
  }

  std::sort(
    mBRDFs.begin(), mBRDFs.end(), [&](const BRDF& b1, const BRDF& b2) -> bool {
      return b1.label < b2.label;
    });
}

torch::Tensor
Merl::pdf(const torch::Tensor& x, std::int64_t embedID) const
{
  torch::Tensor xCPU = x.to(torch::kCPU);

  return mBRDFs[embedID].distribution.pdf(x);
}

torch::Tensor
Merl::logPdf(const torch::Tensor& x, std::int64_t embedID) const
{
  return pdf(x, embedID).log();
}

torch::Tensor
Merl::sample(std::int64_t numSamples, std::int64_t embedID) const
{
  torch::Tensor rand =
    torch::rand({ static_cast<std::int64_t>(numSamples), 3 });
  auto [samples, pdfs, offsets] =
    mBRDFs[embedID].distribution.sampleContinuous(rand);

  (void)pdfs;
  (void)offsets;

  return samples;
}

// TODO: need to verify its correctness
torch::Tensor
Merl::load(const fs::path& path)
{
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    throw std::logic_error("Couldn't open material file");
  }

  int dim[3];
  if (fread(dim, sizeof(int), 3, f) != 3) {
    throw std::logic_error("Invalid number of elements read");
  }

  std::size_t n = dim[0] * dim[1] * dim[2];
  if (n != BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D *
             BRDF_SAMPLING_RES_PHI_D) {
    throw std::logic_error("Dimensions don't match");
  }

  std::vector<double> brdf(3 * n);
  if (fread(brdf.data(), sizeof(double), 3 * n, f) != 3 * n) {
    throw std::logic_error("Invalid number of elements read");
  }

  std::fclose(f);

  torch::Tensor merl = torch::tensor(brdf)
                         .to(torch::kFloat)
                         .view({ 3,
                                 BRDF_SAMPLING_RES_THETA_H,
                                 BRDF_SAMPLING_RES_THETA_D,
                                 BRDF_SAMPLING_RES_PHI_D });

  return merl;
}

void
Merl::save(const torch::Tensor& x,
           const std::experimental::filesystem::path& path,
           bool normalize)
{
  torch::Tensor data = x.view({ 3, -1 });

  if (normalize) {
    torch::Tensor scale =
      torch::tensor({ 1.0f, RED_SCALE / GREEN_SCALE, RED_SCALE / BLUE_SCALE },
                    data.options())
        .unsqueeze(-1);
    data *= scale;
  }

  torch::Tensor dataCPU = data.flatten().to(torch::kDouble).to(torch::kCPU);
  torch::TensorAccessor dataCPUAccessor = dataCPU.accessor<double, 1>();

  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) {
    throw std::logic_error("Couldn't open material file");
  }

  fwrite(&BRDF_SAMPLING_RES_THETA_H, sizeof(int), 1, file);
  fwrite(&BRDF_SAMPLING_RES_THETA_D, sizeof(int), 1, file);
  fwrite(&BRDF_SAMPLING_RES_PHI_D, sizeof(int), 1, file);

  for (long i = 0; i < dataCPUAccessor.size(0); ++i) {
    fwrite(&dataCPUAccessor[i], sizeof(double), 1, file);
  }
  std::fclose(file);
}
} // namespace ibrdf
