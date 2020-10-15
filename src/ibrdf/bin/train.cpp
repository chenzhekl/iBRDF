/**
 * iBRDF training scripts.
 *
 * Usage: ./build/bin/ibrdf_train [MERL root] [Number of BRDFs per batch]
 * [Number of samples per BRDF] [Number of epochs] [Output]
 *
 * Example: ./build/bin/ibrdf_train ../datasets/merl 50 10000 10000 ./run/ibrdf
 */

#include <cstddef>
#include <iostream>
#include <string>

#include "../datasets/merl.h"
#include "../ibrdf.h"
#include "../merl_materials.h"

using namespace ibrdf;

void
trainStep(IBRDF& model,
          Merl& dataset,
          torch::optim::Optimizer& optimizer,
          std::int64_t brdfsPerBatch,
          std::int64_t samplesPerBRDF,
          std::int64_t epoch,
          torch::Tensor& z,
          torch::optim::Optimizer& zOptimizer)
{
  std::clog << "Epoch: " << epoch << " ";

  model->train();
  torch::Device device = model->parameters()[0].device();

  // Sample training set
  std::vector<torch::Tensor> samples_;
  torch::Tensor ids =
    torch::multinomial(torch::ones(dataset.numBRDFs()), brdfsPerBatch)
      .to(torch::kInt64);
  torch::TensorAccessor idsAccessor = ids.accessor<std::int64_t, 1>();

  for (std::int64_t i = 0; i < idsAccessor.size(0); ++i) {
    std::int64_t id = idsAccessor[i];
    samples_.emplace_back(dataset.sample(samplesPerBRDF, id));
  }

  torch::Tensor samples = torch::cat(samples_).to(device);

  // Calculate loss
  torch::Tensor logQx = model->logPdf(
    samples,
    z.gather(0, ids.unsqueeze(-1).repeat({ 1, z.size(1) }).to(device))
      .repeat_interleave(samplesPerBRDF, 0));
  std::vector<torch::Tensor> logQxChunks = logQx.split(samplesPerBRDF);
  std::vector<torch::Tensor> ll_;

  for (const auto& chunk : logQxChunks) {
    ll_.emplace_back(chunk.mean());
  }

  torch::Tensor ll = torch::stack(ll_);
  torch::Tensor loss = (-ll).mean();

  // Logging
  {
    torch::NoGradGuard noGradGuard;

    std::clog << "Loss: " << loss.item<float>() << ", "
              << "Max latent norm: " << z.norm(2, -1).max().item<float>()
              << std::endl;
  }

  model->zero_grad();
  if (z.grad().defined()) {
    z.grad().zero_();
  }

  loss.backward();
  optimizer.step();
  zOptimizer.step();

  // Re-project z
  {
    torch::NoGradGuard noGradGuard;

    constexpr float kRadius = 1.0f;

    torch::Tensor zNorm = z.norm(2, -1, true);
    torch::Tensor scale = kRadius / (zNorm + 1e-7f);
    z.masked_scatter_(zNorm > kRadius, z * scale);
  }
}

int
main(int argc, char** argv)
{
  torch::manual_seed(0);

  torch::Device device(torch::kCUDA);

  const std::string kMerlRoot = std::string(argv[1]);
  const std::int64_t kBRDFsPerBatch = std::atoi(argv[2]);
  const std::int64_t kSamplesPerBRDF = std::atoi(argv[3]);
  const std::int64_t kNumEpochs = std::atoi(argv[4]);
  const std::string kOut = std::string(argv[5]);
  constexpr std::size_t kSaveInterval = 1000;

  Merl dataset(kMerlRoot);

  std::clog << "Loaded: " << dataset.numBRDFs() << " BRDFs: " << std::endl;

  constexpr std::size_t kNumLayers = 6;
  constexpr std::size_t kNumInputFeatures = 3;
  constexpr std::size_t kNumEmbedDim = 16;
  constexpr std::size_t kNumPiecesPerLayer = 8;

  IBRDF model(kNumLayers, kNumInputFeatures, kNumEmbedDim, kNumPiecesPerLayer);
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.0001f));

  torch::Tensor z =
    (0.01f * torch::randn({ dataset.numBRDFs(), kNumEmbedDim }, device))
      .set_requires_grad(true);
  torch::optim::Adam zOptimizer(std::vector<torch::Tensor>{ z },
                                torch::optim::AdamOptions(0.01f));

  for (std::int64_t e = 0; e < kNumEpochs; ++e) {
    trainStep(model,
              dataset,
              optimizer,
              kBRDFsPerBatch,
              kSamplesPerBRDF,
              e,
              z,
              zOptimizer);

    if (e % kSaveInterval == 0) {
      torch::save(model, kOut + "_" + std::to_string(e) + ".pt");
      torch::save(z, kOut + "_z_" + std::to_string(e) + ".pt");
    }
  }

  torch::save(model, kOut + ".pt");
  torch::save(z, kOut + "_z.pt");

  return 0;
}
