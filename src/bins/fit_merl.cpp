#include "../kernels/merl_constant.h"
#include "../util.h"

int
main(int argc, char** argv)
{
  torch::manual_seed(0);

  constexpr std::size_t kNumLobes = 3;
  constexpr std::size_t kNumOptimEpoch = 500;
  constexpr std::size_t kSplitSize = 50000;
  constexpr std::size_t kSaveInterval = 10;
  constexpr std::size_t kPatience = 50;
  constexpr std::size_t kEpsilon = 5e-5f;
  // const std::string kMaterialFile =
  //   "../datasets/rgl_halfangle/" + std::string(argv[1]) + ".binary";
  const std::string kMaterialFile =
    "../datasets/merl/" + std::string(argv[1]) + ".binary";
  const std::string kRefMaterialFile = "../datasets/merl/alum-bronze.binary";

  torch::Device device(torch::kCUDA);

  torch::Tensor gtMaterial = LoadMERL(kMaterialFile).to(device);
  gtMaterial.masked_fill_(gtMaterial < 0.0f, 0.0f);

  torch::Tensor refMaterial = LoadMERL(kRefMaterialFile).to(device);
  refMaterial.masked_fill_(refMaterial < 0.0f, 0.0f);

  constexpr std::size_t kNumLayers = 6;
  constexpr std::size_t kNumInputFeatures = 3;
  constexpr std::size_t kNumEmbedDim = 16;
  constexpr std::size_t kNumPiecesPerLayer = 8;

  ibrdf::IBRDF model(
    kNumLayers, kNumInputFeatures, kNumEmbedDim, kNumPiecesPerLayer);
  model->to(device);
  torch::load(model, "./run/fit_conditional/model_exclude_purple-paint.pt");
  // torch::load(model, "./run/checkpoints/model_final.pt");
  // torch::load(model,
  //             "../results/fit_conditional/model_exclude_" +
  //               std::string(argv[1]) + ".pt");
  // torch::load(model,
  //             "../results/fit_conditional/model_exclude_tungsten-carbide.pt");
  model->eval();

  torch::Tensor color = torch::ones({ 3, kNumLobes }, torch::kFloat32)
                          .to(device)
                          .uniform_(0.0f, 1.0f)
                          .set_requires_grad(true);
  torch::Tensor embedCode =
    torch::zeros({ kNumLobes, kNumEmbedDim }, torch::kFloat32)
      .to(device)
      .set_requires_grad(true);

  torch::optim::Adam embedCodeOptimizer(std::vector<torch::Tensor>{ embedCode },
                                        torch::optim::AdamOptions(0.01f));
  torch::optim::Adam colorOptimizer(std::vector<torch::Tensor>{ color },
                                    torch::optim::AdamOptions(0.1f));

  torch::Tensor grid;
  {
    torch::NoGradGuard noGradGuard;

    // torch::Tensor thetaH =
    //   torch::arange(static_cast<std::int64_t>(kBRDFSamplingResThetaH));
    // torch::Tensor thetaD =
    //   torch::arange(static_cast<std::int64_t>(kBRDFSamplingResThetaD));
    // torch::Tensor phiD =
    //   torch::arange(static_cast<std::int64_t>(kBRDFSamplingResPhiD / 2));

    // std::vector<torch::Tensor> axisTuple =

    //   torch::meshgrid({ thetaH, thetaD, phiD });
    // grid = torch::stack({ axisTuple[0].flatten(),
    //                       axisTuple[1].flatten(),
    //                       axisTuple[2].flatten() })
    //          .t()
    //          .view({ -1, 3 })
    //          .to(device);
    grid = (gtMaterial[0] != 0.0f).nonzero();
  }
  std::vector<torch::Tensor> gridChunks = grid.split(kSplitSize);
  // std::vector<torch::Tensor> gtChunks =
  //   gtMaterial.view({ 3, -1 }).split(kSplitSize, 1);
  std::vector<torch::Tensor> gtChunks =
    gtMaterial.masked_select(gtMaterial[0] != 0.0f)
      .view({ 3, -1 })
      .split(kSplitSize, 1);

  const std::size_t validEntries =
    (gtMaterial[0] != 0.0f).sum().item<std::int64_t>() * 3;

  // std::clog << "#Valid entries: " << validEntries << std::endl;

  float lastBestErr = 10.0e7f;
  torch::Tensor bestEmbedCode, bestColor;
  std::size_t numStalls = 0;

  for (std::size_t epoch = 0; epoch < kNumOptimEpoch; ++epoch) {
    ZeroGrad(embedCode);
    ZeroGrad(color);

    float lossAccu = 0.0f;

    for (std::size_t chunkIdx = 0; chunkIdx < gridChunks.size(); ++chunkIdx) {
      torch::Tensor material =
        torch::zeros({ 3, gridChunks[chunkIdx].size(0) }, torch::kFloat32)
          .to(device)
          .set_requires_grad(true);

      // std::vector<torch::Tensor> channels(3);

      for (std::size_t lobeIdx = 0; lobeIdx < kNumLobes; ++lobeIdx) {
        torch::Tensor lobe =
          (GenerateMERLSamples(
             model,
             gridChunks[chunkIdx],
             refMaterial,
             embedCode[lobeIdx].repeat({ gridChunks[chunkIdx].size(0), 1 }))
             .view({ 1, -1 }) *
           color.exp().select(1, lobeIdx).unsqueeze(-1))
            .expm1();

        material = material + lobe;
        // channels[chanIdx] = channel;
      } // Mixing lobes

      // torch::Tensor material = (torch::stack(channels) * color).expm1();

      torch::Tensor loss = torch::mse_loss((material + 1e-7f).log(),
                                           (gtChunks[chunkIdx] + 1e-7f).log(),
                                           torch::Reduction::Sum);

      {
        torch::NoGradGuard noGradGuard;
        torch::Tensor lossValue =
          torch::mse_loss((material + 1e-6f).log(),
                          (gtChunks[chunkIdx] + 1e-6f).log(),
                          torch::Reduction::Sum);

        lossAccu += lossValue.item<float>();
      }

      (loss).backward();
    } // Finish a chunk

    float valErr = std::sqrt(lossAccu / validEntries);
    std::clog << argv[1] << " | " << epoch << ": " << valErr << ", "
              << numStalls << std::endl;

    if (valErr < lastBestErr) {
      if (lastBestErr - valErr > kEpsilon) {
        numStalls = 0;
      } else {
        ++numStalls;
      }

      lastBestErr = valErr;
      bestEmbedCode = embedCode.detach();
      bestColor = color.detach();
    } else {
      ++numStalls;
    }

    if (numStalls > kPatience) {
      break;
    }

    colorOptimizer.step();
    embedCodeOptimizer.step();

    {
      torch::NoGradGuard noGradGuard;

      constexpr float kRadius = 1.01f;

      torch::Tensor embedCodeNorm = embedCode.norm(2, -1, true);
      embedCode.masked_scatter_(embedCodeNorm > kRadius,
                                embedCode / embedCodeNorm * kRadius);
    }

    // if (epoch % kSaveInterval == 0) {
    //   torch::NoGradGuard noGradGuard;

    //   torch::Tensor material =
    //     torch::zeros({ 3, 90 * 90 * 180 }, torch::kFloat32).to(device);

    //   // std::vector<torch::Tensor> channels(3);

    //   for (std::size_t lobeIdx = 0; lobeIdx < kNumLobes; ++lobeIdx) {
    //     torch::Tensor lobe =
    //       (GenerateMERLSlice(model,
    //                          gtMaterial,
    //                          embedCode[lobeIdx].repeat({ 90 * 90 * 180, 1 }),
    //                          true)
    //          .view({ 1, -1 }) *
    //        color.select(1, lobeIdx).unsqueeze(-1))
    //         .expm1();

    //     material += lobe;
    //     // channels[chanIdx] = channel;
    //   }

    //   // torch::Tensor material = (torch::stack(channels) * color).expm1();

    //   SaveMERL("./run/fit.binary", material);
    // }
  } // Finish an epoch

  {
    torch::NoGradGuard noGradGuard;
    torch::Tensor material =
      GenerateMERL(model, refMaterial, bestEmbedCode, bestColor.exp());
    // SaveMERL("./run/fit_rgl_rmse/" + std::string(argv[1]) + ".binary",
    //          material);
    // SaveMERL("./run/brdf.binary", material);
  }

  std::cout << lastBestErr;

  return 0;
}
