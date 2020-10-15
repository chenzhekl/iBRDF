/**
 * This binary estimates material given geometry and illumination.
 *
 * Usage: ./build/bin/estbrdf [Input] [Geometry] [Illumination] [iBRDF model]
 * [Number of lobes] [Number of optimization steps] [Output]
 *
 * Example: ./build/bin/estbrdf ./run/render.exr ./data/sphere.pt
 * ./data/uffizi-large.exr ./data/ibrdf.pt 2 200 ./run/brdf.binary
 */
#include <torch/script.h>

#include "../backend.h"
#include "../ibrdf/ibrdf.h"
#include "../util.h"

int
main(int argc, char** argv)
{
  torch::manual_seed(0);
  torch::Device device(torch::kCUDA);

  const std::string kIn = std::string(argv[1]);
  const std::string kNormal = std::string(argv[2]);
  const std::string kEnvMapFile = std::string(argv[3]);
  const std::string kIBRDF = std::string(argv[4]);
  const std::int64_t kNumLobes = std::atoi(argv[5]);
  const std::int64_t kNumOptimEpoch = std::atoi(argv[6]);
  const std::string kOut = std::string(argv[7]);
  const std::string kMaterialFile = "./data/alum-bronze.binary";

  std::clog << "--------------" << std::endl
            << "Input: " << kIn << std::endl
            << "Geometry: " << kNormal << std::endl
            << "Illumination: " << kEnvMapFile << std::endl
            << "iBRDF model: " << kIBRDF << std::endl
            << "Number of lobes: " << kNumLobes << std::endl
            << "Output: " << kOut << std::endl
            << "--------------" << std::endl;

  OptiXBackendCreateOptions createOptions = {};
  createOptions.MissPTXFile =
    "./build/src/CMakeFiles/miss.dir/kernels/miss.ptx";
  createOptions.HitGroupPTXFile =
    "./build/src/CMakeFiles/hitgroup.dir/kernels/hitgroup.ptx";
  createOptions.RayGenPTXFile =
    "./build/src/CMakeFiles/raygen.dir/kernels/raygen.ptx";
  createOptions.NormalFile = kNormal;
  createOptions.Debug = false;

  OptiXBackend optix(createOptions);

  torch::Tensor mask = (LoadNormal(kNormal).sum(-1, true) != 0.0).to(device);

  // Camera
  {
    float3 lookFrom = make_float3(0.0f, 0.0f, 1.0f);
    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);

    optix.SetCamera(lookFrom, lookAt);
  }

  // Environment map
  {
    torch::Tensor envMap =
      torch::adaptive_avg_pool2d(
        LoadEXR(kEnvMapFile).permute({ 2, 0, 1 }).unsqueeze(0), { 256, 512 })
        .squeeze(0)
        .permute({ 1, 2, 0 });
    optix.SetEnvMap(envMap);
    // optix.SetPointLight(true, -60, make_float3(5.0f, 5.0f, 5.0f));
  }

  // Material
  {
    optix.AllocateMaterial();
    optix.AllocateMaterialD();
  }

  // Framebuffer
  {
    torch::Tensor gtImage = LoadEXR(kIn);
    optix.AllocateFrameBuffer(gtImage.size(1), gtImage.size(0));
    optix.SetGTColorBuffer(gtImage);
  }

  // BRDF models
  constexpr std::int64_t kNumLayers = 6;
  constexpr std::int64_t kNumInputFeatures = 3;
  constexpr std::int64_t kNumEmbedDim = 16;
  constexpr std::int64_t kNumPiecesPerLayer = 8;

  ibrdf::IBRDF model(
    kNumLayers, kNumInputFeatures, kNumEmbedDim, kNumPiecesPerLayer);
  {
    torch::load(model, kIBRDF);
    model->to(device);
    model->eval();
  }

  torch::Tensor refMaterial = LoadMERL(kMaterialFile);

  torch::Tensor color = torch::empty({ 3, kNumLobes }, torch::kFloat32)
                          .uniform_(1.4, 1.5)
                          .to(device)
                          .set_requires_grad(true);
  torch::Tensor embedCode =
    torch::zeros({ kNumLobes, kNumEmbedDim }, torch::kFloat32)
      .to(device)
      .set_requires_grad(true);

  // Optimizers
  torch::optim::Adam embedCodeOptimizer(std::vector<torch::Tensor>{ embedCode },
                                        torch::optim::AdamOptions(0.01));
  torch::optim::Adam colorOptimizer(std::vector<torch::Tensor>{ color },
                                    torch::optim::AdamOptions(0.01));

  float lastBestErr = 10.0e7f;
  std::int64_t numStalled = 0;
  torch::Tensor bestMat;

  for (std::int64_t i = 0; i < kNumOptimEpoch; ++i) {
    {
      torch::NoGradGuard noGradGuard;

      torch::Tensor material =
        GenerateMERL(model, refMaterial, embedCode, color.exp());

      optix.SetMaterial(material);
      optix.Render();
    }

    torch::Tensor loss, lossD;
    {
      torch::Tensor colorBuffer =
        optix.GetColorBuffer().set_requires_grad(true);
      torch::Tensor source = (colorBuffer + 1e-7f).log();
      torch::Tensor target = (optix.GetGTColorBuffer() + 1e-7f).log();

      loss = 0.01f * torch::l1_loss(
                       source * mask, target * mask, torch::Reduction::Sum);
      lossD = torch::autograd::grad({ loss }, { colorBuffer })[0];

      colorBuffer.set_requires_grad(false);
    }

    torch::Tensor nonZeroGradsIdx;
    {
      torch::NoGradGuard noGradGuard;

      optix.SetColorBufferD(lossD);
      optix.SetMaterialD(0.0f);
      optix.DiffRenderBRDF();

      nonZeroGradsIdx = optix.GetMaterialD().nonzero();
      nonZeroGradsIdx =
        std::get<0>(torch::unique_dim(nonZeroGradsIdx.slice(1, 1), 0));
    }

    constexpr std::int64_t kChunkSize = 150000;
    std::vector<torch::Tensor> nonZeroGradsIdxChunks =
      nonZeroGradsIdx.split(kChunkSize);

    torch::Tensor materialD = optix.GetMaterialD();

    ZeroGrad(color);
    ZeroGrad(embedCode);

    for (std::int64_t chunkIdx = 0;
         chunkIdx < static_cast<std::int64_t>(nonZeroGradsIdxChunks.size());
         ++chunkIdx) {
      torch::Tensor materialSubset =
        torch::zeros({ 3, nonZeroGradsIdxChunks[chunkIdx].size(0) },
                     torch::kFloat32)
          .to(device)
          .set_requires_grad(true);

      for (std::int64_t lobeIdx = 0; lobeIdx < kNumLobes; ++lobeIdx) {
        torch::Tensor lobeSubset =
          (GenerateMERLSamples(
             model,
             nonZeroGradsIdxChunks[chunkIdx],
             refMaterial,
             std::make_optional(embedCode[lobeIdx].repeat(
               { nonZeroGradsIdxChunks[chunkIdx].size(0), 1 })),
             true)
             .view({ 1, -1 }) *
           color.exp().select(1, lobeIdx).unsqueeze(-1))
            .expm1();

        materialSubset = materialSubset + lobeSubset;
      }

      std::vector<torch::Tensor> materialDSubset_;
      torch::Tensor materialDSubset;
      {
        torch::NoGradGuard noGradGuard;

        materialDSubset_.emplace_back(
          materialD[0]
            .index(nonZeroGradsIdxChunks[chunkIdx].split(1, 1))
            .squeeze(-1));
        materialDSubset_.emplace_back(
          materialD[1]
            .index(nonZeroGradsIdxChunks[chunkIdx].split(1, 1))
            .squeeze(-1));
        materialDSubset_.emplace_back(
          materialD[2]
            .index(nonZeroGradsIdxChunks[chunkIdx].split(1, 1))
            .squeeze(-1));
        materialDSubset = torch::stack(materialDSubset_, 0);
      }

      materialSubset.backward(materialDSubset);
    }

    std::clog << loss.sum().item<float>() << ", " << lastBestErr << ", "
              << numStalled << std::endl;

    if (loss.sum().item<float>() < lastBestErr) {
      if (lastBestErr - loss.sum().item<float>() > 1e-3f) {
        numStalled = 0;
      }

      lastBestErr = loss.sum().item<float>();
      bestMat = optix.GetMaterial().detach();
      ++numStalled;
    } else {
      ++numStalled;
    }

    if (numStalled > 20) {
      break;
    }

    torch::Tensor normPrior = 0.00001f * embedCode.pow(2).sum();
    normPrior.backward();

    embedCodeOptimizer.step();
    colorOptimizer.step();
  }

  SaveMERL(kOut, bestMat.cpu());

  return 0;
}
