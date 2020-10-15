/**
 * This binary joinotly estimates material and illumination given geometry.
 *
 * Usage: ./build/bin/estboth [Input] [Geometry] [iBRDF model] [Number of lobes]
 * [Illumination width] [Illumination height] [Number of optimization steps]
 * [Number of material optimization steps] [Number of illumination optimization
 * steps] [Number of gray world steps] [Output material] [Output illumination]
 *
 * Example: ./build/bin/estboth ./run/render.exr ./data/sphere.pt
 * ./data/ibrdf.pt 2 256 128 10 100 300 3 ./run/brdf.binary ./run/illu.exr
 */
#include <torch/script.h>

#include "../backend.h"
#include "../ibrdf/ibrdf.h"
#include "../nn/skip.h"
#include "../priors.h"
#include "../util.h"

int
main(int argc, char** argv)
{
  torch::manual_seed(0);
  torch::Device device(torch::kCUDA);

  const std::string kIn = std::string(argv[1]);
  const std::string kNormal = std::string(argv[2]);
  const std::string kIBRDF = std::string(argv[3]);
  const std::int64_t kNumLobes = std::atoi(argv[4]);
  const std::int64_t kEnvMapWidth = std::atoi(argv[5]);
  const std::int64_t kEnvMapHeight = std::atoi(argv[6]);
  const std::int64_t kNumOptimEpoch = std::atoi(argv[7]);
  const std::int64_t kNumBRDFSteps = std::atoi(argv[8]);
  const std::int64_t kNumIlluSteps = std::atoi(argv[9]);
  const std::int64_t kNumGrayWorldSteps = std::atoi(argv[10]);
  const std::string kOutMat = std::string(argv[11]);
  const std::string kOutIllu = std::string(argv[12]);
  const std::string kMaterialFile = "./data/alum-bronze.binary";

  std::clog << "--------------" << std::endl
            << "Input: " << kIn << std::endl
            << "Geometry: " << kNormal << std::endl
            << "iBRDF model: " << kIBRDF << std::endl
            << "Number of lobes: " << kNumLobes << std::endl
            << "Illumination size: " << kEnvMapWidth << " x " << kEnvMapHeight
            << std::endl
            << "#Optimization steps: " << kNumOptimEpoch << std::endl
            << "#Illumination steps: " << kNumIlluSteps << std::endl
            << "#Material steps: " << kNumBRDFSteps << std::endl
            << "#Grayworld steps: " << kNumGrayWorldSteps << std::endl
            << "Material output: " << kOutMat << std::endl
            << "Illumination output" << kOutIllu << std::endl
            << "--------------" << std::endl;

  constexpr std::int64_t kBRDFPatience = 15;
  constexpr std::int64_t kIlluPatience = 50;

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
    optix.AllocateEnvMap(kEnvMapWidth, kEnvMapHeight);
    optix.AllocateEnvMapD(kEnvMapWidth, kEnvMapHeight);
  }

  // Material
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

    optix.AllocateMaterial();
    optix.AllocateMaterialD();
  }

  // Framebuffer
  {
    torch::Tensor gtImage = LoadEXR(kIn).to(device);
    torch::Tensor gtImageLog = (gtImage + 1e-6f).log();
    torch::Tensor indices = mask * (gtImage.sum(-1, true) > 0.0f);
    torch::Tensor gtImageAvg = gtImageLog.masked_select(indices).mean();
    gtImage *= 0.01f / gtImageAvg.exp();
    // gtImage /= gtImage.max();
    gtImage.masked_fill_(gtImage < 0.0f, 0.0f);
    std::clog << "Image max: " << gtImage.max().item<float>() << std::endl;

    optix.AllocateFrameBuffer(gtImage.size(1), gtImage.size(0));
    optix.SetGTColorBuffer(gtImage);
  }

  torch::Tensor refMaterial = LoadMERL(kMaterialFile);

  torch::Tensor color = torch::zeros({ 3, kNumLobes }, torch::kFloat32)
                          .uniform_(1.4, 1.5)
                          .to(device)
                          .set_requires_grad(true);
  torch::Tensor embedCode =
    torch::zeros({ kNumLobes, kNumEmbedDim }, torch::kFloat32)
      .to(device)
      .set_requires_grad(true);

  // Deep illumination prior
  Skip dip(8,
           3,
           std::vector<std::int64_t>{ 128, 128, 128, 128, 128 },
           std::vector<std::int64_t>{ 128, 128, 128, 128, 128 },
           std::vector<std::int64_t>{ 16, 16, 16, 16, 16 });
  dip->to(device);
  dip->train();

  torch::Tensor noise =
    torch::rand({ 1, 8, kEnvMapHeight, kEnvMapWidth }, device)
      .uniform_(0.0f, 0.1f);

  // Set initial material
  {
    torch::NoGradGuard noGradGuard;

    torch::Tensor material =
      GenerateMERL(model, refMaterial, embedCode, color.exp());
    optix.SetMaterial(material);
  }

  // Optimizers
  // torch::optim::Adam illuOptimizer(dip->parameters(),
  //                                  torch::optim::AdamOptions(0.01));
  torch::optim::Adam illuOptimizer(dip->parameters(),
                                   torch::optim::AdamOptions(0.001));
  // torch::optim::Adam embedCodeOptimizer(std::vector<torch::Tensor>{ embedCode
  // },
  //                                       torch::optim::AdamOptions(0.01));
  // torch::optim::Adam colorOptimizer(std::vector<torch::Tensor>{ color },
  //                                   torch::optim::AdamOptions(0.05));
  torch::optim::Adam embedCodeOptimizer(std::vector<torch::Tensor>{ embedCode },
                                        torch::optim::AdamOptions(0.01));
  torch::optim::Adam colorOptimizer(std::vector<torch::Tensor>{ color },
                                    torch::optim::AdamOptions(0.01));

  torch::Tensor bestMat, bestIllu;
  float lastIlluBestErr = 10.0e7f;
  float lastBRDFBestErr = 10.0e7f;

  for (std::int64_t epochIdx = 0; epochIdx < kNumOptimEpoch; ++epochIdx) {
    std::int64_t numIlluStalled = 0;
    std::int64_t numBRDFStalled = 0;

    // Estimate illumination
    std::clog << "Illumination estimation " << epochIdx << std::endl;

    for (std::int64_t illuEpochIdx = 0; illuEpochIdx < kNumIlluSteps;
         ++illuEpochIdx) {
      torch::Tensor currIllu = dip(noise).squeeze(0).permute({ 1, 2, 0 });

      optix.SetEnvMap(currIllu);
      optix.Render();

      torch::Tensor lossD;
      torch::Tensor loss;
      {
        torch::Tensor colorBuffer =
          optix.GetColorBuffer().set_requires_grad(true);
        torch::Tensor source = (colorBuffer + 1e-6f).log();
        torch::Tensor target = (optix.GetGTColorBuffer() + 1e-6f).log();

        loss =
          torch::l1_loss(source * mask, target * mask, torch::Reduction::Sum);
        lossD = torch::autograd::grad({ loss }, { colorBuffer })[0];

        colorBuffer.set_requires_grad(false);
      }
      optix.SetColorBufferD(lossD);
      optix.SetEnvMapD(0.0f);

      optix.DiffRenderIllu();

      dip->zero_grad();
      currIllu.backward(optix.GetEnvMapD());

      illuOptimizer.step();

      const float maxGrad = optix.GetEnvMapD().abs().max().item<float>();
      std::clog << illuEpochIdx << ": " << loss.item<float>() << " " << maxGrad
                << std::endl;

      if (loss.item<float>() >= lastIlluBestErr) {
        ++numIlluStalled;
        std::clog << "Stalled for " << numIlluStalled << " times" << std::endl;
      } else {
        lastIlluBestErr = loss.item<float>();
        numIlluStalled = 0;
        bestIllu = currIllu.detach();
      }

      if (numIlluStalled > kIlluPatience) {
        break;
      }
    } // End of illumination estimation

    optix.SetEnvMap(bestIllu);

    // Gray world assumption
    if (epochIdx < kNumGrayWorldSteps) {
      torch::NoGradGuard noGradGuard;

      torch::Tensor grayWorld =
        optix.GetEnvMap().mean(-1, true).repeat({ 1, 1, 3 });

      optix.SetEnvMap(grayWorld);
    }

    // Estimate BRDF
    std::clog << "BRDF estimation " << epochIdx << std::endl;

    for (std::int64_t brdfEpochIdx = 0; brdfEpochIdx < kNumBRDFSteps;
         ++brdfEpochIdx) {
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
        torch::Tensor source = (colorBuffer + 1e-6f).log();
        torch::Tensor target = (optix.GetGTColorBuffer() + 1e-6f).log();

        loss =
          0.00001f *
          torch::l1_loss(source * mask, target * mask, torch::Reduction::Sum);
        lossD = torch::autograd::grad({ loss }, { colorBuffer })[0];

        colorBuffer.set_requires_grad(false);
      }

      std::clog << brdfEpochIdx << ": " << loss.item<float>() << std::endl;

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
             (color.exp()).select(1, lobeIdx).unsqueeze(-1))
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

        materialSubset.backward(materialDSubset.to(device));
      }

      torch::Tensor normPrior = 0.00001f * embedCode.pow(2).sum();
      normPrior.backward();

      embedCodeOptimizer.step();
      colorOptimizer.step();

      if (loss.item<float>() >= lastBRDFBestErr) {
        ++numBRDFStalled;
        std::clog << "Stalled for " << numBRDFStalled << " times" << std::endl;
      } else {
        lastBRDFBestErr = loss.item<float>();
        numBRDFStalled = 0;
        bestMat = optix.GetMaterial().detach();
      }

      if (numBRDFStalled > kBRDFPatience) {
        break;
      }
    } // End of estimating BRDF

    optix.SetMaterial(bestMat);
  }

  SaveMERL(kOutMat, bestMat.cpu());
  SaveEXR(kOutIllu, bestIllu.cpu(), kEnvMapWidth, kEnvMapHeight);

  return 0;
}
