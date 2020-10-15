/**
 * This binary estimates illumination given geometry and material.
 *
 * Usage: ./build/bin/estillu [Input] [Geometry] [Material] [Mirror reflection]
 * [Illumination width] [Illumination height] [Output]
 *
 * Example: ./build/bin/estillu ./run/a.exr ./data/sphere.pt
 * ./data/alum-bronze.binary 0 512 256 2000 ./run/illu.exr
 */
#include "../backend.h"
#include "../math.h"
#include "../nn/dcgan.h"
#include "../nn/downsampler.h"
#include "../nn/skip.h"
#include "../priors.h"
#include "../ssim.h"
#include "../util.h"

int
main(int argc, char** argv)
{
  torch::manual_seed(0);
  torch::Device device(torch::kCUDA);

  const std::string kIn = std::string(argv[1]);
  const std::string kNormal = std::string(argv[2]);
  const std::string kMaterialFile = std::string(argv[3]);
  const bool kMirror = bool(std::atoi(argv[4]));
  const std::int64_t kEnvMapWidth = std::atoi(argv[5]);
  const std::int64_t kEnvMapHeight = std::atoi(argv[6]);
  const std::int64_t kNumOptimEpoch = std::atoi(argv[7]);
  const std::string kOut = std::string(argv[8]);

  std::clog << "--------------" << std::endl
            << "Input: " << kIn << std::endl
            << "Geometry: " << kNormal << std::endl
            << "Material: " << kMaterialFile << std::endl
            << "Mirror reflection: " << kMirror << std::endl
            << "Illumination size: " << kEnvMapWidth << " x " << kEnvMapHeight
            << std::endl
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
  const std::int64_t canvasWidth = mask.size(1);
  const std::int64_t canvasHeight = mask.size(0);

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
  {
    torch::Tensor material = LoadMERL(kMaterialFile);
    optix.SetMaterial(material);
  }

  // Framebuffer
  {
    torch::Tensor gtImage = LoadEXR(kIn);
    optix.AllocateFrameBuffer(canvasWidth, canvasHeight);
    optix.SetGTColorBuffer(gtImage);
  }

  // Deep illumination prior
  Skip dip(8,
           3,
           std::vector<std::int64_t>{ 128, 128, 128, 128, 128 },
           std::vector<std::int64_t>{ 128, 128, 128, 128, 128 },
           std::vector<std::int64_t>{ 16, 16, 16, 16, 16 });
  dip->to(device);
  dip->train();

  // std::clog << dip << std::endl;

  torch::Tensor noise =
    torch::empty({ 1, 8, kEnvMapHeight, kEnvMapWidth }, device)
      .uniform_(0.0f, 0.1f);
  // torch::Tensor noise =
  //   torch::stack(
  //     torch::meshgrid({
  //     torch::arange(static_cast<std::int64_t>(kEnvMapDHeight),
  //                                     torch::kFloat32) /
  //                         static_cast<float>(kEnvMapDHeight - 1),
  //                       torch::arange(static_cast<std::int64_t>(kEnvMapDWidth),
  //                                     torch::kFloat32) /
  //                         static_cast<float>(kEnvMapDWidth - 1) }),
  //     0)
  //     .unsqueeze(0)
  //     .to(device);
  // torch::Tensor noise = torch::rand({ 1, 100, 1, 1 }, device);

  // Optimizer
  torch::optim::Adam optimizer(dip->parameters(),
                               torch::optim::AdamOptions(0.001));

  float bestErr = 10e7f;
  torch::Tensor bestIllu;
  std::int64_t numStalls = 0;

  for (std::int64_t i = 0; i < kNumOptimEpoch; ++i) {
    torch::Tensor currIllu = dip(noise).squeeze(0).permute({ 1, 2, 0 });

    optix.SetEnvMap(currIllu);
    optix.Render(false);

    torch::Tensor lossD;
    torch::Tensor loss;
    {
      torch::Tensor colorBuffer =
        optix.GetColorBuffer().set_requires_grad(true);
      torch::Tensor source = (colorBuffer + 1e-7f).log();
      torch::Tensor target = (optix.GetGTColorBuffer() + 1e-7f).log();

      loss =
        torch::l1_loss(source * mask, target * mask, torch::Reduction::Sum);
      // loss = 1.0 - SSIM(source * mask, target * mask);
      lossD = torch::autograd::grad({ loss }, { colorBuffer })[0];

      colorBuffer.set_requires_grad(false);
    }
    optix.SetColorBufferD(lossD);
    optix.SetEnvMapD(0.0f);
    optix.DiffRenderIllu(false);

    dip->zero_grad();
    currIllu.backward(optix.GetEnvMapD());

    std::clog << i << ": " << loss.item<float>() << std::endl;

    optimizer.step();

    if (loss.item<float>() >= bestErr) {
      ++numStalls;
    } else {
      bestErr = loss.item<float>();
      bestIllu = currIllu.detach();
      numStalls = 0;
    }

    // if (numStalls > 50) {
    //   break;
    // }
  }

  SaveEXR(kOut, bestIllu.detach().cpu(), kEnvMapWidth, kEnvMapHeight);

  return 0;
}
