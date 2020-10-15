/**
 * This binary renders a image using specified geometry, material and
 * illumination
 *
 * Usage: ./build/bin/render [Geometry] [Material] [Illumination] [Mirror
 * reflection] [Output]
 *
 * Example: ./build/bin/render ./data/sphere.pt ./data/alum-bronze.binary
 * ./data/uffizi-large.exr 0 ./run/render.exr
 */
#include "../backend.h"
#include "../util.h"

int
main(int argc, char** argv)
{
  const std::string kNormal = std::string(argv[1]);
  const std::string kMaterialFile = std::string(argv[2]);
  const std::string kEnvMapFile = std::string(argv[3]);
  const bool kMirror = bool(std::atoi(argv[4]));
  const std::string kOut = std::string(argv[5]);

  std::clog << "--------------" << std::endl
            << "Geometry: " << kNormal << std::endl
            << "Material: " << kMaterialFile << std::endl
            << "Illumination: " << kEnvMapFile << std::endl
            << "Mirror reflection: " << kMirror << std::endl
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
  torch::Tensor normal = LoadNormal(createOptions.NormalFile);
  std::int64_t canvasWidth = normal.size(1);
  std::int64_t canvasHeight = normal.size(0);

  // Camera
  {
    float3 lookFrom = make_float3(0.0f, 0.0f, 1.0f);
    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);

    optix.SetCamera(lookFrom, lookAt);
  }

  // Environment map
  {
    // torch::Tensor envMap =
    //   torch::adaptive_avg_pool2d(
    //     LoadEXR(kEnvMapFile).permute({ 2, 0, 1 }).unsqueeze(0), { 512, 1024
    //     }) .squeeze(0) .permute({ 1, 2, 0 });
    torch::Tensor envMap = LoadEXR(kEnvMapFile);

    optix.SetEnvMap(envMap);
    // optix.SetPointLight(true, -60, make_float3(5.0f, 5.0f, 5.0f));
  }

  // Material
  {
    torch::Tensor material = LoadMERL(kMaterialFile);
    optix.SetMaterial(material);
  }

  // Framebuffer
  {
    optix.AllocateFrameBuffer(canvasWidth, canvasHeight);
  }

  optix.Render(kMirror);
  torch::Tensor image = optix.GetColorBuffer().cpu();

  SaveEXR(kOut, image, canvasWidth, canvasHeight);

  return 0;
}
