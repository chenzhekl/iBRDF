#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <torch/torch.h>

#include "kernels/merl_constant.h"
#include "math.h"
#include "util.h"

struct OptiXBackendCreateOptions
{
  std::string MissPTXFile;
  std::string RayGenPTXFile;
  std::string HitGroupPTXFile;
  std::string NormalFile;
  bool Debug;
};

class OptiXBackend
{
public:
  explicit OptiXBackend(const OptiXBackendCreateOptions& options);
  ~OptiXBackend();

  void SetCamera(const float3& lookFrom, const float3& lookAt);
  void SetEnvMap(const torch::Tensor& data);
  void SetEnvMap(float fillValue);
  void SetEnvMapD(const torch::Tensor& data);
  void SetEnvMapD(float fillValue);
  void SetGTColorBuffer(const torch::Tensor& gtColorBuffer);
  void SetColorBufferD(const torch::Tensor& colorBufferD);
  void SetMaterial(const torch::Tensor& data);
  void SetMaterialD(float fillValue);
  void SetPointLight(bool use, float phi, const float3& radiance);

  void AllocateEnvMap(std::size_t width, std::size_t height);
  void AllocateEnvMapD(std::size_t width, std::size_t height);
  void AllocateFrameBuffer(std::size_t width, std::size_t height);
  void AllocateMaterial();
  void AllocateMaterialD();

  void Render(bool specular = false);
  void DiffRenderIllu(bool specular = false);
  void DiffRenderBRDF();

  torch::Tensor& GetEnvMap();
  torch::Tensor& GetEnvMapD();
  torch::Tensor& GetColorBuffer();
  torch::Tensor& GetGTColorBuffer();
  torch::Tensor& GetMaterial();
  torch::Tensor& GetMaterialD();

private:
  void InitOptiX(const OptiXBackendCreateOptions& option);

  void SetupAS(const OptiXBackendCreateOptions& option);
  void SetupModules(const OptiXBackendCreateOptions& option,
                    OptixPipelineCompileOptions& pipelineCompileOptions);
  void SetupProgramGroups(const OptiXBackendCreateOptions& option);
  void SetupPipelines(
    const OptiXBackendCreateOptions& option,
    const OptixPipelineCompileOptions& pipelineCompileOptions);
  void SetupSBT(const OptiXBackendCreateOptions& option);
  void SetupDenoiser(const OptiXBackendCreateOptions& option);

  void Cleanup();

  // Optix
  CUstream mCUDAStream = nullptr;
  OptixDeviceContext mOptixDeviceContext = nullptr;

  OptixModule mMissModule = nullptr;
  OptixModule mHitGroupModule = nullptr;
  OptixModule mRayGenModule = nullptr;

  OptixProgramGroup mForwardRaygenProgGroup = nullptr;
  OptixProgramGroup mBackwardIlluRaygenProgGroup = nullptr;
  OptixProgramGroup mBackwardBRDFRaygenProgGroup = nullptr;
  OptixProgramGroup mForwardMissProgGroup = nullptr;
  OptixProgramGroup mBackwardMissProgGroup = nullptr;
  OptixProgramGroup mHitgroupRadianceProgGroup = nullptr;
  OptixProgramGroup mHitgroupOcclusionProgGroup = nullptr;

  OptixPipeline mOptixForwardPipeline = nullptr;
  OptixPipeline mOptixBackwardIlluPipeline = nullptr;
  OptixPipeline mOptixBackwardBRDFPipeline = nullptr;

  OptixShaderBindingTable mForwardSBT = {};
  OptixShaderBindingTable mBackwardIlluSBT = {};
  OptixShaderBindingTable mBackwardBRDFSBT = {};

  OptixTraversableHandle mGASHandle = {};
  CUdeviceptr mGASOutputBuffer = 0;
  CUdeviceptr mAABB = 0;

  torch::Tensor mNormals;
  torch::Tensor mMask;
  float3 mInvViewDir;

  struct
  {
    bool Use = false;
    float Phi;
    float3 Radiance;
  } mPointLight;

  // Camera
  float3 mOrigin;
  float3 mUpperLeftCorner;
  float3 mHorizontal;
  float3 mVertical;

  // Environment map
  torch::Tensor mEnvMap;
  torch::Tensor mEnvMapD;

  struct
  {
    torch::Tensor func;
    torch::Tensor condCdf;
    torch::Tensor condFuncInt;
    torch::Tensor margCdf;
    float margFuncInt;
  } mEnvMapDist;

  // Framebuffer
  torch::Tensor mColorBuffer;
  torch::Tensor mGTColorBuffer;
  torch::Tensor mColorBufferD;

  // MERL material
  torch::Tensor mMaterial;
  torch::Tensor mMaterialD;

  // Launch parameters
  CUdeviceptr mLaunchParams = 0;
};

inline OptiXBackend::~OptiXBackend()
{
  Cleanup();
}

inline void
OptiXBackend::AllocateEnvMap(std::size_t width, std::size_t height)
{
  mEnvMap = torch::zeros(
    { static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3 },
    torch::device(torch::kCUDA).dtype(torch::kFloat));
  CreateEnvMapSamplingDist(mEnvMap,
                           mEnvMapDist.func,
                           mEnvMapDist.condCdf,
                           mEnvMapDist.condFuncInt,
                           mEnvMapDist.margCdf,
                           mEnvMapDist.margFuncInt);
}

inline void
OptiXBackend::AllocateEnvMapD(std::size_t width, std::size_t height)
{
  mEnvMapD = torch::zeros(
    { static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3 },
    torch::device(torch::kCUDA).dtype(torch::kFloat));
}

inline void
OptiXBackend::AllocateFrameBuffer(std::size_t width, std::size_t height)
{
  mColorBuffer = torch::zeros(
    { static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3 },
    torch::device(torch::kCUDA).dtype(torch::kFloat));
}

inline void
OptiXBackend::AllocateMaterial()
{
  mMaterial = torch::ones({ 3,
                            kBRDFSamplingResThetaH,
                            kBRDFSamplingResThetaD,
                            kBRDFSamplingResPhiD / 2 },
                          torch::device(torch::kCUDA).dtype(torch::kFloat));
}

inline void
OptiXBackend::AllocateMaterialD()
{
  mMaterialD = torch::zeros({ 3,
                              kBRDFSamplingResThetaH,
                              kBRDFSamplingResThetaD,
                              kBRDFSamplingResPhiD / 2 },
                            torch::device(torch::kCUDA).dtype(torch::kFloat));
}

inline void
OptiXBackend::SetEnvMap(const torch::Tensor& data)
{
  mEnvMap = data.cuda();
  CreateEnvMapSamplingDist(mEnvMap,
                           mEnvMapDist.func,
                           mEnvMapDist.condCdf,
                           mEnvMapDist.condFuncInt,
                           mEnvMapDist.margCdf,
                           mEnvMapDist.margFuncInt);
}

inline void
OptiXBackend::SetEnvMap(float fillValue)
{
  mEnvMap.fill_(fillValue);
  CreateEnvMapSamplingDist(mEnvMap,
                           mEnvMapDist.func,
                           mEnvMapDist.condCdf,
                           mEnvMapDist.condFuncInt,
                           mEnvMapDist.margCdf,
                           mEnvMapDist.margFuncInt);
}

inline void
OptiXBackend::SetEnvMapD(const torch::Tensor& data)
{
  mEnvMapD = data.cuda();
}

inline void
OptiXBackend::SetEnvMapD(float fillValue)
{
  mEnvMapD.fill_(fillValue);
}

inline void
OptiXBackend::SetGTColorBuffer(const torch::Tensor& gtColorBuffer)
{
  mGTColorBuffer = gtColorBuffer.cuda();
}

inline void
OptiXBackend::SetColorBufferD(const torch::Tensor& colorBufferD)
{
  mColorBufferD = colorBufferD.cuda();
}

inline void
OptiXBackend::SetMaterial(const torch::Tensor& data)
{
  mMaterial = data.cuda();
}

inline void
OptiXBackend::SetMaterialD(float fillValue)
{
  mMaterialD.fill_(fillValue);
}

inline void
OptiXBackend::SetPointLight(bool use, float phi, const float3& radiance)
{
  mPointLight.Use = use;
  mPointLight.Phi = phi / 180.0 * kPI;
  mPointLight.Radiance = radiance;
}

inline torch::Tensor&
OptiXBackend::GetEnvMap()
{
  return mEnvMap;
}

inline torch::Tensor&
OptiXBackend::GetEnvMapD()
{
  return mEnvMapD;
}

inline torch::Tensor&
OptiXBackend::GetColorBuffer()
{
  return mColorBuffer;
}

inline torch::Tensor&
OptiXBackend::GetGTColorBuffer()
{
  return mGTColorBuffer;
}

inline torch::Tensor&
OptiXBackend::GetMaterial()
{
  return mMaterial;
}

inline torch::Tensor&
OptiXBackend::GetMaterialD()
{
  return mMaterialD;
}
