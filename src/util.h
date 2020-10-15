#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <torch/torch.h>

#include "ibrdf/ibrdf.h"
// #include "sbtrecord.h"

struct Geometry
{
  std::vector<float3> Vertices;
  std::vector<float3> Normals;
  std::vector<uint3> Indices;
};

std::string
LoadPTX(const std::string& path);

Geometry
LoadPLY(const std::string& path);

torch::Tensor
LoadNormal(const std::string& path);

torch::Tensor
LoadMERL(const std::string& path);

void
SaveMERL(const std::string& path, const torch::Tensor& material);

torch::Tensor
GenerateMERLSamples(ibrdf::IBRDF& model,
                    const torch::Tensor& positions,
                    const torch::Tensor& reference,
                    const std::optional<torch::Tensor>& embedCode,
                    bool unwarp = true);

torch::Tensor
GenerateMERLSlice(ibrdf::IBRDF& model,
                  const torch::Tensor& reference,
                  const std::optional<torch::Tensor>& embedCode,
                  bool unwarp = true);

torch::Tensor
GenerateMERL(ibrdf::IBRDF& model,
             const torch::Tensor& reference,
             const torch::Tensor& embedCode,
             const torch::Tensor& color,
             bool unwarp = true);

torch::Tensor
LoadEXR(const std::string& filename);

bool
SaveEXR(const std::string& filename,
        const torch::Tensor& img,
        std::int64_t width,
        std::int64_t height);

void
CreateDist1D(const torch::Tensor& f, torch::Tensor& pdf, torch::Tensor& cdf);

void
CreateDist2D(const torch::Tensor& f,
             torch::Tensor& condCdf,
             torch::Tensor& condFuncInt,
             torch::Tensor& margCdf,
             float& margFuncInt);

void
CreateEnvMapSamplingDist(const torch::Tensor& envMap,
                         torch::Tensor& func,
                         torch::Tensor& condCdf,
                         torch::Tensor& condFuncInt,
                         torch::Tensor& margCdf,
                         float& margFuncInt);

inline torch::Tensor
TVLoss(const torch::Tensor& x)
{
  return (x.slice(0, 0, -1) - x.slice(0, 1)).abs().sum() +
         (x.slice(1, 0, -1) - x.slice(1, 1)).abs().sum();
}

inline void
Log(const float3& a)
{
  std::clog << a.x << " " << a.y << " " << a.z << std::endl;
}

inline std::vector<float>
Tensor2Vector(const torch::Tensor& a)
{
  torch::Tensor cpuTensor = a.cpu().flatten();
  torch::TensorAccessor accessor = cpuTensor.accessor<float, 1>();
  std::vector<float> ret(accessor.size(0));

  for (std::int64_t i = 0; i < accessor.size(0); ++i) {
    ret[i] = accessor[i];
  }

  return ret;
}

inline void
ZeroGrad(torch::Tensor& a)
{
  if (a.grad().defined()) {
    a.grad().detach_();
    a.grad().zero_();
  }
}

inline void
PrintStats(const torch::Tensor& t)
{
  std::clog << "Min: " << t.min().item<float>()
            << ", Max: " << t.max().item<float>()
            << ", Abs mean: " << t.abs().mean().item<float>()
            << ", Abs median: " << t.abs().median().item<float>() << std::endl;
}

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK(call)                                                      \
  do {                                                                         \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__   \
         << ")\n";                                                             \
      throw std::runtime_error(ss.str().c_str());                              \
    }                                                                          \
  } while (0)

#define OPTIX_CHECK_LOG(call)                                                  \
  do {                                                                         \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__   \
         << ")\nLog:\n"                                                        \
         << log << (sizeofLog > sizeof(log) ? "<TRUNCATED>" : "") << "\n";     \
      throw std::runtime_error(ss.str().c_str());                              \
    }                                                                          \
  } while (0)

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA call (" << #call << " ) failed with error: '"                \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw std::runtime_error(ss.str().c_str());                              \
    }                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA error on synchronize with error '"                           \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw std::runtime_error(ss.str().c_str());                              \
    }                                                                          \
  } while (0)
