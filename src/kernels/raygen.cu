#include "util.h"

extern "C"
{
  __constant__ LaunchParams params;
}

extern "C" __global__ void
__raygen__perspective_forward()
{
  const unsigned int ix = optixGetLaunchIndex().x;
  const unsigned int iy = optixGetLaunchIndex().y;

  const unsigned int width = params.FrameBuffer.Size.x;
  const unsigned int height = params.FrameBuffer.Size.y;

  const unsigned int ind = ix + iy * width;

  if (params.Mask[ind] == 0) {
    return;
  }

  const MERL brdf(params.Material);

  float3 normal = make_float3(params.Normals[ind * 3 + 0],
                              params.Normals[ind * 3 + 1],
                              params.Normals[ind * 3 + 2]);

  float cosThetaO = Dot(normal, params.InvViewDir);
  if (cosThetaO <= 0.0f) {
    return;
  }

  if (params.Specular) {
    float3 wi = Reflect(params.InvViewDir, normal);
    unsigned int lightIdx =
      GetEnvMapIndexNM(wi, params.EnvMap.Size.x, params.EnvMap.Size.y);
    float3 l = make_float3(params.EnvMap.Data[lightIdx * 3 + 0],
                           params.EnvMap.Data[lightIdx * 3 + 1],
                           params.EnvMap.Data[lightIdx * 3 + 2]) *
               cosThetaO;

    params.FrameBuffer.ColorBuffer[ind * 3 + 0] = l.x;
    params.FrameBuffer.ColorBuffer[ind * 3 + 1] = l.y;
    params.FrameBuffer.ColorBuffer[ind * 3 + 2] = l.z;

    return;
  }

  if (params.PointLight.Use) {
    float3 wi = make_float3(
      sinf(params.PointLight.Phi), 0.0f, cosf(params.PointLight.Phi));

    float cosTheta = Dot(wi, normal);

    float3 brdfVal = brdf.Eval(params.InvViewDir, wi, normal);

    float3 li = params.PointLight.Radiance;

    float3 l = li * brdfVal * cosTheta;

    params.FrameBuffer.ColorBuffer[ind * 3 + 0] = l.x;
    params.FrameBuffer.ColorBuffer[ind * 3 + 1] = l.y;
    params.FrameBuffer.ColorBuffer[ind * 3 + 2] = l.z;

    return;
  }

  float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
  for (unsigned int row = 0; row < params.EnvMap.Size.y; ++row) {
    float wiTheta = (row + 0.5f) * kPI / params.EnvMap.Size.y;
    for (unsigned int col = 0; col < params.EnvMap.Size.x; ++col) {
      float wiPhi =
        (col + 0.5f) * 2.0f * kPI / params.EnvMap.Size.x - kPI * 0.5f;
      float wiSinTheta = sinf(wiTheta);

      float3 wi = make_float3(
        wiSinTheta * sinf(wiPhi), cosf(wiTheta), wiSinTheta * cosf(wiPhi));

      float cosTheta = Dot(wi, normal);

      if (cosTheta <= 0.0f) {
        continue;
      }

      unsigned int lightIdx = row * params.EnvMap.Size.x + col;

      float3 li = make_float3(params.EnvMap.Data[lightIdx * 3 + 0],
                              params.EnvMap.Data[lightIdx * 3 + 1],
                              params.EnvMap.Data[lightIdx * 3 + 2]);

      float3 brdfVal = brdf.Eval(params.InvViewDir, wi, normal);

      pixelColor = pixelColor + li * brdfVal * cosTheta * wiSinTheta;
    }
  }

  const float integnorm =
    2.0f * kPI * kPI / (params.EnvMap.Size.x * params.EnvMap.Size.y);
  pixelColor = pixelColor * integnorm;
  params.FrameBuffer.ColorBuffer[ind * 3 + 0] = pixelColor.x;
  params.FrameBuffer.ColorBuffer[ind * 3 + 1] = pixelColor.y;
  params.FrameBuffer.ColorBuffer[ind * 3 + 2] = pixelColor.z;
}

extern "C" __global__ void
__raygen__perspective_backward_illu()
{
  const unsigned int ix = optixGetLaunchIndex().x;
  const unsigned int iy = optixGetLaunchIndex().y;

  const unsigned int width = params.FrameBuffer.Size.x;
  const unsigned int height = params.FrameBuffer.Size.y;

  const unsigned int ind = ix + iy * width;

  const float integnorm =
    2.0f * kPI * kPI / (params.EnvMap.Size.x * params.EnvMap.Size.y);

  if (params.Mask[ind] == 0) {
    return;
  }

  const MERL brdf(params.Material);

  float3 normal = make_float3(params.Normals[ind * 3 + 0],
                              params.Normals[ind * 3 + 1],
                              params.Normals[ind * 3 + 2]);

  float cosThetaO = Dot(normal, params.InvViewDir);
  if (cosThetaO <= 0.0f) {
    return;
  }

  if (params.Specular) {
    float3 wi = Reflect(params.InvViewDir, normal);
    unsigned int lightIdx =
      GetEnvMapIndexNM(wi, params.EnvMap.Size.x, params.EnvMap.Size.y);

    float d = cosThetaO;

    atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 0]),
              params.FrameBuffer.ColorBufferD[ind * 3 + 0] * d);
    atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 1]),
              params.FrameBuffer.ColorBufferD[ind * 3 + 1] * d);
    atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 2]),
              params.FrameBuffer.ColorBufferD[ind * 3 + 2] * d);

    return;
  }

  for (unsigned int row = 0; row < params.EnvMap.Size.y; ++row) {
    float wiTheta = (row + 0.5f) * kPI / params.EnvMap.Size.y;
    for (unsigned int col = 0; col < params.EnvMap.Size.x; ++col) {
      float wiPhi =
        (col + 0.5f) * 2.0f * kPI / params.EnvMap.Size.x - kPI * 0.5f;
      float wiSinTheta = sinf(wiTheta);

      float3 wi = make_float3(
        wiSinTheta * sinf(wiPhi), cosf(wiTheta), wiSinTheta * cosf(wiPhi));

      float cosTheta = Dot(wi, normal);

      if (cosTheta <= 0.0f) {
        continue;
      }

      unsigned int lightIdx = row * params.EnvMap.Size.x + col;

      float3 brdfVal = brdf.Eval(params.InvViewDir, wi, normal);

      float3 d = brdfVal * cosTheta * wiSinTheta * integnorm;

      // if (d.x == 0.0f && d.y == 0.0f && d.z == 0.0f) {
      //   printf("boom");
      // }

      atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 0]),
                params.FrameBuffer.ColorBufferD[ind * 3 + 0] * d.x);
      atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 1]),
                params.FrameBuffer.ColorBufferD[ind * 3 + 1] * d.y);
      atomicAdd(&(params.EnvMapD.Data[lightIdx * 3 + 2]),
                params.FrameBuffer.ColorBufferD[ind * 3 + 2] * d.z);
    }
  }
}

// TODO: implement it
extern "C" __global__ void
__raygen__perspective_backward_brdf()
{
  const unsigned int ix = optixGetLaunchIndex().x;
  const unsigned int iy = optixGetLaunchIndex().y;

  const unsigned int width = params.FrameBuffer.Size.x;
  const unsigned int height = params.FrameBuffer.Size.y;

  const unsigned int ind = ix + iy * width;

  const float integnorm =
    2.0f * kPI * kPI / (params.EnvMap.Size.x * params.EnvMap.Size.y);

  if (params.Mask[ind] == 0) {
    return;
  }

  const MERL brdf(params.Material);

  float3 normal = make_float3(params.Normals[ind * 3 + 0],
                              params.Normals[ind * 3 + 1],
                              params.Normals[ind * 3 + 2]);

  if (Dot(normal, params.InvViewDir) <= 0.0f) {
    return;
  }

  if (params.PointLight.Use) {
    float3 wi = make_float3(
      sinf(params.PointLight.Phi), 0.0f, cosf(params.PointLight.Phi));

    float cosTheta = Dot(wi, normal);

    unsigned int merlId = kMaxUInt32;
    float3 brdfVal = brdf.Eval(params.InvViewDir, wi, normal, &merlId);

    if (merlId == kMaxUInt32) {
      return;
    }

    float3 li = params.PointLight.Radiance;

    constexpr unsigned int kMERLStride = kBRDFSamplingResThetaH *
                                         kBRDFSamplingResThetaD *
                                         kBRDFSamplingResPhiD / 2;
    float3 d = li * cosTheta;
    atomicAdd(&params.MaterialD[merlId],
              params.FrameBuffer.ColorBufferD[ind * 3 + 0] * d.x);
    atomicAdd(&params.MaterialD[merlId + kMERLStride],
              params.FrameBuffer.ColorBufferD[ind * 3 + 1] * d.y);
    atomicAdd(&params.MaterialD[merlId + kMERLStride * 2],
              params.FrameBuffer.ColorBufferD[ind * 3 + 2] * d.z);

    return;
  }

  float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
  for (unsigned int row = 0; row < params.EnvMap.Size.y; ++row) {
    float wiTheta = (row + 0.5f) * kPI / params.EnvMap.Size.y;
    for (unsigned int col = 0; col < params.EnvMap.Size.x; ++col) {
      float wiPhi =
        (col + 0.5f) * 2.0f * kPI / params.EnvMap.Size.x - kPI * 0.5f;
      float wiSinTheta = sinf(wiTheta);

      float3 wi = make_float3(
        wiSinTheta * sinf(wiPhi), cosf(wiTheta), wiSinTheta * cosf(wiPhi));

      float cosTheta = Dot(wi, normal);

      if (cosTheta <= 0.0f) {
        continue;
      }

      unsigned int lightIdx = row * params.EnvMap.Size.x + col;

      float3 li = make_float3(params.EnvMap.Data[lightIdx * 3 + 0],
                              params.EnvMap.Data[lightIdx * 3 + 1],
                              params.EnvMap.Data[lightIdx * 3 + 2]);

      unsigned int merlId = kMaxUInt32;
      float3 brdfVal = brdf.Eval(params.InvViewDir, wi, normal, &merlId);

      if (merlId == kMaxUInt32) {
        continue;
      }

      constexpr unsigned int kMERLStride = kBRDFSamplingResThetaH *
                                           kBRDFSamplingResThetaD *
                                           kBRDFSamplingResPhiD / 2;
      float3 d = li * cosTheta * wiSinTheta * integnorm;
      atomicAdd(&params.MaterialD[merlId],
                params.FrameBuffer.ColorBufferD[ind * 3 + 0] * d.x);
      atomicAdd(&params.MaterialD[merlId + kMERLStride],
                params.FrameBuffer.ColorBufferD[ind * 3 + 1] * d.y);
      atomicAdd(&params.MaterialD[merlId + kMERLStride * 2],
                params.FrameBuffer.ColorBufferD[ind * 3 + 2] * d.z);
    }
  }
}
