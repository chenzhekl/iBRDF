#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>

template<typename T>
struct SbtRecord
{
  __align__(
    OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<int> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;
typedef SbtRecord<int> HitGroupSbtRecordNM;

struct LaunchParams
{
  float* Normals;
  std::int32_t* Mask;
  float3 InvViewDir;
  bool Specular;

  struct
  {
    bool Use;
    float Phi;
    float3 Radiance;
  } PointLight;

  struct
  {
    float* ColorBuffer;
    float* GTColorBuffer;
    float* ColorBufferD;
    uint2 Size;
  } FrameBuffer;

  struct
  {
    float* Data;
    uint2 Size;
  } EnvMap;

  struct
  {
    float* Func;
    float* CondCdf;
    float* CondFuncInt;
    float* MargCdf;
    float MargFuncInt;
  } EnvMapDist;

  struct
  {
    float* Data;
    uint2 Size;
  } EnvMapD;

  float* Material;
  float* MaterialD;
};
