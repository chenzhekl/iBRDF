#pragma once

#include "../sbtrecord.h"
#include "envmap.h"
#include "math.h"
#include "memory.h"
#include "merl.h"
#include "pcg_rng.h"
#include "phong.h"

__device__ inline void
Log(const float3& a)
{
  printf("%f %f %f\n", a.x, a.y, a.z);
}

__device__ inline float
PowerHeuristic(unsigned int nF, float fPdf, unsigned int nG, float gPdf)
{
  float f = nF * fPdf;
  float g = nG * gPdf;

  return (f * f) / (f * f + g * g);
}
