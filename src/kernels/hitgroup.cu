#include <cuda_runtime.h>

#include "util.h"

extern "C"
{
  __constant__ LaunchParams params;
}

extern "C" __global__ void
__closesthit__radiance()
{}

extern "C" __global__ void
__anyhit__occlusion()
{}
