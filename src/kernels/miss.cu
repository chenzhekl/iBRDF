#include "util.h"

extern "C"
{
  __constant__ LaunchParams params;
}

extern "C" __global__ void
__miss__radiance_forward()
{}

extern "C" __global__ void
__miss__radiance_backward()
{}
