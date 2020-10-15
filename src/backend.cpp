#include "backend.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <optix_function_table_definition.h>

#include "kernels/merl_constant.h"
#include "math.h"
#include "sbtrecord.h"
#include "util.h"

static void
ContextLogCB(unsigned int level,
             const char* tag,
             const char* message,
             void* /*cbdata */)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << "\n";
}

OptiXBackend::OptiXBackend(const OptiXBackendCreateOptions& option)
{
  InitOptiX(option);
  SetupAS(option);

  OptixPipelineCompileOptions pipelineCompileOptions = {};
  SetupModules(option, pipelineCompileOptions);
  SetupProgramGroups(option);
  SetupPipelines(option, pipelineCompileOptions);
  SetupSBT(option);
}

void
OptiXBackend::InitOptiX(const OptiXBackendCreateOptions& option)
{
  // Initialize CUDA
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaStreamCreate(&mCUDAStream));

  // Zero means take the current context
  CUcontext cuCtx = 0;
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &ContextLogCB;
  if (option.Debug) {
    options.logCallbackLevel = 4;
  } else {
    options.logCallbackLevel = 0;
  }

  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &mOptixDeviceContext));
}

void
OptiXBackend::SetupAS(const OptiXBackendCreateOptions& option)
{
  mNormals = LoadNormal(option.NormalFile).cuda();
  mMask = (mNormals.sum(-1) != 0.0f).to(torch::kInt32);

  OptixBuildInput buildInputs[1];
  std::memset(buildInputs, 0, sizeof(OptixBuildInput));

  OptixAabb aabb;
  aabb.minX = -200.0f;
  aabb.minY = -200.0f;
  aabb.minZ = -200.0f;
  aabb.maxX = 200.0f;
  aabb.maxY = 200.0f;
  aabb.maxZ = 200.0f;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mAABB), sizeof(aabb)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mAABB),
                        &aabb,
                        sizeof(aabb),
                        cudaMemcpyHostToDevice));

  OptixBuildInputCustomPrimitiveArray& buildInput = buildInputs[0].aabbArray;
  buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  buildInput.aabbBuffers = &mAABB;
  buildInput.numPrimitives = 1;
  buildInput.numSbtRecords = 1;
  buildInput.sbtIndexOffsetBuffer = 0;
  buildInput.sbtIndexOffsetSizeInBytes = 0;
  buildInput.sbtIndexOffsetStrideInBytes = 0;

  unsigned int flagsPerSBTRecord[1];
  flagsPerSBTRecord[0] = OPTIX_GEOMETRY_FLAG_NONE;

  buildInput.flags = flagsPerSBTRecord;

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes bufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
    mOptixDeviceContext, &accelOptions, buildInputs, 1, &bufferSizes));

  CUdeviceptr dOutput, dTemp;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput),
                        bufferSizes.outputSizeInBytes));
  CUDA_CHECK(
    cudaMalloc(reinterpret_cast<void**>(&dTemp), bufferSizes.tempSizeInBytes));

  CUdeviceptr dCompactedSize;
  CUDA_CHECK(
    cudaMalloc(reinterpret_cast<void**>(&dCompactedSize), sizeof(std::size_t)));
  OptixAccelEmitDesc accelEmitDesc = {};
  accelEmitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  accelEmitDesc.result = dCompactedSize;

  OPTIX_CHECK(optixAccelBuild(mOptixDeviceContext,
                              mCUDAStream,
                              &accelOptions,
                              buildInputs,
                              1,
                              dTemp,
                              bufferSizes.tempSizeInBytes,
                              dOutput,
                              bufferSizes.outputSizeInBytes,
                              &mGASHandle,
                              &accelEmitDesc,
                              1));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTemp)));
  mGASOutputBuffer = dOutput;

  std::size_t compactedSize;
  CUDA_CHECK(cudaMemcpy(&compactedSize,
                        reinterpret_cast<void*>(dCompactedSize),
                        sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dCompactedSize)));

  if (compactedSize < bufferSizes.outputSizeInBytes) {
    CUdeviceptr dCompactedOutput;
    CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dCompactedOutput), compactedSize));

    OptixTraversableHandle compactedGASHandle = {};
    OPTIX_CHECK(optixAccelCompact(mOptixDeviceContext,
                                  mCUDAStream,
                                  mGASHandle,
                                  dCompactedOutput,
                                  compactedSize,
                                  &compactedGASHandle));

    mGASHandle = compactedGASHandle;
    mGASOutputBuffer = dCompactedOutput;

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dOutput)));
  }
}

void
OptiXBackend::SetupModules(const OptiXBackendCreateOptions& option,
                           OptixPipelineCompileOptions& pipelineCompileOptions)
{
  OptixModuleCompileOptions moduleCompileOptions = {};
  moduleCompileOptions.maxRegisterCount =
    OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.traversableGraphFlags =
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags =
    OPTIX_EXCEPTION_FLAG_NONE; // OPTIX_EXCEPTION_FLAG_DEBUG
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

  std::string missPTX = LoadPTX(option.MissPTXFile);
  std::string hitgroupPTX = LoadPTX(option.HitGroupPTXFile);
  std::string raygenPTX = LoadPTX(option.RayGenPTXFile);

  char log[2048];
  size_t sizeofLog = sizeof(log);

  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(mOptixDeviceContext,
                                           &moduleCompileOptions,
                                           &pipelineCompileOptions,
                                           missPTX.c_str(),
                                           missPTX.size(),
                                           log,
                                           &sizeofLog,
                                           &mMissModule));
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(mOptixDeviceContext,
                                           &moduleCompileOptions,
                                           &pipelineCompileOptions,
                                           hitgroupPTX.c_str(),
                                           hitgroupPTX.size(),
                                           log,
                                           &sizeofLog,
                                           &mHitGroupModule));
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(mOptixDeviceContext,
                                           &moduleCompileOptions,
                                           &pipelineCompileOptions,
                                           raygenPTX.c_str(),
                                           raygenPTX.size(),
                                           log,
                                           &sizeofLog,
                                           &mRayGenModule));
}

void
OptiXBackend::SetupProgramGroups(const OptiXBackendCreateOptions&)
{
  OptixProgramGroupOptions programGroupOptions = {}; // Initialize to zeros

  OptixProgramGroupDesc raygenProgGroupDesc = {};
  raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module = mRayGenModule;
  raygenProgGroupDesc.raygen.entryFunctionName =
    "__raygen__perspective_forward";
  char log[2048];
  size_t sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &raygenProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mForwardRaygenProgGroup));

  std::memset(&raygenProgGroupDesc, 0, sizeof(raygenProgGroupDesc));
  raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module = mRayGenModule;
  raygenProgGroupDesc.raygen.entryFunctionName =
    "__raygen__perspective_backward_illu";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &raygenProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mBackwardIlluRaygenProgGroup));

  std::memset(&raygenProgGroupDesc, 0, sizeof(raygenProgGroupDesc));
  raygenProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module = mRayGenModule;
  raygenProgGroupDesc.raygen.entryFunctionName =
    "__raygen__perspective_backward_brdf";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &raygenProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mBackwardBRDFRaygenProgGroup));

  OptixProgramGroupDesc missProgGroupDesc = {};
  missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  missProgGroupDesc.miss.module = mMissModule;
  missProgGroupDesc.miss.entryFunctionName = "__miss__radiance_forward";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &missProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mForwardMissProgGroup));

  std::memset(&missProgGroupDesc, 0, sizeof(missProgGroupDesc));
  missProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  missProgGroupDesc.miss.module = mMissModule;
  missProgGroupDesc.miss.entryFunctionName = "__miss__radiance_backward";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &missProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mBackwardMissProgGroup));

  OptixProgramGroupDesc hitgroupProgGroupDesc = {};
  hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroupProgGroupDesc.hitgroup.moduleCH = mHitGroupModule;
  hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &hitgroupProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mHitgroupRadianceProgGroup));

  std::memset(&hitgroupProgGroupDesc, 0, sizeof(OptixProgramGroupDesc));
  hitgroupProgGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroupProgGroupDesc.hitgroup.moduleAH = mHitGroupModule;
  hitgroupProgGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";
  sizeofLog = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(mOptixDeviceContext,
                                          &hitgroupProgGroupDesc,
                                          1, // num program groups
                                          &programGroupOptions,
                                          log,
                                          &sizeofLog,
                                          &mHitgroupOcclusionProgGroup));
}

void
OptiXBackend::SetupPipelines(
  const OptiXBackendCreateOptions&,
  const OptixPipelineCompileOptions& pipelineCompileOptions)
{
  OptixProgramGroup forwardProgramGroups[] = { mForwardRaygenProgGroup,
                                               mForwardMissProgGroup,
                                               mHitgroupRadianceProgGroup,
                                               mHitgroupOcclusionProgGroup };
  OptixProgramGroup backwardIlluProgramGroups[] = {
    mBackwardIlluRaygenProgGroup,
    mBackwardMissProgGroup,
    mHitgroupRadianceProgGroup,
    mHitgroupOcclusionProgGroup
  };
  OptixProgramGroup backwardBRDFProgramGroups[] = {
    mBackwardBRDFRaygenProgGroup,
    mForwardMissProgGroup,
    mHitgroupRadianceProgGroup,
    mHitgroupOcclusionProgGroup
  };

  OptixPipelineLinkOptions pipelineLinkOptions = {};
  pipelineLinkOptions.maxTraceDepth = 5;
  pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  pipelineLinkOptions.overrideUsesMotionBlur = false;

  char log[2048];
  size_t sizeofLog = sizeof(log);

  OPTIX_CHECK_LOG(optixPipelineCreate(mOptixDeviceContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      forwardProgramGroups,
                                      sizeof(forwardProgramGroups) /
                                        sizeof(forwardProgramGroups[0]),
                                      log,
                                      &sizeofLog,
                                      &mOptixForwardPipeline));
  OPTIX_CHECK_LOG(optixPipelineCreate(mOptixDeviceContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      backwardIlluProgramGroups,
                                      sizeof(backwardIlluProgramGroups) /
                                        sizeof(backwardIlluProgramGroups[0]),
                                      log,
                                      &sizeofLog,
                                      &mOptixBackwardIlluPipeline));
  OPTIX_CHECK_LOG(optixPipelineCreate(mOptixDeviceContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      backwardBRDFProgramGroups,
                                      sizeof(backwardBRDFProgramGroups) /
                                        sizeof(backwardBRDFProgramGroups[0]),
                                      log,
                                      &sizeofLog,
                                      &mOptixBackwardBRDFPipeline));
}

void
OptiXBackend::SetupSBT(const OptiXBackendCreateOptions&)
{
  // Ray generation SBT
  CUdeviceptr forwardRaygenRecord;
  const std::size_t raygenRecordSize = sizeof(RayGenSbtRecord);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&forwardRaygenRecord),
                        raygenRecordSize));
  RayGenSbtRecord rgSbtForward;
  OPTIX_CHECK(optixSbtRecordPackHeader(mForwardRaygenProgGroup, &rgSbtForward));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(forwardRaygenRecord),
                        &rgSbtForward,
                        raygenRecordSize,
                        cudaMemcpyHostToDevice));

  CUdeviceptr backwardIlluRaygenRecord;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&backwardIlluRaygenRecord),
                        raygenRecordSize));
  RayGenSbtRecord rgSbtBackwardIllu;
  OPTIX_CHECK(
    optixSbtRecordPackHeader(mBackwardIlluRaygenProgGroup, &rgSbtBackwardIllu));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(backwardIlluRaygenRecord),
                        &rgSbtBackwardIllu,
                        raygenRecordSize,
                        cudaMemcpyHostToDevice));

  CUdeviceptr backwardBRDFRaygenRecord;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&backwardBRDFRaygenRecord),
                        raygenRecordSize));
  RayGenSbtRecord rgSbtBackwardBRDF;
  OPTIX_CHECK(
    optixSbtRecordPackHeader(mBackwardBRDFRaygenProgGroup, &rgSbtBackwardBRDF));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(backwardBRDFRaygenRecord),
                        &rgSbtBackwardBRDF,
                        raygenRecordSize,
                        cudaMemcpyHostToDevice));

  // Miss SBT
  CUdeviceptr forwardMissRecord;
  std::size_t missRecordSize = sizeof(MissSbtRecord);

  CUDA_CHECK(
    cudaMalloc(reinterpret_cast<void**>(&forwardMissRecord), missRecordSize));
  MissSbtRecord msSbtForward;
  OPTIX_CHECK(optixSbtRecordPackHeader(mForwardMissProgGroup, &msSbtForward));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(forwardMissRecord),
                        &msSbtForward,
                        missRecordSize,
                        cudaMemcpyHostToDevice));

  CUdeviceptr backwardMissRecord;
  CUDA_CHECK(
    cudaMalloc(reinterpret_cast<void**>(&backwardMissRecord), missRecordSize));
  MissSbtRecord msSbtBackward;
  OPTIX_CHECK(optixSbtRecordPackHeader(mBackwardMissProgGroup, &msSbtBackward));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(backwardMissRecord),
                        &msSbtBackward,
                        missRecordSize,
                        cudaMemcpyHostToDevice));

  // Hit group SBT
  CUdeviceptr hitgroupRecord;
  std::size_t hitgroupRecordSize = sizeof(HitGroupSbtRecordNM);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecord),
                        hitgroupRecordSize * 2));
  HitGroupSbtRecordNM hgSbt[2];
  OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupRadianceProgGroup, &hgSbt[0]));
  OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupOcclusionProgGroup, &hgSbt[1]));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecord),
                        &hgSbt,
                        hitgroupRecordSize * 2,
                        cudaMemcpyHostToDevice));

  mForwardSBT.raygenRecord = forwardRaygenRecord;
  mForwardSBT.missRecordBase = forwardMissRecord;
  mForwardSBT.missRecordStrideInBytes = sizeof(MissSbtRecord);
  mForwardSBT.missRecordCount = 1;
  mForwardSBT.hitgroupRecordBase = hitgroupRecord;
  mForwardSBT.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecordNM);
  mForwardSBT.hitgroupRecordCount = 2;

  mBackwardIlluSBT.raygenRecord = backwardIlluRaygenRecord;
  mBackwardIlluSBT.missRecordBase = backwardMissRecord;
  mBackwardIlluSBT.missRecordStrideInBytes = sizeof(MissSbtRecord);
  mBackwardIlluSBT.missRecordCount = 1;
  mBackwardIlluSBT.hitgroupRecordBase = hitgroupRecord;
  mBackwardIlluSBT.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecordNM);
  mBackwardIlluSBT.hitgroupRecordCount = 2;

  mBackwardBRDFSBT.raygenRecord = backwardBRDFRaygenRecord;
  mBackwardBRDFSBT.missRecordBase = forwardMissRecord;
  mBackwardBRDFSBT.missRecordStrideInBytes = sizeof(MissSbtRecord);
  mBackwardBRDFSBT.missRecordCount = 1;
  mBackwardBRDFSBT.hitgroupRecordBase = hitgroupRecord;
  mBackwardBRDFSBT.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecordNM);
  mBackwardBRDFSBT.hitgroupRecordCount = 2;
}

void
OptiXBackend::SetCamera(const float3& lookFrom, const float3& lookAt)
{
  mInvViewDir = Normalize(lookFrom - lookAt);
}

void
OptiXBackend::Render(bool specular)
{
  mColorBuffer = mColorBuffer.contiguous();
  mEnvMap = mEnvMap.contiguous();
  mEnvMapDist.condCdf = mEnvMapDist.condCdf.contiguous();
  mEnvMapDist.condFuncInt = mEnvMapDist.condFuncInt.contiguous();
  mEnvMapDist.func = mEnvMapDist.func.contiguous();
  mEnvMapDist.margCdf = mEnvMapDist.margCdf.contiguous();
  mNormals = mNormals.contiguous();
  mMask = mMask.contiguous();

  LaunchParams params;
  params.Normals = static_cast<float*>(mNormals.data_ptr());
  params.Mask = static_cast<std::int32_t*>(mMask.data_ptr());
  params.Specular = specular;
  params.PointLight.Use = mPointLight.Use;
  params.PointLight.Phi = mPointLight.Phi;
  params.PointLight.Radiance = mPointLight.Radiance;
  params.FrameBuffer.ColorBuffer = static_cast<float*>(mColorBuffer.data_ptr());
  params.FrameBuffer.Size =
    make_uint2(mColorBuffer.size(1), mColorBuffer.size(0));
  params.InvViewDir = mInvViewDir;
  params.EnvMap.Data = static_cast<float*>(mEnvMap.data_ptr());
  params.EnvMap.Size = make_uint2(mEnvMap.size(1), mEnvMap.size(0));
  params.EnvMapDist.CondCdf =
    static_cast<float*>(mEnvMapDist.condCdf.data_ptr());
  params.EnvMapDist.CondFuncInt =
    static_cast<float*>(mEnvMapDist.condFuncInt.data_ptr());
  params.EnvMapDist.Func = static_cast<float*>(mEnvMapDist.func.data_ptr());
  params.EnvMapDist.MargCdf =
    static_cast<float*>(mEnvMapDist.margCdf.data_ptr());
  params.EnvMapDist.MargFuncInt = mEnvMapDist.margFuncInt;
  params.Material = static_cast<float*>(mMaterial.data_ptr());

  if (mLaunchParams == 0) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mLaunchParams),
                          sizeof(LaunchParams)));
  }

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mLaunchParams),
                        &params,
                        sizeof(params),
                        cudaMemcpyHostToDevice));

  OPTIX_CHECK(optixLaunch(mOptixForwardPipeline,
                          mCUDAStream,
                          mLaunchParams,
                          sizeof(LaunchParams),
                          &mForwardSBT,
                          mColorBuffer.size(1),
                          mColorBuffer.size(0),
                          1));

  CUDA_SYNC_CHECK();
}

void
OptiXBackend::DiffRenderIllu(bool specular)
{
  mColorBuffer = mColorBuffer.contiguous();
  mGTColorBuffer = mGTColorBuffer.contiguous();
  mColorBufferD = mColorBufferD.contiguous();
  mEnvMap = mEnvMap.contiguous();
  mEnvMapDist.condCdf = mEnvMapDist.condCdf.contiguous();
  mEnvMapDist.condFuncInt = mEnvMapDist.condFuncInt.contiguous();
  mEnvMapDist.func = mEnvMapDist.func.contiguous();
  mEnvMapDist.margCdf = mEnvMapDist.margCdf.contiguous();
  mEnvMapD = mEnvMapD.contiguous();
  mMaterial = mMaterial.contiguous();
  mNormals = mNormals.contiguous();
  mMask = mMask.contiguous();

  LaunchParams params = {};
  params.Normals = static_cast<float*>(mNormals.data_ptr());
  params.Mask = static_cast<std::int32_t*>(mMask.data_ptr());
  params.Specular = specular;
  params.FrameBuffer.ColorBuffer = static_cast<float*>(mColorBuffer.data_ptr());
  params.FrameBuffer.GTColorBuffer =
    static_cast<float*>(mGTColorBuffer.data_ptr());
  params.FrameBuffer.ColorBufferD =
    static_cast<float*>(mColorBufferD.data_ptr());
  params.FrameBuffer.Size =
    make_uint2(mColorBuffer.size(1), mColorBuffer.size(0));
  params.InvViewDir = mInvViewDir;
  params.EnvMap.Data = static_cast<float*>(mEnvMap.data_ptr());
  params.EnvMap.Size = make_uint2(mEnvMap.size(1), mEnvMap.size(0));
  params.EnvMapD.Data = static_cast<float*>(mEnvMapD.data_ptr());
  params.EnvMapD.Size = make_uint2(mEnvMapD.size(1), mEnvMapD.size(0));
  params.EnvMapDist.CondCdf =
    static_cast<float*>(mEnvMapDist.condCdf.data_ptr());
  params.EnvMapDist.CondFuncInt =
    static_cast<float*>(mEnvMapDist.condFuncInt.data_ptr());
  params.EnvMapDist.Func = static_cast<float*>(mEnvMapDist.func.data_ptr());
  params.EnvMapDist.MargCdf =
    static_cast<float*>(mEnvMapDist.margCdf.data_ptr());
  params.EnvMapDist.MargFuncInt = mEnvMapDist.margFuncInt;
  params.Material = static_cast<float*>(mMaterial.data_ptr());

  if (mLaunchParams == 0) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mLaunchParams),
                          sizeof(LaunchParams)));
  }

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mLaunchParams),
                        &params,
                        sizeof(params),
                        cudaMemcpyHostToDevice));

  OPTIX_CHECK(optixLaunch(mOptixBackwardIlluPipeline,
                          mCUDAStream,
                          mLaunchParams,
                          sizeof(LaunchParams),
                          &mBackwardIlluSBT,
                          mColorBuffer.size(1),
                          mColorBuffer.size(0),
                          1));

  CUDA_SYNC_CHECK();
}

void
OptiXBackend::DiffRenderBRDF()
{
  mColorBuffer = mColorBuffer.contiguous();
  mGTColorBuffer = mGTColorBuffer.contiguous();
  mColorBufferD = mColorBufferD.contiguous();
  mEnvMap = mEnvMap.contiguous();
  mEnvMapDist.condCdf = mEnvMapDist.condCdf.contiguous();
  mEnvMapDist.condFuncInt = mEnvMapDist.condFuncInt.contiguous();
  mEnvMapDist.func = mEnvMapDist.func.contiguous();
  mEnvMapDist.margCdf = mEnvMapDist.margCdf.contiguous();
  mMaterial = mMaterial.contiguous();
  mMaterialD = mMaterialD.contiguous();
  mNormals = mNormals.contiguous();
  mMask = mMask.contiguous();

  LaunchParams params = {};
  params.Normals = static_cast<float*>(mNormals.data_ptr());
  params.Mask = static_cast<std::int32_t*>(mMask.data_ptr());
  params.PointLight.Use = mPointLight.Use;
  params.PointLight.Phi = mPointLight.Phi;
  params.PointLight.Radiance = mPointLight.Radiance;
  params.FrameBuffer.ColorBuffer = static_cast<float*>(mColorBuffer.data_ptr());
  params.FrameBuffer.GTColorBuffer =
    static_cast<float*>(mGTColorBuffer.data_ptr());
  params.FrameBuffer.ColorBufferD =
    static_cast<float*>(mColorBufferD.data_ptr());
  params.FrameBuffer.Size =
    make_uint2(mColorBuffer.size(1), mColorBuffer.size(0));
  params.InvViewDir = mInvViewDir;
  params.EnvMap.Data = static_cast<float*>(mEnvMap.data_ptr());
  params.EnvMap.Size = make_uint2(mEnvMap.size(1), mEnvMap.size(0));
  params.EnvMapDist.CondCdf =
    static_cast<float*>(mEnvMapDist.condCdf.data_ptr());
  params.EnvMapDist.CondFuncInt =
    static_cast<float*>(mEnvMapDist.condFuncInt.data_ptr());
  params.EnvMapDist.Func = static_cast<float*>(mEnvMapDist.func.data_ptr());
  params.EnvMapDist.MargCdf =
    static_cast<float*>(mEnvMapDist.margCdf.data_ptr());
  params.EnvMapDist.MargFuncInt = mEnvMapDist.margFuncInt;
  params.Material = static_cast<float*>(mMaterial.data_ptr());
  params.MaterialD = static_cast<float*>(mMaterialD.data_ptr());

  if (mLaunchParams == 0) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mLaunchParams),
                          sizeof(LaunchParams)));
  }

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mLaunchParams),
                        &params,
                        sizeof(params),
                        cudaMemcpyHostToDevice));

  OPTIX_CHECK(optixLaunch(mOptixBackwardBRDFPipeline,
                          mCUDAStream,
                          mLaunchParams,
                          sizeof(LaunchParams),
                          &mBackwardBRDFSBT,
                          mColorBuffer.size(1),
                          mColorBuffer.size(0),
                          1));

  CUDA_SYNC_CHECK();
}

void
OptiXBackend::Cleanup()
{
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mLaunchParams)));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mGASOutputBuffer)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mAABB)));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mForwardSBT.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mForwardSBT.missRecordBase)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mForwardSBT.hitgroupRecordBase)));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mBackwardIlluSBT.raygenRecord)));
  CUDA_CHECK(
    cudaFree(reinterpret_cast<void*>(mBackwardIlluSBT.missRecordBase)));
  // CUDA_CHECK(
  //   cudaFree(reinterpret_cast<void*>(mBackwardIlluSBT.hitgroupRecordBase)));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mBackwardBRDFSBT.raygenRecord)));
  // CUDA_CHECK(
  //   cudaFree(reinterpret_cast<void*>(mBackwardBRDFSBT.missRecordBase)));
  // CUDA_CHECK(
  //   cudaFree(reinterpret_cast<void*>(mBackwardBRDFSBT.hitgroupRecordBase)));

  OPTIX_CHECK(optixPipelineDestroy(mOptixForwardPipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(mHitgroupRadianceProgGroup));
  OPTIX_CHECK(optixProgramGroupDestroy(mHitgroupOcclusionProgGroup));

  OPTIX_CHECK(optixProgramGroupDestroy(mForwardMissProgGroup));
  OPTIX_CHECK(optixProgramGroupDestroy(mBackwardMissProgGroup));

  OPTIX_CHECK(optixProgramGroupDestroy(mForwardRaygenProgGroup));
  OPTIX_CHECK(optixProgramGroupDestroy(mBackwardIlluRaygenProgGroup));
  OPTIX_CHECK(optixProgramGroupDestroy(mBackwardBRDFRaygenProgGroup));

  OPTIX_CHECK(optixModuleDestroy(mMissModule));
  OPTIX_CHECK(optixModuleDestroy(mRayGenModule));
  OPTIX_CHECK(optixModuleDestroy(mHitGroupModule));
  OPTIX_CHECK(optixDeviceContextDestroy(mOptixDeviceContext));
}
