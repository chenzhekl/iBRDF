# ========================
# Kernels
# ========================

# Hitgroup
add_library(
  hitgroup OBJECT
  kernels/hitgroup.cu
)
set_property(TARGET hitgroup PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(
  hitgroup
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE ${CMAKE_SOURCE_DIR}/third_party/optix/include
)

# Raygen
add_library(
  raygen OBJECT
  kernels/raygen.cu
)
set_property(TARGET raygen PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(
  raygen
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE ${CMAKE_SOURCE_DIR}/third_party/optix/include
)

# Miss
add_library(
  miss OBJECT
  kernels/miss.cu
)
set_property(TARGET miss PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(
  miss
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE ${CMAKE_SOURCE_DIR}/third_party/optix/include
)

# ====================
# Main library
# ====================
add_library(
  ranix
  util.h util.cpp
  sbtrecord.h
  math.h
  backend.h backend.cpp
  tinyexr.cpp
  nn/concat.h nn/concat.cpp
  nn/dcgan.h nn/dcgan.cpp
  nn/skip.h nn/skip.cpp
  nn/stack_sequential.h nn/stack_sequential.cpp
  nn/downsampler.h nn/downsampler.cpp
  nn/lap_pyramid.h
  priors.h priors.cpp
  ssim.h ssim.cpp
)

target_include_directories(
  ranix
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PUBLIC ${CMAKE_SOURCE_DIR}/third_party/optix/include
  PUBLIC ${CMAKE_SOURCE_DIR}/third_party/single_header
)

target_link_libraries(
  ranix
  PUBLIC ibrdf
  PUBLIC ${CUDA_LIBRARIES}
  PUBLIC ${TORCH_LIBRARIES}
  PUBLIC dl
  PUBLIC OpenMP::OpenMP_CXX
)

# =====================
# Binaries
# =====================
add_executable(
  render
  bins/render.cpp
)

target_link_libraries(
  render
  PRIVATE ranix
)

add_executable(
  estillu
  bins/estillu.cpp
)

target_link_libraries(
  estillu
  PRIVATE ranix
)

add_executable(
  estbrdf
  bins/estbrdf.cpp
)

target_link_libraries(
  estbrdf
  PRIVATE ranix
  PRIVATE ibrdf
)

add_executable(
  estboth
  bins/estboth.cpp
)

target_link_libraries(
  estboth
  PRIVATE ranix
  PRIVATE ibrdf
)

add_executable(
  fit_merl
  bins/fit_merl.cpp
)

target_link_libraries(
  fit_merl
  PRIVATE ranix
  PRIVATE ibrdf
)

# =====================
# iBRDF
# =====================
add_subdirectory(ibrdf)
