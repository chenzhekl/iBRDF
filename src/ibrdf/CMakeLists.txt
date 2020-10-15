add_library(
  ibrdf
  ibrdf.cpp ibrdf.h
  unet.cpp unet.h
  util.cpp util.h
  piecewise_quadratic.cpp piecewise_quadratic.h
  coupling_layer.cpp coupling_layer.h
  datasets/merl.cpp datasets/merl.h
  distributions/piecewise.cpp distributions/piecewise.h
  merl_materials.h merl_materials.cpp
)

target_include_directories(
  ibrdf
  PUBLIC ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(ibrdf ${TORCH_LIBRARIES} stdc++fs OpenMP::OpenMP_CXX)

# ==============================
# iBRDF train
# ==============================

add_executable(
  ibrdf_train
  bin/train.cpp
)

target_link_libraries(ibrdf_train ibrdf)
