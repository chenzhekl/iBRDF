#include "util.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <tinyexr.h>
#include <torch/script.h>

#include "kernels/merl_constant.h"
#include "math.h"

namespace {
torch::Tensor
LastNonZero(const torch::Tensor& t)
{
  if (t.dim() != 3) {
    throw std::runtime_error("Input tensor must be of 3 dimension");
  }

  torch::Tensor res = torch::empty({ t.size(1), t.size(2) }, torch::kInt64);
  torch::TensorAccessor resAccessor = res.accessor<std::int64_t, 2>();
  torch::TensorAccessor tAccessor = t.accessor<std::int64_t, 3>();

  for (std::int64_t i = 0; i < tAccessor.size(1); ++i) {
    for (std::int64_t j = 0; j < tAccessor.size(2); ++j) {
      bool found = false;
      for (std::int64_t k = 0; k < tAccessor.size(0); ++k) {
        if (tAccessor[k][i][j] == 0) {
          resAccessor[i][j] = k - 1;
          found = true;
          break;
        }
      }
      if (!found) {
        resAccessor[i][j] = tAccessor.size(0) - 1;
      }
    }
  }

  return res;
}
}

std::string
LoadPTX(const std::string& path)
{
  std::ifstream file(path);
  std::stringstream sstr;
  sstr << file.rdbuf();
  return sstr.str();
}

torch::Tensor
LoadNormal(const std::string& path)
{
  torch::jit::script::Module container = torch::jit::load(path);
  torch::Tensor n = container.attr("n").toTensor();

  return n;
}

// Read BRDF data
torch::Tensor
LoadMERL(const std::string& path)
{
  FILE* f = fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error("MERL: Unable to open file");

  int dims[3];
  std::size_t readNumItems = fread(dims, sizeof(int), 3, f);
  if (readNumItems != 3) {
    fclose(f);
    throw std::runtime_error("MERL: corrupted file header");
  }
  int n = dims[0] * dims[1] * dims[2];
  if (n != kBRDFSamplingResThetaH * kBRDFSamplingResThetaD *
             kBRDFSamplingResPhiD / 2) {
    fclose(f);
    throw std::runtime_error("MERL: dimensions don't match");
  }

  double* brdf = (double*)malloc(sizeof(double) * 3 * n);
  readNumItems = fread(brdf, sizeof(double), 3 * n, f);
  if (readNumItems != 3 * static_cast<std::size_t>(n)) {
    fclose(f);
    throw std::runtime_error("MERL: corrupted file body");
  }
  fclose(f);

  torch::Tensor ret =
    torch::empty({ 3 * kBRDFSamplingResThetaH * kBRDFSamplingResThetaD *
                   kBRDFSamplingResPhiD / 2 },
                 torch::kFloat);
  torch::TensorAccessor retAccessor = ret.accessor<float, 1>();

  for (std::int64_t i = 0; i < 3 * n; ++i) {
    retAccessor[i] = *(brdf + i);
  }

  free(brdf);

  return ret.view({ 3,
                    kBRDFSamplingResThetaH,
                    kBRDFSamplingResThetaD,
                    kBRDFSamplingResPhiD / 2 });
}

void
SaveMERL(const std::string& path, const torch::Tensor& material)
{
  torch::Tensor m =
    material.detach().cpu().flatten().contiguous().to(torch::kDouble);

  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    throw std::runtime_error("MERL: Unable to open file");
  }

  int dims[3] = { kBRDFSamplingResThetaH,
                  kBRDFSamplingResThetaD,
                  kBRDFSamplingResPhiD / 2 };
  fwrite(dims, sizeof(int), 3, f);

  double* mPtr = static_cast<double*>(m.data_ptr());
  constexpr std::size_t kTotalMerlSamples = 3 * kBRDFSamplingResThetaH *
                                            kBRDFSamplingResThetaD *
                                            kBRDFSamplingResPhiD / 2;
  std::size_t writeItems = fwrite(mPtr, sizeof(double), kTotalMerlSamples, f);
  if (writeItems != kTotalMerlSamples) {
    fclose(f);
    throw std::runtime_error("MERL: Failed to write data");
  }

  fclose(f);
}

torch::Tensor
GenerateMERLSamples(ibrdf::IBRDF& model,
                    const torch::Tensor& positions,
                    const torch::Tensor& reference,
                    const std::optional<torch::Tensor>& embedCode,
                    bool unwarp)
{
  torch::Device device = model->parameters()[0].device();

  torch::Tensor maskSubset = torch::empty({ positions.size(0) }, torch::kInt64);
  torch::Tensor pos;

  {
    torch::NoGradGuard noGradGuard;

    torch::Tensor positionsCPU = positions.cpu();

    pos = positionsCPU.to(torch::kFloat32) /
            torch::tensor({ static_cast<float>(kBRDFSamplingResThetaH),
                            static_cast<float>(kBRDFSamplingResThetaD),
                            static_cast<float>(kBRDFSamplingResPhiD / 2) },
                          torch::kFloat32) +
          torch::tensor({ 0.5f / static_cast<float>(kBRDFSamplingResThetaH),
                          0.5f / static_cast<float>(kBRDFSamplingResThetaD),
                          0.5f / static_cast<float>(kBRDFSamplingResPhiD / 2) },
                        torch::kFloat32);

    torch::Tensor mask = (reference[0] > 0.0f).to(torch::kInt64).cpu();

    torch::TensorAccessor positionsAccessor =
      positionsCPU.accessor<std::int64_t, 2>();
    torch::TensorAccessor maskAccessor = mask.accessor<std::int64_t, 3>();
    torch::TensorAccessor maskSubsetAccessor =
      maskSubset.accessor<std::int64_t, 1>();

    for (std::int64_t i = 0; i < positionsAccessor.size(0); ++i) {
      maskSubsetAccessor[i] =
        maskAccessor[positionsAccessor[i][0]][positionsAccessor[i][1]]
                    [positionsAccessor[i][2]];
    }

    if (unwarp) {
      torch::Tensor bound = LastNonZero(mask);
      torch::Tensor scaleRatio = (bound.to(torch::kFloat32) + 1.0f) /
                                 static_cast<float>(kBRDFSamplingResThetaH);
      torch::Tensor scaleRatioSubset =
        torch::empty({ positions.size(0) }, torch::kFloat32);

      torch::TensorAccessor scaleRatioAccessor =
        scaleRatio.accessor<float, 2>();
      torch::TensorAccessor scaleRatioSubsetAccessor =
        scaleRatioSubset.accessor<float, 1>();

      for (std::int64_t i = 0; i < positionsAccessor.size(0); ++i) {
        scaleRatioSubsetAccessor[i] =
          scaleRatioAccessor[positionsAccessor[i][1]][positionsAccessor[i][2]];
      }

      pos.select(-1, 0) = pos.select(-1, 0) / scaleRatioSubset;
      pos.masked_fill_(pos >= 1.0f, 0.0f);
    }
  }

  pos = pos.to(device);

  return model->logPdf(pos, embedCode).exp().squeeze(-1) *
         maskSubset.to(device);
}

torch::Tensor
GenerateMERLSlice(ibrdf::IBRDF& model,
                  const torch::Tensor& reference,
                  const std::optional<torch::Tensor>& embedCode,
                  bool unwarp)
{
  torch::Device device = model->parameters()[0].device();
  torch::Tensor grid;
  torch::Tensor mask;

  {
    torch::NoGradGuard noGradGuard;

    mask = (reference[0] > 0.0f).to(torch::kInt64);

    torch::Tensor thetaH = torch::linspace(0.5f / kBRDFSamplingResThetaH,
                                           1.0f - 0.5f / kBRDFSamplingResThetaH,
                                           kBRDFSamplingResThetaH);
    torch::Tensor thetaD = torch::linspace(0.5f / kBRDFSamplingResThetaD,
                                           1.0f - 0.5f / kBRDFSamplingResThetaD,
                                           kBRDFSamplingResThetaD);
    torch::Tensor phiD = torch::linspace(1.0f / kBRDFSamplingResPhiD,
                                         1.0f - 1.0f / kBRDFSamplingResPhiD,
                                         kBRDFSamplingResPhiD / 2);

    std::vector<torch::Tensor> axisTuple =
      torch::meshgrid({ thetaH, thetaD, phiD });
    grid = torch::stack({ axisTuple[0].flatten(),
                          axisTuple[1].flatten(),
                          axisTuple[2].flatten() })
             .t()
             .view({ kBRDFSamplingResThetaH,
                     kBRDFSamplingResThetaD,
                     kBRDFSamplingResPhiD / 2,
                     3 });
    if (unwarp) {
      torch::Tensor bound = LastNonZero(mask.detach().cpu());
      torch::Tensor scaleRatio = (bound.to(torch::kFloat32) + 1.0f) /
                                 static_cast<float>(kBRDFSamplingResThetaH);

      grid.select(-1, 0) = grid.select(-1, 0) / scaleRatio;
      grid.masked_fill_(grid >= 1.0f, 0.0f);
    }
  }

  grid = grid.view({ -1, 3 }).to(device);

  return model->logPdf(grid, embedCode)
           .exp()
           .view({
             kBRDFSamplingResThetaH,
             kBRDFSamplingResThetaD,
             kBRDFSamplingResPhiD / 2,
           }) *
         mask.to(device);
}

torch::Tensor
GenerateMERL(ibrdf::IBRDF& model,
             const torch::Tensor& reference,
             const torch::Tensor& embedCode,
             const torch::Tensor& color,
             bool unwarp)
{
  torch::Device device = model->parameters()[0].device();
  std::int64_t numLobes = embedCode.size(0);

  torch::Tensor material =
    torch::zeros({ 3, 90 * 90 * 180 }, torch::kFloat32).to(device);

  for (std::int64_t lobeIdx = 0; lobeIdx < numLobes; ++lobeIdx) {
    torch::Tensor lobe =
      (GenerateMERLSlice(model,
                         reference,
                         embedCode[lobeIdx].repeat({ 90 * 90 * 180, 1 }),
                         unwarp)
         .view({ 1, -1 }) *
       color.select(1, lobeIdx).unsqueeze(-1))
        .expm1();

    material += lobe;
  }

  return material;
}

torch::Tensor
LoadEXR(const std::string& filename)
{
  float* out; // width * height * RGBA
  int width;
  int height;
  const char* err = NULL; // or nullptr in C++11

  int ret = LoadEXR(&out, &width, &height, filename.c_str(), &err);

  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      fprintf(stderr, "ERR : %s\n", err);
      FreeEXRErrorMessage(err); // release memory of error message.
    }
  }

  torch::Tensor img = torch::empty(
    { static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3 },
    torch::kFloat);
  torch::TensorAccessor imgAccessor = img.accessor<float, 3>();

  for (std::int64_t row = 0; row < height; ++row) {
    for (std::int64_t col = 0; col < width; ++col) {
      std::int64_t i = row * width + col;
      imgAccessor[row][col][0] = out[i * 4 + 0];
      imgAccessor[row][col][1] = out[i * 4 + 1];
      imgAccessor[row][col][2] = out[i * 4 + 2];
    }
  }

  free(out); // relase memory of image data

  return img.view(
    { static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3 });
}

bool
SaveEXR(const std::string& filename,
        const torch::Tensor& img,
        std::int64_t width,
        std::int64_t height)
{

  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 3;

  std::vector<float> images[3];
  images[0].resize(width * height);
  images[1].resize(width * height);
  images[2].resize(width * height);

  torch::Tensor imgCPU = img.cpu();
  torch::TensorAccessor imgAccessor = imgCPU.accessor<float, 3>();

  // Split RGBRGBRGB... into R, G and B layer
  for (std::int64_t row = 0; row < height; ++row) {
    for (std::int64_t col = 0; col < width; ++col) {
      std::int64_t i = row * width + col;
      images[0][i] = imgAccessor[row][col][0];
      images[1][i] = imgAccessor[row][col][1];
      images[2][i] = imgAccessor[row][col][2];
    }
  }

  float* image_ptr[3];
  image_ptr[0] = &(images[2].at(0)); // B
  image_ptr[1] = &(images[1].at(0)); // G
  image_ptr[2] = &(images[0].at(0)); // R

  image.images = (unsigned char**)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = 3;
  header.channels =
    (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  std::strncpy(header.channels[0].name, "B", 255);
  header.channels[0].name[strlen("B")] = '\0';
  std::strncpy(header.channels[1].name, "G", 255);
  header.channels[1].name[strlen("G")] = '\0';
  std::strncpy(header.channels[2].name, "R", 255);
  header.channels[2].name[strlen("R")] = '\0';

  header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types =
    (int*)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] =
      TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    header.requested_pixel_types[i] =
      TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
  }

  const char* err = NULL; // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err); // free's buffer for an error message
    return false;
  }

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  return true;
}

void
CreateDist1D(const torch::Tensor& f, torch::Tensor& cdf, float& funcInt)
{
  torch::NoGradGuard noGradGuard;

  torch::Tensor tmp = f / f.size(0);
  cdf = torch::cat({ torch::zeros({ 1 }, tmp.options()), tmp }, 0);
  cdf = cdf.cumsum(0);
  funcInt = cdf[-1].item<float>();

  if (funcInt == 0.0f) {
    cdf /= f.size(0);
  } else {
    cdf /= funcInt;
  }
}

void
CreateDist2D(const torch::Tensor& f,
             torch::Tensor& condCdf,
             torch::Tensor& condFuncInt,
             torch::Tensor& margCdf,
             float& margFuncInt)
{
  std::vector<float> funcInts(f.size(0));
  std::vector<torch::Tensor> cdfs(f.size(0));

  for (std::int64_t row = 0; row < f.size(0); ++row) {
    CreateDist1D(f[row], cdfs[row], funcInts[row]);
  }

  condCdf = torch::stack(cdfs, 0);
  condFuncInt = torch::tensor(funcInts, f.options());

  CreateDist1D(condFuncInt, margCdf, margFuncInt);
}

torch::Tensor
ACESToneMapping(const torch::Tensor& hdr, float adapted_lum)
{
  constexpr float A = 2.51f;
  constexpr float B = 0.03f;
  constexpr float C = 2.43f;
  constexpr float D = 0.59f;
  constexpr float E = 0.14f;

  torch::Tensor sdr = hdr * adapted_lum;
  return (sdr * (A * sdr + B)) / (sdr * (C * sdr + D) + E);
}

void
CreateEnvMapSamplingDist(const torch::Tensor& envMap,
                         torch::Tensor& func,
                         torch::Tensor& condCdf,
                         torch::Tensor& condFuncInt,
                         torch::Tensor& margCdf,
                         float& margFuncInt)
{
  torch::NoGradGuard noGradGuard;

  std::int64_t height = envMap.size(0);
  std::int64_t width = envMap.size(1);

  torch::Tensor toneMapped = envMap;

  torch::Tensor luminanceWeight =
    torch::tensor({ 0.2126, 0.7152, 0.0722 }, envMap.options());
  torch::Tensor luminance =
    toneMapped.view({ -1, 3 }).matmul(luminanceWeight).view({ height, width });

  torch::Tensor sinTheta = ((torch::arange(height, envMap.options()) + 0.5f) /
                            static_cast<float>(height) * kPI)
                             .sin();
  luminance *= sinTheta.unsqueeze(-1);
  func = luminance;

  CreateDist2D(luminance, condCdf, condFuncInt, margCdf, margFuncInt);
}
