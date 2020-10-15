# iBRDF

This is the code for the paper "Invertible Neural BRDF for Object Inverse Rendering".

## Prerequesties

1. A C++ 17-compatible compiler
2. CUDA 10.2
3. CMake
4. LibTorch 1.6.0
5. OptiX 7.0.0
6. TinyEXR (bundled)

## Usage

A Dockerfile is provided to help setup the environment. The following commands are assumed to be run inside the Docker container, though it is not a necessity.

### Building

Download and extract LibTorch and OptiX into `./third_party/libtorch` and `./third_party/optix` respectively. Then run the following commands to build the project:

```bash
mkdir build
cd build
cmake ..
make
```

### Training iBRDF

Assuming that MERL BRDF binaries are placed inside the `../datasets/merl` folder. Run the following command to pre-process them into a format that is suitable for training:

```bash
python ./scripts/preprocess_merl.py ../datasets/merl ../datasets/merl_processed
```

The resulting BRDFs will be placed under `../datasets/merl_processed`. After conversion, run the following command to train iBRDF:

```bash
Usage: ./build/bin/ibrdf_train [MERL root] [Number of BRDFs per batch] [Number of samples per BRDF] [Number of epochs] [Output]

Example: ./build/bin/ibrdf_train ../datasets/merl_processed 50 10000 10000 ./run/ibrdf
```

### Material Estimation

```bash
Usage: ./build/bin/estbrdf [Input] [Geometry] [Illumination] [iBRDF model] [Number of lobes] [Number of optimization steps] [Output]

Example: ./build/bin/estbrdf ./run/render.exr ./data/sphere.pt ./data/uffizi-large.exr ./data/ibrdf.pt 2 200 ./run/brdf.binary
```

### Illumination Estimation

```bash
Usage: ./build/bin/estillu [Input] [Geometry] [Material] [Mirror reflection] [Illumination width] [Illumination height] [Output]

Example: ./build/bin/estillu ./run/a.exr ./data/sphere.pt ./data/alum-bronze.binary 0 512 256 2000 ./run/illu.exr
```

### Joint Estimation of Illumination and Material

```bash
Usage: ./build/bin/estboth [Input] [Geometry] [iBRDF model] [Number of lobes] [Illumination width] [Illumination height] [Number of optimization steps] [Number of material optimization steps] [Number of illumination optimization steps] [Number of gray world steps] [Output material] [Output illumination]

Example: ./build/bin/estboth ./run/render.exr ./data/sphere.pt ./data/ibrdf.pt 2 256 128 10 100 300 3 ./run/brdf.binary ./run/illu.exr
```

### Rendering

```bash
Usage: ./build/bin/render [Geometry] [Material] [Illumination] [Mirror reflection] [Output]

Example: ./build/bin/render ./data/sphere.pt ./data/alum-bronze.binary ./data/uffizi-large.exr 0 ./run/render.exr
```

## Citation

    @INPROCEEDINGS {Chen_2020_ECCV,
        author    = "Chen, Zhe and Nobuhara, Shohei and Nishino, Ko",
        title     = "Invertible Neural BRDF for Object Inverse Rendering",
        booktitle = "Proceedings of the European Conference on Computer Vision (ECCV)",
        year      = "2020"
    }

## LICENSE

[MIT](LICENSE) Liencse

