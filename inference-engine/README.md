## Repository components

The Inference Engine can infer models in different formats with various input and output formats.

The open source version of Inference Engine includes the following plugins:

| PLUGIN               | DEVICE TYPES |
| ---------------------| -------------|
| CPU plugin           | Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
| GPU plugin           | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
| GNA plugin           | Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel® Pentium® Silver processor J5005, Intel® Celeron® processor J4005, Intel® Core™ i3-8121U processor |
| Heterogeneous plugin | Heterogeneous plugin enables computing for inference on one network on several Intel® devices. |

Inference Engine plugins for Intel® FPGA and Intel® Movidius™ Neural Compute Stick are distributed only in a binary form as a part of [Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/openvino-toolkit).

## Build on Linux\* Systems

The software was validated on:
- Ubuntu\* 16.04 with default GCC\* 5.4.0
- CentOS\* 7.4 with default GCC\* 4.8.5
- [Intel® Graphics Compute Runtime for OpenCL™ Driver package 18.28.11080](https://github.com/intel/compute-runtime/releases/tag/18.28.11080).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- GCC\* 4.8 or higher to build the Inference Engine
- Python 2.7 or higher for Inference Engine Python API wrapper

### Build Steps
1. Clone submodules:
    ```sh
    cd dldt/inference-engine
    git submodule init
    git submodule update --recursive
    ```
2. Install build dependencies using the `install_dependencies.sh` script in the project root folder.
3. Create a build folder:
```sh
  mkdir build
```
4. Inference Engine uses a CMake-based build system. In the created `build` directory, run `cmake` to fetch project dependencies and create Unix makefiles, then run `make` to build the project:
```sh
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j16
```
You can use the following additional build options:
- Internal JIT GEMM implementation is used by default.
- To switch to OpenBLAS\* implementation, use `GEMM=OPENBLAS` option and `BLAS_INCLUDE_DIRS` and `BLAS_LIBRARIES` cmake options to specify path to OpenBLAS headers and library, for example use the following options on CentOS\*: `-DGEMM=OPENBLAS -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`
- To switch to optimized MKL-ML\* GEMM implementation, use `GEMM=MKL` and `MKLROOT` cmake options to specify path to unpacked MKL-ML with `include` and `lib` folders, for example use the following options: `-DGEMM=MKL -DMKLROOT=<path_to_MKL>`. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_lnx_2019.0.1.20180928.tgz)

- OpenMP threading is used by default. To build Inference Engine with TBB threading, set `-DTHREADING=TBB` option.

- To build Python API wrapper, use -DENABLE_PYTHON=ON option. To specify exact Python version, use the following options: `-DPYTHON_EXECUTABLE=`which python3.6` -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.6`

- To switch on/off the CPU and GPU plugins, use `cmake` options `-DENABLE_MKL_DNN=ON/OFF` and `-DENABLE_CLDNN=ON/OFF`.

5. Adding to your project

    For CMake projects, set an environment variable `InferenceEngine_DIR`:

    ```sh
    export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/
    ```

    Then you can find Inference Engine by `find_package`:

    ```cmake
    find_package(InferenceEngine)

    include_directories(${InferenceEngine_INCLUDE_DIRS})

    target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)
    ```

## Build on Windows\* Systems:

The software was validated on:
- Microsoft\* Windows\* 10 with Visual Studio 2017 and Intel® C++ Compiler 2018 Update 3
- [Intel® Graphics Driver for Windows* [24.20] driver package](https://downloadcenter.intel.com/download/27803/Graphics-Intel-Graphics-Driver-for-Windows-10?v=t).

### Software Requirements
- [CMake\*](https://cmake.org/download/) 3.9 or higher
- [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download) and [mingw64\* runtime dependencies](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download).
- [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0 to build the Inference Engine on Windows.
- Python 3.4 or higher for Inference Engine Python API wrapper

### Build Steps
1. Clone submodules:
    ```sh
    git submodule init
    git submodule update --recursive
    ```
2. Download and install [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 18.0
3. Install OpenBLAS:
    1. Download [OpenBLAS\*](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download)
    2. Unzip the downloaded package to a directory on your machine. In this document, this directory is referred to as `<OPENBLAS_DIR>`.
4. Create build directory:
    ```sh
    mkdir build
    ```
5. In the `build` directory, run `cmake` to fetch project dependencies and generate a Visual Studio solution:
```sh
cd build
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 18.0" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DICCLIB="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\compiler\lib" ..
```

- Internal JIT GEMM implementation is used by default.
- To switch to OpenBLAS GEMM implementation, use -DGEMM=OPENBLAS cmake option and specify path to OpenBLAS using `-DBLAS_INCLUDE_DIRS=<OPENBLAS_DIR>\include` and `-DBLAS_LIBRARIES=<OPENBLAS_DIR>\lib\libopenblas.dll.a` options. Prebuilt OpenBLAS\* package can be downloaded [here](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int64.zip/download), mingw64* runtime dependencies [here](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download)
- To switch to optimized MKL-ML GEMM implementation, use `GEMM=MKL` and `MKLROOT` cmake options to specify path to unpacked MKL-ML with `include` and `lib` folders, for example use the following options: `-DGEMM=MKL -DMKLROOT=<path_to_MKL>`. MKL-ML\* package can be downloaded [here](https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_win_2019.0.1.20180928.zip)

- OpenMP threading is used by default. To build Inference Engine with TBB threading, set `-DTHREADING=TBB` option.

- To build Python API wrapper, use -DENABLE_PYTHON=ON option. To specify exact Python version, use the following options: `-DPYTHON_EXECUTABLE="C:\Program Files\Python36\python.exe" -DPYTHON_INCLUDE_DIR="C:\Program Files\Python36\include" -DPYTHON_LIBRARY="C:\Program Files\Python36\libs\python36.lib"`.

6. Build generated solution in Visual Studio 2017 or run `cmake --build . --config Release` to build from the command line.

### Building Inference Engine with Ninja

```sh
call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\bin\ipsxe-comp-vars.bat" intel64 vs2017
set CXX=icl
set CC=icl
cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Before running the samples on Microsoft\* Windows\*, please add path to OpenMP library (<dldt_repo>/inference-engine/temp/omp/lib) and OpenCV libraries (<dldt_repo>/inference-engine/temp/opencv_4.0.0/bin) to the %PATH% environment variable.

---
\* Other names and brands may be claimed as the property of others.
