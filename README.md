<div align="center">
<h1>Learning <i>Programming Massively Parallel Processors</i></h1>
<img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-blue?style=plastic&logo=cplusplus&logoColor=blue"> <img alt="CUDA-12" src="https://img.shields.io/badge/CUDA-12-green?style=plastic&logo=nvidia"> <img alt="Static Badge" src="https://img.shields.io/badge/python-3-blue?style=plastic&logo=python&logoColor=blue"> <img alt="Static Badge" src="https://img.shields.io/badge/pytorch-2-orange?style=plastic&logo=pytorch">
</div>

## 1. Environment

### 1.1. Method 1: Use Docker Image

The simplest way is to use my docker image [jamesnulliu/deeplearning:latest](https://hub.docker.com/r/jamesnulliu/deeplearning) which contains all the softwares you need to build the project:

```bash
docker pull jamesnulliu/deeplearning:latest
```

> Check my blog: [Docker Container with Nvidia GPU Support](https://jamesnulliu.github.io/blogs/docker-container-with-nvidia-gpu-support) if you need any help.

### 1.2. Method 2: Setup Environment Manually

Or if you are planing to setup your own environment, here are some tips:

You should install all the softwares with corresponding versions listed bellow:

- Miniconda/Anaconda
- gcc >= 12.0, nvcc >= 12.0
- CMake >= 3.30
- Ninja
- vcpkg, pkg-config
- [managed by conda] python >= 3.10, pytorch >= 2.0
- [managed by vcpkg] cxxopts, fmt, spdlog, proxy, gtest, yaml-cpp

**🎯Miniconda**

Managing python environments with miniconda is always a good choice. Check [the official website](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) for an installation guide.

After installation, if you do not intend to install all the packages in `base` environment, create a new conda environment named `PMPP` (or whatever you like) and activate it:

```bash {linenos=true}
# python version should be larger than 3.10
conda create -n PMPP python=3.12
conda activate PMPP  # Activate this environment
# In my experience, when your system gcc version is larger than 12, it is
# highly possible that you have to update libstd++ in conda for running the
# later compiled targets. All you need to do is to run this command:
conda upgrade libstdcxx-ng -c conda-forge
```

**🎯PyTorch**

Install pytorch **with pip (not conda)** in environment `PMPP` following the steps on [the official website](https://pytorch.org/get-started/locally/#start-locally). In my case I installed `torch-2.6.0 + cuda 12.6`.

> 📝**NOTE**  
> All the python packages you installed can be found under the directory of `$CONDA_PREFIX/lib/python3.12/site-packages`.

**🎯CUDA**

To compile cuda code, you need to install **cuda toolkit** on your system. Usually, even if `torch-2.6.0 + cuda 12.6` is installed in your conda environment while `cuda 12.1` is installed on the system, you can run torch in python without any mistakes. But in some cases, you still have to install `cuda 12.6` to exactly match the torch you chose.

You can find all versions of cuda on [the official website](https://developer.nvidia.com/cuda-toolkit-archive).

> 📝**NOTE**  
> Installing and using multiple versions of cuda is possible by managing the `PATH` and `LD_LIBRARY_PATH` environment variables on linux, and you can do this manually or refering to my methods in [this blog](/blogs/environment-variable-management).

## 2. Quick Start

To build the C++ part only:

```bash
bash scripts/build.sh
```

> 📝**NOTE**  
> See "[cmake-parameters.md](csrc/cmake/cmake-parameters.md)" for details about setting up the build process.

You will find "./build/lib/libPmppTorchOps.so" which is the operator library and "./build/test/pmpp_test" which is the test executable (with gtest).

Execute the test executable to test the library manually:

```bash
./build/test/pmpp_test
```

Note that the test is already integrated into CMake build system (with ctest); In "[scripts/build.sh](scripts/build.sh)", the last line shows how to run the test:

```bash
# $BUILD_DIR is "./build" by default
# Set `GTEST_COLOR` to yes or no to enable or disable colored output

# If the library has not been build, target `all` before `check` is required
cmake --build $BUILD_DIR -j $(nproc) --target all check
# Or if the library has been build, `check` is enough
cmake --build $BUILD_DIR -j $(nproc) --target check
```

To build and install the python package `pmpp` in current activated conda environment (pmpp operator library would be built automatically if it has not been built yet):

```bash
pip3 install --no-build-isolation -v .
```

`torch.ops.pmpp.vector_add` will be available after installation; See [test.py](test/test.py) for example.
