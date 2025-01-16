<div align="center">
<h1>Learning <i>Programming Massively Parallel Processors</i></h1>
<img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-blue?style=plastic&logo=cplusplus&logoColor=blue"> <img alt="CUDA-12" src="https://img.shields.io/badge/CUDA-12-green?style=plastic&logo=nvidia"> <img alt="Static Badge" src="https://img.shields.io/badge/python-3-blue?style=plastic&logo=python&logoColor=blue"> <img alt="Static Badge" src="https://img.shields.io/badge/pytorch-2-orange?style=plastic&logo=pytorch">
</div>

## Quick Start

Create a new conda environment:

```bash
conda create -n cuda-learn python=3.12
conda activate cuda-learn
pip install torch torchvision torchaudio
```

To build the C++ part only (lib pmpp):

```bash
bash scripts/build.sh
```

Run ctest to test lib pmpp:

```bash
ctest --test-dir ./build --output-on-failure
```

To build and instll the corresponding python lib:

```bash
pip3 install --no-build-isolation -v .
```

`torch.ops.pmpp.vector_add` will be available after installation;  
See [test.py](test/test.py) for usage.
