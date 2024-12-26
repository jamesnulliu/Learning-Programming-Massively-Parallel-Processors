# Learning *Programming Massively Parallel Processors*

## Quick Start

Create a new conda environment:

```bash
conda create -n cuda-learn python=3.12
conda activate cuda-learn
pip install torch torchvision torchaudio
```

To build the C++ part only (lib pmpp):

```bash
bash scripts/build.sh -S ./csrc -B ./build
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
