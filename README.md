# LEARN *Programming Massively Parallel Processors*

## How to Build

Create a new conda environment:

```bash
conda create -n cuda-learn python=3.12
conda activate cuda-learn
```

Install `example_package`:

```bash
pip3 install --no-build-isolation .
```

`torch.ops.example_package.vector_add` will be available after installation; See [test.py](test/test.py) for usage.

## How to Test

```bash
conda activate pytemplate
python test/test.py
```