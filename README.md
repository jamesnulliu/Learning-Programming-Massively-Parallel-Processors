# VSC-Python-Project-Template
A Template of Python Project in Visual Studio Code with Github Actions CI/CD (Especially for PyTorch based Deeplearning Projects).

## How to Build

Create a new conda environment:

```bash
conda create -n pytemplate python=3.12
conda activate pytemplate
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