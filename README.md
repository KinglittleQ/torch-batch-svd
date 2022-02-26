# Pytorch Batched SVD

## Introduction

A 100x faster SVD for PyTorch including forward and backward function.

Performance:

| matrix size      | torch_batch_svd.svd  | torch.svd  |
| ---------------  |:--------------------:| :---------:|
| `(10000, 9, 9)`  | **0.043** s          | 19.352 s   |
| `(20000, 9, 9)`  | **0.073** s          | 34.578 s   |


``` python
import torch
from torch_batch_svd import svd

A = torch.rand(1000000, 3, 3).cuda()
u, s, v = svd(A)
u, s, v = torch.svd(A)  # probably you should take a coffee break here
```

The catch here is that it only works for matrices whose row and column are smaller than `32`.
Other than that, `torch_batch_svd.svd` can be a drop-in for the native one.

The forward function is modified from [ShigekiKarita/pytorch-cusolver](https://github.com/ShigekiKarita/pytorch-cusolver) and I fixed several bugs of it. The backward function is borrowed from the PyTorch official [svd backward function](https://github.com/pytorch/pytorch/blob/b0545aa85f7302be5b9baf8320398981365f003d/tools/autograd/templates/Functions.cpp#L1476). I converted it to a batched version.

**NOTE**: `batch_svd` supports all `torch.half`, `torch.float` and `torch.double` tensors now.

**NOTE**: SVD for `torch.half` is performed by casting to `torch.float`
as there is no CuSolver implementation for `c10::half`.

**NOTE**: Sometimes, tests will fail for `torch.double` tensor due to numerical imprecision.

## Get Started

### Requirements

- Pytorch >= 1.0

- CUDA 9.0/10.2 (should work with 10.0/10.1 too)

- Tested in Pytorch 1.4 & 1.5, with CUDA 10.2

### Install

``` shell
export CUDA_HOME=/your/cuda/home/directory/
python setup.py install
```

### Test

```shell
python tests/test.py
```

## Differences between `torch.svd()`

- The sign of column vectors at U and V may be different from `torch.svd()`.

- Much more faster than `torch.svd()` using loop.

## Example

See `test.py` and [introduction](#1-introduction).
