# Pytorch Batched SVD

## 1) Introduction

A batched version of SVD in Pytorch implemented using cuSolver 
including forward and backward function.
In terms of speed, it is superior to that of `torch.svd`.

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
 
The forward function is modified from [ShigekiKarita/pytorch-cusolver](https://github.com/ShigekiKarita/pytorch-cusolver) and I fixed several bugs of it. The backward function is borrowed from the pytorch official [svd backward function](https://github.com/pytorch/pytorch/blob/b0545aa85f7302be5b9baf8320398981365f003d/tools/autograd/templates/Functions.cpp#L1476). I converted it to a batched version.

**NOTE**: `batch_svd` supports all `torch.half`, `torch.float` and `torch.double` tensors now. 

**NOTE**: SVD for `torch.half` is performed by casting to `torch.float` 
as there is no CuSolver implementation for `c10::half`.   

**NOTE**: Sometimes, tests will fail for `torch.double` tensor due to numerical imprecision.

## 2) Requirements

- Pytorch >= 1.0

  `diag_embed()` is used in torch_batch_svd.cpp at the backward function. Pytorch with version lower than 1.0 does not contain `diag_embed()`. If you want to use it in a lower version of pytorch, you can replace `diag_embed()` by some existing functions.

- CUDA 9.0/10.2 (should work with 10.0/10.1 too)

- Tested in Pytorch 1.4 & 1.5, with CUDA 10.2

## 3) Install

Set environment variables

``` shell
export CUDA_HOME=/your/cuda/home/directory/
export LIBRARY_PATH=$LIBRARY_PATH:/your/cuda/lib64/  (optional)
```

Run `setup.py`

``` shell
python setup.py install
```

Run `test.py`

```shell
cd tests
python -m pytest test.py
```

## 4) Differences between `torch.svd()`

- The sign of column vectors at U and V may be different from `torch.svd()`.

- Much more faster than `torch.svd()` using loop.

## 5) Example

See `test.py` and [introduction](#1-introduction).
