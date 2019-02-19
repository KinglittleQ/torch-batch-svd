# Pytorch Batch SVD

## 1) Introduction

It is a batch version SVD of pytorch implemented by cuSolver including forward and backward function.

``` python
def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N]
    return:
        U, S, V = batch_svd(x) where x = USV^T
    """
```

The forward function is modified from [ShigekiKarita/pytorch-cusolver](https://github.com/ShigekiKarita/pytorch-cusolver) and I fixed several bugs of it. The backward function is adapted from pytorch official [svd backward function](https://github.com/pytorch/pytorch/blob/b0545aa85f7302be5b9baf8320398981365f003d/tools/autograd/templates/Functions.cpp#L1476). I converted it to a batch version.

NOTE: `batch_svd` only supports `CudaFloatTensor` now. Other types may be supported in the future.

## 2) Requirements

- Pytorch >= 1.0

    > diag_embed() is used in torch_batch_svd.cpp at the backward function. Pytorch with version lower than 1.0 does not contains diag_embed(). If you want to use it in lower version pytorch, you can replace diag_embed() by some existing function.

- CUDA 9.0

## 3) Install

### 1.1 Set environment variable

``` shell
export CUDA_HOME=/your/cuda/home/directory/
export LIBRARY_PATH=$LIBRARY_PATH:/your/cuda/lib64/  (optional)
```

### 1.2 Run `setup.py`

``` shell
python setup.py install
```

### 1.3 Run `test.py`

```shell
python test.py
```

## 4) Different between `torch.svd()`

- `batch_svd()` has no configurations of `some`, `compute_uv` like `torch.svd()`. `batch_svd(x)` is equivalent to `torch.svd(x, some=True, compute_uv=True)`.

- The sign of column vectors at U and V may be different from `torch.svd()`.

- `batch_svd()`is much more faster than `torch.svd()` using loop.

## 5) Example

See `test.py`.

