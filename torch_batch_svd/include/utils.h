#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK((x.scalar_type() == at::ScalarType::Float) || (x.scalar_type() == at::ScalarType::Half), \
             #x " must be a float or half tensor");            \
  } while (0)
