#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                     \
  } while (0)

#define CHECK_IS_FLOAT(x)                                                      \
  do {                                                                         \
    TORCH_CHECK((x.scalar_type() == at::ScalarType::Float) ||                  \
                    (x.scalar_type() == at::ScalarType::Half) ||               \
                    (x.scalar_type() == at::ScalarType::Double),               \
                #x " must be a double, float or half tensor");                 \
  } while (0)

template <int success = CUSOLVER_STATUS_SUCCESS, class T,
          class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
std::unique_ptr<T, Status (*)(T *)> unique_allocate(Status(allocator)(T **),
                                                    Status(deleter)(T *)) {
  T *ptr;
  auto stat = allocator(&ptr);
  TORCH_CHECK(stat == success);
  return {ptr, deleter};
}

template <class T>
std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
  T *ptr;
  auto stat = cudaMalloc(&ptr, sizeof(T) * len);
  TORCH_CHECK(stat == cudaSuccess);
  return {ptr, cudaFree};
}
