#pragma once
#include <torch/extension.h>
#include <vector>

void batch_svd_forward(at::Tensor a, at::Tensor U, at::Tensor s, at::Tensor V,
                       bool is_sort, double tol = 1e-7, int max_sweeps = 100,
                       bool is_double = false);
at::Tensor batch_svd_backward(const std::vector<at::Tensor> &grads,
                              const at::Tensor &self, bool some,
                              bool compute_uv, const at::Tensor &raw_u,
                              const at::Tensor &sigma, const at::Tensor &raw_v);
