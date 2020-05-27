#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> batch_svd_forward(at::Tensor a, bool is_sort, double tol=1e-7, int max_sweeps=100);
at::Tensor batch_svd_backward(const std::vector<at::Tensor> &grads, const at::Tensor& self,
          bool some, bool compute_uv, const at::Tensor& raw_u, const at::Tensor& sigma, const at::Tensor& raw_v);
