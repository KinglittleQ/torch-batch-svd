#include "torch_batch_svd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_svd_forward", &batch_svd_forward,
        "cusolver based batch svd forward");
  m.def("batch_svd_backward", &batch_svd_backward, "batch svd backward");
}
