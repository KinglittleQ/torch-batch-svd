import time
import torch
from torch_batch_svd import svd


def bench_speed(N, H, W):
    torch.manual_seed(0)
    a = torch.randn(N, H, W).cuda()
    b = a.clone().cuda()
    a.requires_grad = True
    b.requires_grad = True

    t0 = time.time()
    U, S, V = svd(a)
    t1 = time.time()
    print("Perform batched SVD on a {}x{}x{} matrix: {} s".format(N, H, W, t1 - t0))

    t0 = time.time()
    U, S, V = torch.svd(b, some=True, compute_uv=True)
    t1 = time.time()
    print("Perform torch.svd on a {}x{}x{} matrix: {} s".format(N, H, W, t1 - t0))


if __name__ == '__main__':
    bench_speed(10000, 9, 9)
    bench_speed(20000, 9, 9)

