import torch
from torch_batch_svd import svd


def bench_speed(N, H, W):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.manual_seed(0)
    a = torch.randn(N, H, W).cuda()
    b = a.clone().cuda()
    torch.cuda.synchronize()

    start.record()
    for i in range(100):
        U, S, V = svd(a)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 100
    print("Perform batched SVD on a {}x{}x{} matrix: {} ms".format(N, H, W, t))

    start.record()
    U, S, V = torch.svd(b, some=True, compute_uv=True)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end)
    print("Perform torch.svd on a {}x{}x{} matrix: {} ms".format(N, H, W, t))


if __name__ == "__main__":
    bench_speed(10000, 9, 9)
    bench_speed(20000, 9, 9)
