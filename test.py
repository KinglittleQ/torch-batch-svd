import torch
from batch_svd import batch_svd


def test():
    torch.manual_seed(0)
    a = torch.randn(100 * 32 * 80, 9, 3).cuda()
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True

    U, S, V = batch_svd(a)
    loss = U.sum() + S.sum() + V.sum()
    loss.backward()

    u, s, v = torch.svd(b[0], some=False, compute_uv=True)
    loss0 = u.sum() + s.sum() + v.sum()
    loss0.backward()

    print(a.grad[0])
    print(b.grad[0])

    print((U[0].abs() - u.abs()).sum())
    print(torch.allclose(U[0].abs(), u.abs()))
    print(torch.allclose(S[0].abs(), s.abs()))
    print(torch.allclose(V[0].abs(), v.abs()))

    print(a[0])
    print(U[0][:, :3] @ S[0].diag() @ V[0].t())


if __name__ == '__main__':
    test()

    print('Finished')
