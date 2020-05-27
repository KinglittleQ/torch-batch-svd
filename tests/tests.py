import torch
from torch import testing

from torch_batch_svd import batch_svd


def test_float():
    torch.manual_seed(0)
    a = torch.randn(1000000, 9, 3).cuda()
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True

    U, S, V = batch_svd(a)
    loss = U.sum() + S.sum() + V.sum()
    loss.backward()

    u, s, v = torch.svd(b[0], some=True, compute_uv=True)
    loss0 = u.sum() + s.sum() + v.sum()
    loss0.backward()

    testing.assert_allclose(U[0].abs(), u.abs())  # eigenvectors are only precise up to sign
    testing.assert_allclose(S[0].abs(), s.abs())
    testing.assert_allclose(V[0].abs(), v.abs())
    testing.assert_allclose(a, torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1)))


def test_double():
    torch.manual_seed(0)
    a = torch.randn(10, 9, 3).cuda().double()
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True

    U, S, V = batch_svd(a)
    loss = U.sum() + S.sum() + V.sum()
    loss.backward()

    u, s, v = torch.svd(b[0], some=True, compute_uv=True)
    loss0 = u.sum() + s.sum() + v.sum()
    loss0.backward()

    assert U.dtype == torch.double
    assert S.dtype == torch.double
    assert V.dtype == torch.double
    assert a.grad.dtype == torch.double
    testing.assert_allclose(U[0].abs(), u.abs())  # eigenvectors are only precise up to sign
    testing.assert_allclose(S[0].abs(), s.abs())
    testing.assert_allclose(V[0].abs(), v.abs())
    testing.assert_allclose(a, torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1)))


def test_half():
    torch.manual_seed(0)
    a = torch.randn(10, 9, 3).cuda().half()
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True

    U, S, V = batch_svd(a)
    loss = U.sum() + S.sum() + V.sum()
    loss.backward()

    assert U.dtype == torch.half
    assert S.dtype == torch.half
    assert V.dtype == torch.half
    assert a.grad.dtype == torch.half
    testing.assert_allclose(a, torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1)))
