import unittest
import torch
from torch import testing

from torch_batch_svd import svd


N, H, W = 100, 9, 3


class TestSVD(unittest.TestCase):
    def test_float(self):
        torch.manual_seed(0)
        a = torch.randn(N, H, W).cuda()
        b = a.clone()
        a.requires_grad = True
        b.requires_grad = True

        U, S, V = svd(a)
        loss = U.sum() + S.sum() + V.sum()
        loss.backward()

        u, s, v = torch.svd(b[0], some=True, compute_uv=True)
        loss0 = u.sum() + s.sum() + v.sum()
        loss0.backward()

        # eigenvectors are only precise up to sign
        testing.assert_allclose(U[0].abs(), u.abs())
        testing.assert_allclose(S[0].abs(), s.abs())
        testing.assert_allclose(V[0].abs(), v.abs())
        testing.assert_allclose(a, U @ torch.diag_embed(S) @ V.transpose(-2, -1))

    def test_double(self):
        torch.manual_seed(0)
        a = torch.randn(N, H, W).cuda().double()
        b = a.clone()
        a.requires_grad = True
        b.requires_grad = True

        U, S, V = svd(a)
        loss = U.sum() + S.sum() + V.sum()
        loss.backward()

        u, s, v = torch.svd(b[0], some=True, compute_uv=True)
        loss0 = u.sum() + s.sum() + v.sum()
        loss0.backward()

        assert U.dtype == torch.double
        assert S.dtype == torch.double
        assert V.dtype == torch.double
        assert a.grad.dtype == torch.double

        # eigenvectors are only precise up to sign
        testing.assert_allclose(U[0].abs(), u.abs())
        testing.assert_allclose(S[0].abs(), s.abs())
        testing.assert_allclose(V[0].abs(), v.abs())
        testing.assert_allclose(a, U @ torch.diag_embed(S) @ V.transpose(-2, -1))

    def test_half(self):
        torch.manual_seed(0)
        a = torch.randn(N, H, W).cuda().half()
        b = a.clone()
        a.requires_grad = True
        b.requires_grad = True

        U, S, V = svd(a)
        loss = U.sum() + S.sum() + V.sum()
        loss.backward()

        assert U.dtype == torch.half
        assert S.dtype == torch.half
        assert V.dtype == torch.half
        assert a.grad.dtype == torch.half

        a_ = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
        testing.assert_allclose(a, a_, atol=0.01, rtol=0.01)

    def test_multiple_gpus(self):
        num_gpus = torch.cuda.device_count()

        for gpu_idx in range(num_gpus):
            device = torch.device("cuda:{}".format(gpu_idx))

            torch.manual_seed(0)
            a = torch.randn(N, H, W).to(device)
            b = a.clone()
            a.requires_grad = True
            b.requires_grad = True

            U, S, V = svd(a)
            loss = U.sum() + S.sum() + V.sum()
            loss.backward()

            u, s, v = torch.svd(b[0], some=True, compute_uv=True)
            loss0 = u.sum() + s.sum() + v.sum()
            loss0.backward()

            # eigenvectors are only precise up to sign
            testing.assert_allclose(U[0].abs(), u.abs())
            testing.assert_allclose(S[0].abs(), s.abs())
            testing.assert_allclose(V[0].abs(), v.abs())

            a_ = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
            testing.assert_allclose(a, a_)


if __name__ == "__main__":
    unittest.main()
