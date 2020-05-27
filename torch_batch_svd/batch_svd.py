import torch

from . import _c


class BatchSVDFunction(torch.autograd.Function):
    """
    Parameters
    ----------
    A : torch.Tensor
        tensor of shape [B, M, N]
    Returns
    -------
    U : torch.Tensor
        an orthorgonal basis of the row space of A
    S : torch.Tensor
        singular values of A
    V : torch.Tensor
        an othorgonal basis of the column space of A
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor):
        assert A.shape[-1] < 32 and A.shape[-2] < 32, \
            'This implementation only supports matrices having dims smaller than 32'

        is_double = True if A.dtype == torch.double else False
        if A.dtype == torch.half:
            A = A.float()
            ctx.is_half = True
        else:
            ctx.is_half = False

        U0, S, V0 = _c.batch_svd_forward(A, True, 1e-7, 100, is_double)
        k = S.size(1)
        U: torch.Tensor = U0[:, :, :k]
        V: torch.Tensor = V0[:, :, :k]
        ctx.save_for_backward(A, U, S, V)
        if ctx.is_half:
            U, S, V = U.half(), S.half(), V.half()

        return U, S, V

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor, grad_s: torch.Tensor, grad_v: torch.Tensor):
        A, U, S, V = ctx.saved_tensors
        if ctx.is_half:
            grad_u, grad_s, grad_v = grad_u.float(), grad_s.float(), grad_v.float()

        grad_out: torch.Tensor = _c.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            A, True, True, U, S, V
        )
        if ctx.is_half:
            grad_out = grad_out.half()

        return grad_out


batch_svd = BatchSVDFunction.apply
