import torch

from . import _c


class BatchSVDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, some=True, compute_uv=True, out=None):
        """
        This function returns `(U, S, V)`
        which is the singular value decomposition
        of a input real matrix or batches of real matrices `input`

        :param ctx:
        :param input:
        :param out:
        :return:
        """
        assert (
            input.shape[-1] < 32 and input.shape[-2] < 32
        ), "This implementation only supports matrices having dims smaller than 32"

        is_double = True if input.dtype == torch.double else False
        if input.dtype == torch.half:
            input = input.float()
            ctx.is_half = True
        else:
            ctx.is_half = False

        if out is None:
            b, m, n = input.shape
            U = torch.empty(b, m, m, dtype=input.dtype).to(input.device)
            S = torch.empty(b, min(m, n), dtype=input.dtype).to(input.device)
            V = torch.empty(b, n, n, dtype=input.dtype).to(input.device)
        else:
            U, S, V = out

        _c.batch_svd_forward(input, U, S, V, True, 1e-7, 100, is_double)
        U.transpose_(1, 2)
        V.transpose_(1, 2)
        if ctx.is_half:
            U, S, V = U.half(), S.half(), V.half()

        k = S.size(1)
        U_reduced: torch.Tensor = U[:, :, :k]
        V_reduced: torch.Tensor = V[:, :, :k]
        ctx.save_for_backward(input, U_reduced, S, V_reduced)

        if not compute_uv:
            U = torch.zeros(b, m, m, dtype=S.dtype).to(input.device)
            V = torch.zeros(b, m, m, dtype=S.dtype).to(input.device)
            return U, S, V

        return (U_reduced, S, V_reduced) if some else (U, S, V)

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor, grad_s: torch.Tensor, grad_v: torch.Tensor):
        A, U, S, V = ctx.saved_tensors
        if ctx.is_half:
            grad_u, grad_s, grad_v = grad_u.float(), grad_s.float(), grad_v.float()

        grad_out: torch.Tensor = _c.batch_svd_backward(
            [grad_u, grad_s, grad_v], A, True, True, U.to(A.dtype), S.to(A.dtype), V.to(A.dtype)
        )
        if ctx.is_half:
            grad_out = grad_out.half()

        return grad_out


svd = BatchSVDFunction.apply
