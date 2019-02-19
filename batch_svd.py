import torch
import torch_batch_svd


class BatchSVDFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, x):
        U0, S, V0 = torch_batch_svd.batch_svd_forward(x, True, 1e-7, 100)
        k = S.size(1)
        U = U0[:, :, :k]
        V = V0[:, :, :k]
        self.save_for_backward(x, U, S, V)

        return U, S, V

    @staticmethod
    def backward(self, grad_u, grad_s, grad_v):
        x, U, S, V = self.saved_variables

        grad_out = torch_batch_svd.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )

        return grad_out


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N]
    return:
        U, S, V = batch_svd(x) where x = USV^T
    """
    return BatchSVDFunction.apply(x)
