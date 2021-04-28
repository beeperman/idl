import torch
import numpy as np
from torch.autograd import Function
from utils.utils import transpose

class ImplicitFunction(Function):
    @staticmethod
    def forward(ctx, A, B, X0, U):
        with torch.no_grad():
            X, err, status = ImplicitFunction.inn_pred(A, B @ U)
        ctx.save_for_backward(A, B, X, U)
        if status not in "converged":
            print("Picard iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        A, B, X, U = ctx.saved_tensors

        grad_output = grad_outputs[0]
        assert grad_output.size() == X.size()


        n, m = X.size()
        p, _ = U.size()

        DPhi = ImplicitFunction.dphi(A @ X + B @ U)
        V, err, status = ImplicitFunction.inn_pred_grad(A.T, DPhi * grad_output, DPhi)
        if status not in "converged":
            print("Gradient iteration not converging!", err, status)
        grad_A = V @ X.T
        grad_B = V @ U.T
        grad_U = B.T @ V

        return (grad_A, grad_B, torch.zeros_like(X), grad_U)

    @staticmethod
    def phi(X):
        return torch.clamp(X, min=0)

    @staticmethod
    def dphi(X):
        grad = X.clone().detach()
        grad[X <= 0] = 0
        grad[X > 0] = 1

        return grad

    @staticmethod
    def inn_pred(A, Z, mitr=300, tol=3e-6):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = ImplicitFunction.phi(A @ X + Z)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def inn_pred_grad(AT, Z, DPhi, mitr=300, tol=3e-6):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = DPhi * (AT @ X) + Z
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status


class ImplicitFunctionTriu(ImplicitFunction):
    def forward(ctx, A, B, X0, U):
        A = A.triu_(1)
        return ImplicitFunction.forward(ctx, A, B, X0, U)

    def backward(ctx, *grad_outputs):
        grad_A, grad_B, x, u = ImplicitFunction.backward(ctx, *grad_outputs)
        return (grad_A.triu(1), grad_B, x, u)


class ImplicitFunctionInf(ImplicitFunction):
    def forward(ctx, A, B, X0, U):

        # project A on |A|_inf=v
        v = 0.95
        #v = 0.2
        # TODO: verify and speed up
        A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                # proposal: alpha <= a[i]
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v)
            # write back
            A_np[idx, :] = a
        # 0.0
        #A_np = np.zeros_like(A_np)
        A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))

        return ImplicitFunction.forward(ctx, A, B, X0, U)

    def backward(ctx, *grad_outputs):
        grad_A, grad_B, x, u = ImplicitFunction.backward(ctx, *grad_outputs)
        return (grad_A, grad_B, x, u)
