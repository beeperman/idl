import torch
from nn.ImplicitFunction import ImplicitFunctionTriu
from torch import nn
from utils import transpose

class ImplicitLayer(torch.nn.Module):
    def __init__(self, n, m, p, q, f=ImplicitFunctionTriu, X0=None, no_D=False):
        super(ImplicitLayer, self).__init__()

        self.n = n  # A: n*n    X: n*m
        self.m = m  # B: n*p    U: p*m
        self.p = p  # C: q*n
        self.q = q  # D: q*p

        self.A = nn.Parameter(torch.randn(n, n)/n)
        self.B = nn.Parameter(torch.randn(n, p)/n)
        self.C = nn.Parameter(torch.randn(q, n)/n)
        self.D = nn.Parameter(torch.randn(q, p)/n) if not no_D else torch.zeros((q, p), requires_grad=False)

        self.f = f

        self.X0 = torch.randn(n, m) if not X0 else X0

    def forward(self, U):
        U = transpose(U)
        X = self.f.apply(self.A, self.B, self.X0, U)
        return transpose(self.C @ X + self.D @ U)

    def set_weights(self, A, B, C, D):
        self.A.data.copy_(torch.from_numpy(A))
        self.B.data.copy_(torch.from_numpy(B))
        self.C.data.copy_(torch.from_numpy(C))
        self.D.data.copy_(torch.from_numpy(D))

        return self

