import torch
import numpy as np
from scipy.sparse import *
from implicit.Phi import Phi

## This would be useful later.. Not now.
class SequentialBlockModel(torch.nn.Module):

    # TODO: implement C
    def __init__(self, blocklist, populate_B=True, populate_C=False):
        super().__init__()
        self.blocklist = torch.nn.ModuleList(blocklist)
        self.populate_B = 1 if populate_B else 0
        self.populate_C = 1 if populate_C else 0
        assert len(blocklist) > 0
        if len(blocklist) <= 1:
            self.populate_C = False
        if populate_C:
            assert len(blocklist[-1].activationops) == 0
        assert len(self.blocklist) >= self.populate_B + self.populate_C, "not enough blocks"

    def forward(self, input):
        out = input
        for block in self.blocklist:
            out = block(out)
        return out

    def getImplicitModel(self, sample_input):

        # This is required to figure out the size of matrices
        out = self.forward(sample_input)
        in_size, out_size = self.blocklist[0].in_size(), np.prod(out.shape[1:])

        A_list = []; b_list = []
        for i in reversed(range(len(self.blocklist))):
            block = self.blocklist[i]
            A, b = block.Ab()
            if i > 0 and block.in_size() < self.blocklist[i-1].out_size(): # dealing with shrinking dim operations
                m, n = A.shape[0], self.blocklist[i-1].out_size() - block.in_size()
                A = bmat([[A, coo_matrix((m,n))]])
            A_list.append(A); b_list.append(b)
        phi_list = [block.phi() for block in reversed(self.blocklist)]
        l = len(A_list)


        # Make A
        in_size_A = self.blocklist[0].out_size() if self.populate_B else self.blocklist[0].in_size()
        out_size_A = self.blocklist[-2].out_size() if self.populate_C else self.blocklist[-1].out_size()

        M = coo_matrix((in_size_A, out_size_A),
                       dtype=np.float)  # Make sure that all A has consistent dimensions for squared A.

        if len(A_list) > self.populate_B + self.populate_C:
            A = block_diag(A_list[self.populate_C:l-self.populate_B])
            A = bmat([[None, A], [M, None]])
        else:
            A = M

        # Make B
        BA = A_list[-1] if self.populate_B else eye(in_size_A)
        Bb = b_list[-1] if self.populate_B else coo_matrix((in_size_A, 1))
        if len(b_list) > self.populate_B + self.populate_C:
            b_stack = b_list[self.populate_C:l - self.populate_B]
            b = vstack(b_stack) if len(b_stack) > 1 else b_stack[0]
            B = bmat([[None, b], [BA, Bb]])
        else:
            B = bmat([[BA, Bb]])

        # Make C
        C = bmat([[A_list[0], coo_matrix((out_size, A.shape[1]-A_list[0].shape[1]))]]) if self.populate_C else eye(out_size, A.shape[1])

        # Make D
        D = bmat([[coo_matrix((out_size, in_size)), b_list[0]]]) if self.populate_C else coo_matrix((out_size, in_size + 1)) # the bias term is added..

        # make phi
        start = 0
        indices = []
        for matrix in A_list[self.populate_C:]:
            indices.append(slice(start, start + matrix.shape[0]))
            start += matrix.shape[0]
        if not self.populate_B:
            phi_list.append(Phi.identity()); indices.append(slice(start, None, None))
        phi = Phi.concatenate(phi_list[self.populate_C:], indices)

        return A, B, C, D, phi

    @staticmethod
    def implicit_forward(A, B, C, D, phi, input):
        m = input.shape[0]
        U = input.clone().detach().cpu().numpy().reshape(m, -1).T if isinstance(input, torch.Tensor) else input.reshape(m, -1).T
        U = np.vstack((U, np.ones((1, U.shape[1]))))
        X, error, status = SequentialBlockModel.inn_pred(A, B@U, phi)

        return np.array(C@X+D@U).T


    @staticmethod
    def inn_pred(A, Z, phi, mitr=300, tol=3e-6):
        X = np.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = phi(A @ X + Z)
            err = np.linalg.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status