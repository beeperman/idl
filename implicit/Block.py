import torch
import numpy as np
from implicit.Phi import Phi
from implicit.Ops import make_activationop, make_layerop
from scipy import sparse as sp



class BasicBlock(torch.nn.Module):
    """
    This is an implicit block. For any sequential feedforward neural net we can construct the A matrix as following:
    A = [0 w2 0 0], B= [[b2, 0...]
        [0 0 w1 0],     [b1, 0...]
        [0 0 0 w0],     [b0, 0...]
        [0 0 0 0 ],     [0, I]]
    x = phi(Ax+Bu). phi = [phi2, phi1, phi0, identity]
    yhat = Cx+Du, C = I, D = 0
    We can rearrange the matrices to restore the matrix formulation in the paper.
    In the 3 layer example, we have 3 blocks.
    Block 0: w0, b0, phi0
    Block 1: w1, b1, phi1
    Block 2: w2, b2, phi2
    For simple dense layer a block is simply A = W, b = b, phi = ReLU
    Note that this formulation is not like the regular nn sequential paradigm.
    This is due to the fundamental difference between deep nets and implicit model.
    However the nn sequential paradigm is achievable if we apply some translation.
    """
    def phi(self):
        raise NotImplementedError()

    def Ab(self):
        """
        Return matrix A and bias vector b
        Matrix A must be sparse to reduce memory consumption
        """
        raise NotImplementedError()

    def in_size(self):
        raise NotImplementedError()

    def out_size(self):
        raise NotImplementedError()

    def implicit_forward(self, input):
        m = input.shape[0]
        u = input.clone().detach().cpu().numpy().reshape(m, -1).T if isinstance(input, torch.Tensor) else input.reshape(m, -1).T

        A, b = self.Ab()
        phi = self.phi()

        out = phi(A@u+b)
        out = out.T

        #assert np.isclose(np.linalg.norm(out-self.forward(input).detach().cpu().numpy().reshape(m, -1).T), 0)

        return out

class MegaBlock(BasicBlock):
    """
    This is going to contain a lot of LayerBlocks and make them into a big block.
    Note that the input size and output size could be different what we infer from A matrix.
    The implicit model would be itself a MegaBlock.
    ResNet would use multiple MegaBlocks.
    """
    def __init__(self):
        pass

class LayerBlock(BasicBlock):
    def __init__(self, layer_list, activation_list):
        super(LayerBlock, self).__init__()

        assert len(layer_list) > 0
        self.layer_list = layer_list
        self.activation_list = activation_list
        self.layerops = torch.nn.ModuleList([make_layerop(l) for l in layer_list])
        self.activationops = torch.nn.ModuleList([make_activationop(a) for a in activation_list])

    def phi(self):
        p = Phi.identity()
        for a in self.activationops:
            p = a.phi().set_pre(p)
        return p

    def Ab(self):
        A = None
        b = None
        for l in self.layerops:
            A_l, b_l = l.Ab()
            if A is None:
                A = A_l
                b = b_l
            else:
                b = A_l @ b + b_l
                A = A_l @ A
        return A, b

    def in_size(self):
        return self.layerops[0].in_size()

    def out_size(self): # intentionally kept not true. Please read true out_size from the in_size of the next block.
        return self.layerops[-1].out_size()

    def out_size_true(self):
        return self.activationops[-1].out_size()

    def forward(self, input):
        out = input
        for l in self.layerops:
            out = l(out)
        for a in self.activationops:
            out = a(out)
        return out

