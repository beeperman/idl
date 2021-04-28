import numpy as np
import scipy.sparse as sp
import torch

from implicit.Block import LayerBlock, BasicBlock
from implicit.Model import SequentialBlockModel
from implicit.Ops import make_activationop, FlattenOp, FlattenOpL
from implicit.Phi import Phi
from nn.resnet import _weights_init




class ResBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = torch.nn.BatchNorm2d(planes)
        relu1 = torch.nn.ReLU()
        conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = torch.nn.BatchNorm2d(planes)
        relu2 = torch.nn.ReLU()

        self.block1 = LayerBlock([conv1, bn1], [relu1])
        self.block2 = LayerBlock([conv2, bn2], [])
        self.act3 = make_activationop(relu2)

        self.As = None

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out += self.shortcut(x)
        out = self.act3(out)
        return out

    def shortcut(self, x):
        if self.stride != 1 or self.in_planes != self.planes:
            return torch.nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)
        else:
            return x

    def Ab(self):
        A1, b1 = self.block1.Ab()
        A2, b2 = self.block2.Ab()
        As = None
        if self.stride != 1 or self.in_planes != self.planes:
            in_shape = self.block1.layerops[0].in_shape
            in_shape[0] = 1
            in_size = np.prod(in_shape)
            out_shape = self.block1.layerops[-1].out_shape
            out_shape[0] = 1
            out_size = np.prod(out_shape)

            # input indices
            x = torch.from_numpy(np.arange(1,in_size+1).reshape(in_shape))
            xx = self.shortcut(x).detach().cpu().numpy()

            assert np.prod(xx.shape == out_shape)

            # output indices
            iout = np.arange(out_size).reshape(out_shape)
            nz = xx.nonzero()
            xnz = iout[nz]
            ynz = xx[nz] - 1
            v = np.ones(len(nz[0]))
            idx = (xnz, ynz)
            As = sp.coo_matrix((v, idx), shape=(self.block2.out_size(), self.block1.in_size()))
        else:
            As = sp.eye(self.block1.in_size())
        #A = sp.bmat([[sp.eye(self.out_size()), None, As], [None, A2, None], [None, None, A1]])
        #b = np.vstack((np.zeros((self.out_size(),1)), b2, b1))
        A = sp.bmat([[A2, As], [None, A1]])
        b = np.vstack((b2, b1))

        #self.As = As
        #self.A1 = A1
        #self.A2 = A2
        #self.b1 = b1
        #self.b2 = b2
        return A,b

    def phi(self):
        phi1 = self.block1.phi()
        #phi2 = self.block2.phi()
        phi3 = self.act3.phi()
        # make phi
        #start = self.out_size()
        #indices = [slice(0, start, None)]
        start = 0
        indices = []
        for block in [self.block2, self.block1]:
            indices.append(slice(start, start + block.out_size()))
            start += block.out_size()
        phi = Phi.concatenate([phi3, phi1], indices)
        return phi

    def in_size(self):
        return self.block1.in_size()

    def out_size(self):
        return self.block2.out_size()

    def implicit_forward(self, input):
        # NOT FOR USE!
        A, b = self.Ab()
        phi = self.phi()

        m = input.shape[0]
        u = input.clone().detach().cpu().numpy().reshape(m, -1).T if isinstance(input, torch.Tensor) else input.reshape(
            m, -1).T

        outb1 = self.block1.implicit_forward(input).T

        outb2 = self.block2.implicit_forward(outb1.T).T

        out = outb2.T + (self.As @ u).T


        u2 = sp.bmat([[sp.coo_matrix((self.block2.in_size(), m))], [u]]).toarray()
        out2 = phi(A@u2+b)
        out2b1 = out2[-self.block2.in_size():, :]
        u2[:self.block2.in_size(), :] = out2b1
        out2 = phi(A@u2+b)
        out2b2 = out2[:self.block2.out_size(),:]
        out2 = out2b2.T

        assert np.isclose(np.abs(out - out2).max(), 0, atol=1e-4)
        return out

class ResNetModel(torch.nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = torch.nn.BatchNorm2d(16)
        relu1 = torch.nn.ReLU()
        avgpool9 = torch.nn.AvgPool2d(8)
        flatten9 = FlattenOpL()
        linear9 = torch.nn.Linear(64, num_classes)

        block1 = LayerBlock([conv1, bn1], [relu1])
        layerlist1 = self._make_layer(16, num_blocks[0], stride=1)
        layerlist2 = self._make_layer(32, num_blocks[1], stride=2)
        layerlist3 = self._make_layer(64, num_blocks[2], stride=2)
        block9 = LayerBlock([avgpool9, flatten9, linear9], [])

        self.apply(_weights_init)
        self.seqmodel = SequentialBlockModel([block1]+layerlist1+layerlist2+layerlist3+[block9])

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * ResBasicBlock.expansion

        return layers

    def forward(self, x):
        return self.seqmodel(x)

    def getImplicitModel(self, sample_input):
        return self.seqmodel.getImplicitModel(sample_input)