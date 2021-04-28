import torch
import numpy as np
import scipy.sparse as sp
from implicit.Phi import Phi

def make_layerop(layer):
    if isinstance(layer, torch.nn.Linear):
        return LinearOp(layer)
    if isinstance(layer, torch.nn.Conv2d):
        return Conv2dOp(layer)
    if isinstance(layer, torch.nn.BatchNorm2d):
        return BatchNorm2dOp(layer)
    if isinstance(layer, torch.nn.AvgPool2d):
        return AvgPool2dOpL(layer)
    if isinstance(layer, FlattenOpL):
        return layer

class Op(torch.nn.Module):
    def __init__(self, Op):
        super().__init__()
        self.op = Op
        if isinstance(Op, torch.nn.Module):
            super().add_module('op', self.op)
        else:
            print("INFO: {} is not a torch Module".format(self.op))
        self.in_shape = None
        self.out_shape = None

    def forward(self, input):
        self.in_shape = np.array(input.shape)
        output = self.op(input)
        self.out_shape = np.array(output.shape)
        return output

    def in_size(self):
        return int(np.prod(self.in_shape[1:]))

    def out_size(self):
        return int(np.prod(self.out_shape[1:]))


class LayerOp(Op):

    def Ab(self):
        raise NotImplementedError()

class LinearOp(LayerOp):

    def Ab(self):
        A = self.op.weight.clone().detach().cpu().numpy()
        b = self.op.bias.clone().detach().cpu().numpy()[None,...].T
        return sp.csr_matrix(A), b

class Conv2dOp(LayerOp):
    # TODO: Implement Ab based on the layer object given. Note that the pytorch object is given in the constructor as layer
    # TODO: You need to read all information from the object and construct corresponding A and b. Implement the following methods
    # Hint you should be able to access the in_shape at this point.

    def Ab(self):
        inputshape = self.in_shape[1:]
        outputshape = self.out_shape[1:]

        ic, ih, iw = inputshape
        oc, oh, ow = outputshape


        stride = self.op.stride
        padding = self.op.padding
        filters = self.op.weight.clone().detach().cpu().numpy()
        fshape = filters.shape # oc, ic, h, w
        bias = self.op.bias
        bias = bias.clone().detach().cpu().numpy() if bias is not None else bias

        assert self.op.stride[0] == self.op.stride[1], "doesn't support"
        #assert self.op.kernel_size[0] == self.op.kernel_size[1], "doesn't support"
        assert self.op.dilation[0] == self.op.dilation[1], "doesn't support"
        assert self.op.dilation[0] == 1, "doesn't support"
        assert self.op.groups == 1, "doesn't support"
        assert self.op.padding_mode == 'zeros', "doesn't support"
        assert fshape[0] == outputshape[0]

        # useful indices
        iin = np.arange(np.prod(inputshape)).reshape(inputshape)
        iout = np.arange(np.prod(outputshape)).reshape(outputshape)


        # indices
        idxx = []
        idxy = []
        v = []

        for j in range(outputshape[2]):
            for i in range(outputshape[1]):
                #for k in range(outputsize[0]):

                # potential starting position
                ii0 = i * stride[0] - padding[0] # stride
                jj0 = j * stride[1] - padding[1]

                # potential ending position
                ie0 = ii0 + fshape[2] # same as kernel_size
                je0 = jj0 + fshape[3]

                # actual starting position
                ii = np.maximum(0, ii0)
                jj = np.maximum(0, jj0)

                # actual ending position
                ie = np.minimum(ih, ie0)
                je = np.minimum(iw, je0)

                # difference between potential and actual
                dii = ii - ii0; djj = jj - jj0
                die = ie0 - ie; dje = je0 - je

                fltrs = filters[:,:,dii:fshape[2]-die,djj:fshape[3]-dje] # padding
                fshp = fltrs.shape

                inidx = iin[np.newaxis, :, ii:ie, jj:je].repeat(fshp[0], axis=0)
                inidx = inidx.flatten()
                outidx = iout[:,i,j,np.newaxis].repeat(np.prod(fshp[1:]), axis=-1)
                outidx = outidx.flatten()
                vv = fltrs.flatten()

                assert np.size(inidx) == np.size(vv)
                assert np.size(inidx) == np.size(outidx)

                idxx.append(outidx)
                idxy.append(inidx)
                v.append(vv)

        idxx = np.concatenate(idxx, axis=0)
        idxy = np.concatenate(idxy, axis=0)
        v = np.concatenate(v, axis=0)


        A = sp.coo_matrix((v, (idxx, idxy)), shape=(np.prod(outputshape), np.prod(inputshape)))

        b = bias[:, np.newaxis, np.newaxis] * np.ones(outputshape) if bias is not None else np.zeros(outputshape)
        b = b.reshape(-1, 1)
        return A, b




class BatchNorm2dOp(LayerOp):
    def Ab(self):
        inputshape = self.in_shape[1:]
        outputshape = self.out_shape[1:]

        c, h, w = inputshape

        running_mean = self.op.running_mean.clone().detach().cpu().numpy()
        running_var = self.op.running_var.clone().detach().cpu().numpy()
        bias = self.op.bias.clone().detach().cpu().numpy()
        weight = self.op.weight.clone().detach().cpu().numpy()
        eps = self.op.eps

        a = weight / (eps + np.sqrt(running_var))
        b = bias - running_mean * weight / (eps + np.sqrt(running_var))
        a = a.reshape([-1, 1, 1]).repeat(h, 1).repeat(w, 2).flatten()
        b = b.reshape([-1, 1, 1]).repeat(h, 1).repeat(w, 2).reshape([-1, 1])

        A = sp.diags(a)

        assert A.shape[0] == np.prod(inputshape)

        return A, b



class AvgPool2dOpL(LayerOp):
    """The LayerOp version of average pooling since it is linear"""
    def Ab(self):
        inputshape = self.in_shape[1:]
        outputshape = self.out_shape[1:]

        ic, ih, iw = inputshape
        oc, oh, ow = outputshape

        assert oc == ic

        stride = self.op.stride
        padding = self.op.padding
        kernel_size = self.op.kernel_size
        filters = np.zeros((oc, ic, kernel_size, kernel_size), dtype=np.float)
        fshape = filters.shape # oc, ic, h, w
        for i in range(ic):
            filters[i, i] = np.ones((kernel_size, kernel_size))

        # useful indices
        iin = np.arange(np.prod(inputshape)).reshape(inputshape)
        iout = np.arange(np.prod(outputshape)).reshape(outputshape)


        # indices
        idxx = []
        idxy = []
        v = []

        for j in range(outputshape[2]):
            for i in range(outputshape[1]):
                #for k in range(outputsize[0]):

                # potential starting position
                ii0 = i * stride - padding # stride
                jj0 = j * stride - padding

                # potential ending position
                ie0 = ii0 + fshape[2] # same as kernel_size
                je0 = jj0 + fshape[3]

                # actual starting position
                ii = np.maximum(0, ii0)
                jj = np.maximum(0, jj0)

                # actual ending position
                ie = np.minimum(ih, ie0)
                je = np.minimum(iw, je0)

                # difference between potential and actual
                dii = ii - ii0; djj = jj - jj0
                die = ie0 - ie; dje = je0 - je0

                fltrs = filters[:,:,dii:fshape[2]-die,djj:fshape[3]-dje] # padding
                fshp = fltrs.shape
                fltrs = fltrs / np.prod(fshp[2:])

                inidx = iin[np.newaxis, :, ii:ie, jj:je].repeat(fshp[0], axis=0)
                inidx = inidx.flatten()
                outidx = iout[:,i,j,np.newaxis].repeat(np.prod(fshp[1:]), axis=-1)
                outidx = outidx.flatten()
                vv = fltrs.flatten()

                assert np.size(inidx) == np.size(vv)
                assert np.size(inidx) == np.size(outidx)

                idxx.append(outidx)
                idxy.append(inidx)
                v.append(vv)

        idxx = np.concatenate(idxx, axis=0)
        idxy = np.concatenate(idxy, axis=0)
        v = np.concatenate(v, axis=0)


        A = sp.coo_matrix((v, (idxx, idxy)), shape=(np.prod(outputshape), np.prod(inputshape)))

        b = np.zeros(outputshape)
        b = b.reshape(-1, 1)
        return A, b


class FlattenOpL(LayerOp):
    def __init__(self):
        super().__init__(lambda x: x.view(x.shape[0], -1))

    def Ab(self):
        A = sp.eye(self.out_size())
        b = np.zeros((self.out_size(), 1))
        return A, b


def make_activationop(activation):
    if isinstance(activation, torch.nn.ReLU):
        return ReLUOp(activation)
    if isinstance(activation, torch.nn.MaxPool2d):
        return MaxPool2dOp(activation)
    if isinstance(activation, torch.nn.AvgPool2d):
        return AvgPool2dOp(activation)
    if isinstance(activation, FlattenOp):
        return activation

class ActivationOp(Op):

    def phi(self):
        raise NotImplementedError()

class ReLUOp(ActivationOp):
    def __init__(self, activation, alpha=0.0):
        super().__init__(activation)
        self.alpha=alpha

    def phi(self):
        def relu(x):
            return np.maximum(x, self.alpha * x)
        return Phi(relu)

class MaxPool2dOp(ActivationOp):
    def phi(self):
        inputshape = self.in_shape[1:]
        outputshape = self.out_shape[1:]
        stride = self.op.stride
        kernel_size = self.op.kernel_size
        padding = self.op.padding
        assert self.op.dilation == 1, "doesn't support"
        def func(input):
            input = input.T
            m = input.shape[0]
            x = input.reshape(np.concatenate(([-1], inputshape)))
            x = torch.nn.functional.max_pool2d(torch.from_numpy(x), kernel_size=kernel_size, stride=stride, padding=padding)
            x = x.detach().cpu().numpy().reshape([m, np.prod(outputshape)])
            out = np.zeros_like(input)
            out[:, :x.shape[-1]] = x
            return out.T
        return Phi(func)


class AvgPool2dOp(ActivationOp):
    def phi(self):
        inputshape = self.in_shape[1:]
        outputshape = self.out_shape[1:]
        stride = self.op.stride
        kernel_size = self.op.kernel_size
        padding = self.op.padding
        def func(input):
            input = input.T
            m = input.shape[0]
            x = input.reshape(np.concatenate(([-1], inputshape)))
            x = torch.nn.functional.avg_pool2d(torch.from_numpy(x), kernel_size=kernel_size, stride=stride, padding=padding)
            x = x.detach().cpu().numpy().reshape([m, np.prod(outputshape)])
            out = np.zeros_like(input)
            out[:, :x.shape[-1]] = x
            return out.T
        return Phi(func)


class FlattenOp(ActivationOp):
    def __init__(self):
        super().__init__(lambda x: x.view(x.shape[0], -1))

    def phi(self):
        return Phi.identity()
