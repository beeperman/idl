import numpy as np





class Phi(object):
    """
    This is a callable class. It is required to be a class because it will contain information about the phi's of
    the implicit model. Phi supports concatenation which is essential for large implicit model.
    """
    def __init__(self, f, i=None, p=None):
        self.pre_phi = p # any previous phi/callable
        self.indices = i # could be the indices of the corresponding input for corresponding phi
        self.function = f # is a callable if indices is None else is a list of phis/callables

    def __call__(self, input):
        if self.pre_phi: # apply any previous phis
            input = self.pre_phi(input)
        if not self.indices: # if not concatenated
            return self.function(input)
        else: # concatenated
            out = np.zeros_like(input)
            counter = 0
            for idx in self.indices:
                out[idx, :] = self.function[counter](input[idx, :])
                counter += 1
            return out

    def set_pre(self, p):
        self.pre_phi = p
        return self

    @classmethod
    def concatenate(cls, phi_list, indices):
        """
        This function concatenate the phi list into a single phi.
        :param phi_list: list of phi/callable to be concatenated
        :param indices: the indices of corresponding phi to the input
        :return: A new Phi class
        """
        return Phi(phi_list, indices)

    @staticmethod
    def identity():
        return Phi(lambda x: x)