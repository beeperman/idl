import torch
import scipy.sparse as sp
import numpy as np



def transpose(X):
    assert len(X.size()) == 2
    X = torch.transpose(X, -1, -2)
    return X



def get_valid_accuracy(model, loss_fn, valid_dl, device):
    for xs, ys in valid_dl:
        xs, ys = xs.to(device), ys.to(device)
        pred = model(xs)
        pred = pred if isinstance(pred, torch.Tensor) else torch.from_numpy(pred).to(device)
        loss = loss_fn(pred, ys)
        pred_i = np.argmax(pred.cpu().detach().numpy(), axis=-1)
        correct = np.sum([1 if ys[i] == pred_i[i] else 0 for i in range(len(ys))])
        return loss, correct/len(ys)


def get_ABCD_from_NN(model_state):
    w0 = model_state['0.weight'].cpu().numpy()
    w1 = model_state['2.weight'].cpu().numpy()

    n, p = w0.shape
    q, _ = w1.shape

    A = np.zeros((n, n))
    B = w0
    C = w1
    D = np.zeros((q, p))

    return A, B, C, D

def get_ABCD_from_NN_NBC(model_state):
    w0 = model_state['0.weight'].cpu().numpy()
    w1 = model_state['2.weight'].cpu().numpy()

    n, p = w0.shape
    q, _ = w1.shape

    A = np.zeros((n, n))
    B = w0
    C = w1
    D = np.zeros((q, p))

    return A, B, C, D

def get_ABCD_from_NN_784604010(model_state):
    dat = model_state

    B = sp.bmat([[sp.coo_matrix((40, 784))], [dat["0.weight"].cpu().numpy()]]).toarray()
    A = sp.bmat([[None, dat["2.weight"].cpu().numpy()], [sp.coo_matrix((60, 40)), None]]).toarray()
    C = sp.bmat([[dat["4.weight"].cpu().numpy(), sp.coo_matrix((10, 60))]]).toarray()
    D = np.zeros((10, 784))

    return A, B, C, D

def set_parameters_uniform(model, parameter=0.05, seed=None):
    if seed:
        print("using random seed: {}".format(seed))
        np.random.seed(seed)
    for name, param in model.named_parameters():
        print("setting weight {} of shape {} to be uniform(-{}, {})".format(name, param.shape, parameter, parameter))
        p = np.random.uniform(low=-parameter, high=parameter, size=param.shape)
        param.data.copy_(torch.from_numpy(p))
    return model