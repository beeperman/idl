import torch
import numpy as np

from torch import nn
from torch import optim
import torch.nn.functional as F

from nn import ImplicitLayer, ImplicitFunctionInf
from utils import transpose, cifar_load, get_valid_accuracy, Logger

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

epoch = 10
bs = 100
lr = 5e-4

train_ds, train_dl, valid_ds, valid_dl = cifar_load(bs)

model = ImplicitLayer(300, 100, 3072, 10, f=ImplicitFunctionInf, no_D=False)
opt = optim.Adam(ImplicitLayer.parameters(model), lr=lr)
loss_fn = F.cross_entropy
model.to(device)


logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc"],
                dir_name="CIFAR_Implicit_300_Inf")

for i in range(epoch):
    j = 0
    for xs, ys in train_dl:
        pred = model(xs.to(device))
        loss = loss_fn(pred, ys.to(device))

        loss.backward()
        opt.step()
        opt.zero_grad()
        valid_res = get_valid_accuracy(model, loss_fn, valid_dl, device)

        log_dict = {
            "batch": j,
            "loss": loss,
            "valid_loss": valid_res[0],
            "valid_acc":valid_res[1],
        }
        logger.log(log_dict, model, "valid_acc")

        j+=1
    print("--------------epoch: {}. loss: {}".format(i, loss))

a = 1
