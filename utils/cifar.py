from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os

__file_path = os.path.abspath(__file__)
__proj_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/'
DATA_PATH = Path(__proj_dir)
PATH = DATA_PATH / "data" / "cifar"

def cifar_load(train_bs, valid_bs=10000):
    train_ds = torchvision.datasets.CIFAR10(root=PATH, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Lambda(lambda x: torch.flatten(x))]), download=True)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    valid_ds = torchvision.datasets.CIFAR10(root=PATH, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Lambda(lambda x: torch.flatten(x))]))
    valid_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True)

    return train_ds, train_dl, valid_ds, valid_dl
