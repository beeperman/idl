import gzip
import pickle
from pathlib import Path
import requests
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

__file_path = os.path.abspath(__file__)
__proj_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/'
DATA_PATH = Path(__proj_dir)
PATH = DATA_PATH / "data" / "mnist"


FILENAME = "mnist.pkl.gz"

def mnist_download():
    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


def mnist_load(train_bs, valid_bs=10000):
    mnist_download()

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
        x_train, y_train, x_valid, y_valid, x_test, y_test = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
        )
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True)

    return train_ds, train_dl, valid_ds, valid_dl