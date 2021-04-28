import warnings
from skimage import exposure
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader


# modified from github hparik11
# /
# German-Traffic-Sign-Recognition

import os

__file_path = os.path.abspath(__file__)
__proj_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/'
DATA_PATH = Path(__proj_dir)
PATH = DATA_PATH / "data" / "traffic"

training_file = PATH / "train.p"
testing_file = PATH / "test.p"
training_preprocessed_file = PATH / "train_preprocessed.p"
validation_preprocessed_file = PATH / "valid_preprocessed.p"
testing_preprocessed_file = PATH / "test_preprocessed.p"

def preprocess_dataset(X, y, one_hot=False):
    '''
    - convert images to grayscale,
    - scale from [0, 255] to [0, 1] range,
    - use localized histogram equalization as images differ
      in brightness and contrast significantly
    ADAPTED FROM: http://navoshta.com/traffic-signs-classification/
    https://github.com/hparik11/German-Traffic-Sign-Recognition/blob/master/German_Traffic_Sign_Classifier.ipynb
    '''

    # Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]

    # Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    # adjust histogram
    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])

    if one_hot:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(43)[y]

    # Flatten
    X = X.reshape(X.shape[0], -1)
    return X, y

def makeDict(X, y):
    return {'features': X, 'labels': y}

def retriveDataFromDict(dictn):
    return dictn['features'], dictn['labels']


def preprocess():
    ## Load the data

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X, y = train['features'], train['labels']
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=4000, random_state=0)

    X_test, y_test = test['features'], test['labels']
    print("X_train shape:", X.shape)
    print("y_train shape:", y.shape)
    #print("X_valid shape:", X_valid.shape)
    #print("y_valid shape:", y_valid.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    ## Please run this only first time and save the data into file
    ## so next time, we can directly load from file.

    print("Preprocessing the data to improve feature extraction...")
    print("This might take a while...")

    X_train_preprocessed, y_train_preprocessed = preprocess_dataset(X, y)
    print("training set preprocessing complete!", X_train_preprocessed.shape)

    #X_valid_preprocessed, y_valid_preprocessed = preprocess_dataset(X_valid, y_valid)
    #print("cross validation set preprocessing complete!", X_valid_preprocessed.shape)

    X_test_preprocessed, y_test_preprocessed = preprocess_dataset(X_test, y_test)
    print("test set preprocessing complete!", X_test_preprocessed.shape)



    pickle.dump(makeDict(X_train_preprocessed, y_train_preprocessed), open(training_preprocessed_file, "wb"))
    #pickle.dump(makeDict(X_valid_preprocessed, y_valid_preprocessed), open(validation_preprocessed_file, "wb"))
    pickle.dump(makeDict(X_test_preprocessed, y_test_preprocessed), open(testing_preprocessed_file, "wb"))

def traffic_load(train_bs):
    if not training_preprocessed_file.exists():
        preprocess()
    with open(training_preprocessed_file, mode='rb') as f:
        x_train, y_train = retriveDataFromDict(pickle.load(f))
    with open(testing_preprocessed_file, mode='rb') as f:
        x_test, y_test = retriveDataFromDict(pickle.load(f))

    x_train, y_train, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_test, y_test)
    )
    y_train, y_test = y_train.type(torch.LongTensor), y_test.type(torch.LongTensor)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)

    valid_ds = TensorDataset(x_test, y_test)
    valid_dl = DataLoader(valid_ds, batch_size=x_test.shape[0], shuffle=True)

    return train_ds, train_dl, valid_ds, valid_dl