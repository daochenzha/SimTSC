import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_dataset_from_npy(path):
    """ Read dataset from .npy file
    """
    data = np.load(path, allow_pickle=True)
    return data[()]['X'], data[()]['y'], data[()]['train_idx'], data[()]['test_idx']

def read_dataset(ucr_root_dir, dataset_name, shot):
    """ Read univariate dataset from UCR
    """
    dataset_dir = os.path.join(ucr_root_dir, dataset_name)
    df_train = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TRAIN.tsv'), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TEST.tsv'), sep='\t', header=None)

    y_train = df_train.values[:, 0].astype(np.int64)
    y_test = df_test.values[:, 0].astype(np.int64)
    y = np.concatenate((y_train, y_test))
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X_train = df_train.drop(columns=[0]).astype(np.float32)
    X_test = df_test.drop(columns=[0]).astype(np.float32)

    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])

    X_train = X_train.values
    X_test = X_test.values
    X = np.concatenate((X_train, X_test))
    idx = np.array([i for i in range(len(X))])

    np.random.shuffle(idx)
    train_idx, test_idx = idx[:int(len(idx)*0.8)], idx[int(len(idx)*0.8):]

    tmp = [[] for _ in range(len(np.unique(y)))]
    for i in train_idx:
        tmp[y[i]].append(i)
    train_idx = []
    for _tmp in tmp:
        train_idx.extend(_tmp[:shot])

    # znorm
    X[np.isnan(X)] = 0
    std_ = X.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    X = (X - X.mean(axis=1, keepdims=True)) / std_

    # add a dimension to make it multivariate with one dimension 
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y, train_idx, test_idx

def read_multivariate_dataset(root_dir, dataset_name, shot):
    """ Read multivariate dataset
    """
    X = np.load(os.path.join(root_dir, dataset_name+".npy"), allow_pickle=True)
    y = np.loadtxt(os.path.join(root_dir, dataset_name+'_label.txt'))
    y = y.astype(np.int64)

    dim = X[0].shape[0]
    max_length = 0
    for _X in X:
        if _X.shape[1] > max_length:
            max_length = _X.shape[1]

    X_list = []
    for i in range(len(X)):
        _X = np.zeros((dim, max_length))
        _X[:, :X[i].shape[1]] = X[i]
        X_list.append(_X)
    X = np.array(X_list, dtype=np.float32)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    idx = np.array([i for i in range(len(X))])

    np.random.shuffle(idx)
    train_idx, test_idx = idx[:int(len(idx)*0.8)], idx[int(len(idx)*0.8):]

    tmp = [[] for _ in range(len(np.unique(y)))]
    for i in train_idx:
        tmp[y[i]].append(i)
    train_idx = []
    for _tmp in tmp:
        train_idx.extend(_tmp[:shot])

    # znorm
    std_ = X.std(axis=2, keepdims=True)
    std_[std_ == 0] = 1.0
    X = (X - X.mean(axis=2, keepdims=True)) / std_

    return X, y, train_idx, test_idx

def read_X(ucr_root_dir, dataset_name):
    """ Read the raw time-series
    """
    dataset_dir = os.path.join(ucr_root_dir, dataset_name)
    df_train = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TRAIN.tsv'), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(dataset_dir, dataset_name+'_TEST.tsv'), sep='\t', header=None)

    X_train = df_train.drop(columns=[0]).astype(np.float32)
    X_test = df_test.drop(columns=[0]).astype(np.float32)

    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])

    X_train = X_train.values
    X_test = X_test.values
    X = np.concatenate((X_train, X_test), axis=0)

    return X

class Logger:
    def __init__(self, f):
        self.f = f

    def log(self, content):
        print(content)
        self.f.write(content + '\n')
        self.f.flush()


