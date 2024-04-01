import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch


def data_to_moving_window(X: pd.DataFrame, attribute: str, window_size: int):
    """
    :param X: dataframe to be used
    :param window_size: size of the moving window
    :param attribute: attribute to be used for the window operation
    :return:
    """
    data = X[attribute].to_numpy().squeeze()
    data = sliding_window_view(data, window_size)
    return data.astype(np.float32)

def csv_to_moving_window(path: str, attribute: str, window_size: int):
    """
    :param path: path of the CSV file
    :param attribute: attribute to slide the window over
    :param window_size: the size of the window
    :return: a matrix of size (|attributes|, n-k+1, k)

    Example: for a dataset of size N: a1, a2, a3, a4, ... , an, and window size k = 3:
    would return:
        a1, a2, a3
        a2, a3, a4,
        a3, a4, a5,
            .
            .
            .
    a(n-2), a(n-1), an
    """
    data =  pd.read_csv(path)[ attribute ].to_numpy().squeeze()
    data = sliding_window_view(data, window_size)
    return data.astype(np.float32)

def get_windowed_data_splits(path, attr, window_size, percentage, convert_to_tensor=True, allow_gpu=True):
    """
    :param path: path to the preprocessed CSV data file
    :param attr: attrbute to retrieve from the CSV
    :param window_size: sliding window size
    :param percentage: how large is the training dataset
    :param convert_to_tensor whether to convert the data to torch tensor
    :param allow_gpu: whether to allow using GPU tensor if CUDA is available
    :return: returns tensors on the device (cpu/gpu):
    windowed_data:
    X_train:
    y_train:
    X_test:
    y_test
    """
    if allow_gpu:
        device = "cuda" if (torch.cuda.is_available() and allow_gpu) else "cpu"

    windowed_data = csv_to_moving_window(path, attr, window_size)
    X = windowed_data[:, :-1]
    y = windowed_data[:, -1]

    if convert_to_tensor:
        X = torch.tensor(X).view(X.shape[0], X.shape[1], 1).to(device)
        y = torch.tensor(y).view(y.shape[0], 1).to(device)

    split_point = int(X.shape[0] * percentage)
    X_train, y_train, X_test, y_test = X[:split_point], y[:split_point], X[split_point:], y[split_point:]
    return windowed_data, X_train, y_train, X_test, y_test

def save_model(model, save_path):
    if save_path is not None:
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        torch.save(model.state_dict(), save_path)

def load_model(model, path):
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
    model.load_state_dict(torch.load(path))