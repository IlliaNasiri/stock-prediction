import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


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