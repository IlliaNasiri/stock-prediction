
from model import rnn_model
from config import TestingConfig
from utils import *
import numpy as np


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def visualize():
    data_path = TestingConfig.get("data_path")
    attribute = TestingConfig.get("attribute")
    window_size = TestingConfig.get("window_size")
    test_size = TestingConfig.get("test_size")
    model_save_path = TestingConfig.get("model_save_path")
    visualization_save_path = TestingConfig.get("visualization_save_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)
    load_model(rnn_model, model_save_path)

    _, _, _, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, 1 - test_size)

    rnn_model.eval()
    with torch.inference_mode():

        predictions = rnn_model(X_test)

        plt.plot( np.arange(y_test.shape[0]), y_test, label="actual", color="blue" )
        plt.plot( np.arange(y_test.shape[0]), predictions.numpy().squeeze(), label="prediction", color="red" )
        plt.savefig(visualization_save_path)

visualize()