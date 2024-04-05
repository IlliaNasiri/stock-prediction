
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
    label_width = TestingConfig.get("label_width")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)
    load_model(rnn_model, model_save_path)

    _, _, _, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, 1 - test_size, label_width=label_width)

    # adjusting for the label width START
    X_test = X_test[::label_width, :]
    y_test = y_test[::label_width, :]

    y_test = y_test.flatten()

    # adjusting for the label width END

    rnn_model.eval()
    with torch.inference_mode():

        predictions = rnn_model(X_test)

        predictions = predictions.cpu()
        y_test = y_test.cpu()

        plt.plot( np.arange(y_test.shape[0]), y_test.numpy(), label="actual", color="blue")
        # plt.scatter( np.arange(y_test.shape[0]), predictions.flatten().numpy().squeeze(), label="prediction", color="red", s=0.2 )
        plt.plot( np.arange(y_test.shape[0]), predictions.flatten().numpy().squeeze(), label="prediction", color="red")
        plt.legend()
        plt.title("Predicted and Actual closing price on a given week")
        plt.xlabel("week #")
        plt.ylabel("closing price")
        plt.savefig(visualization_save_path)

visualize()