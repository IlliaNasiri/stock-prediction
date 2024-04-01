
from model import rnn_model
from training_config import TrainingConfig
from utils import *
import numpy as np


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def visualize():
    data_path = TrainingConfig.get("data_path")
    attribute = TrainingConfig.get("attribute")
    window_size = TrainingConfig.get("window_size")
    train_size = TrainingConfig.get("train_size")
    model_save_path = TrainingConfig.get("model_save_path")
    visualization_save_path = TrainingConfig.get("visualization_save_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)
    load_model(rnn_model, model_save_path)

    _, _, _, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, train_size)

    rnn_model.eval()
    with torch.inference_mode():
        predictions = rnn_model(X_test)

        plt.plot( np.arange(y_test.shape[0]), y_test, label="actual", color="blue" )
        plt.plot( np.arange(y_test.shape[0]), predictions.numpy().squeeze(), label="prediction", color="red" )
        plt.savefig(visualization_save_path)

visualize()
