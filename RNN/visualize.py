
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
    train_size = TestingConfig.get("train_size")
    model_save_path = TestingConfig.get("model_save_path")
    visualization_save_path = TestingConfig.get("visualization_save_path")

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

# preds = X_test[0].squeeze().tolist()
#
# for i in range( 20 ):
#     new_pred = rnn_model(torch.tensor(preds[-4:], dtype=torch.float32).view(1,4,1))
#     preds += [new_pred.item()]
#
# preds = np.array(preds[4:])
#
# plt.plot(np.arange(y_test.shape[0]), y_test, label="actual", color="blue" )
# plt.plot( np.arange(20), preds, label="prediction", color="red" )
# plt.savefig("RNN/visualizations/aaa.jpg")