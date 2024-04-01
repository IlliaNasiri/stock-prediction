import torch
from model import rnn_model, loss
from config import TestingConfig
from utils import *

def test():

    data_path = TestingConfig.get("data_path")
    attribute = TestingConfig.get("attribute")
    window_size = TestingConfig.get("window_size")
    train_size = TestingConfig.get("train_size")
    save_path = TestingConfig.get("model_save_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)
    load_model(rnn_model, save_path)

    _, _, _, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, train_size)

    rnn_model.eval()
    with torch.inference_mode():
        predictions = rnn_model(X_test)
        l = loss(predictions, y_test)
        print("Testing loss is: ", l.item())


test()