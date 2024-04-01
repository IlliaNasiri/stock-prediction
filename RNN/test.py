import torch
from model import rnn_model
from training_config import TrainingConfig
from utils import *
import torch.nn as nn

def test():

    data_path = TrainingConfig.get("data_path")
    attribute = TrainingConfig.get("attribute")
    window_size = TrainingConfig.get("window_size")
    train_size = TrainingConfig.get("train_size")
    save_path = TrainingConfig.get("model_save_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)
    load_model(rnn_model, save_path)

    loss = nn.L1Loss()

    _, _, _, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, train_size)

    rnn_model.eval()
    with torch.inference_mode():
        predictions = rnn_model(X_test)
        l = loss(predictions, y_test)
        print("Testing loss is: ", l.item())


test()