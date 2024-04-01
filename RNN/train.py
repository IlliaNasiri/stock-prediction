from model import RNN, rnn_model
import torch.nn as nn
from utils import *
from training_config import TrainingConfig

def train():
    data_path = TrainingConfig.get("data_path")
    attribute = TrainingConfig.get("attribute")
    window_size = TrainingConfig.get("window_size")
    train_size = TrainingConfig.get("train_size")
    epochs = TrainingConfig.get("epochs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)

    windowed_data, X_train, y_train, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, train_size)

    # define loss fn and an optimizer
    # TODO: make optimizer and loss global so it could be used for both test.py and train.py
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(rnn_model.parameters())

    for epoch in range(epochs):
        rnn_model.train()
        # forward:
        y_pred = rnn_model(X_train)
        # compute loss
        l = loss(y_pred, y_train)
        if epoch % 10 == 0:
            print("Loss on epoch ", epoch, ": ", l.item())
        # zero grad
        optimizer.zero_grad()
        # backward
        l.backward()
        # step
        optimizer.step()

    return rnn_model

def main():
    model = train()
    save_model(model, TrainingConfig.get("model_save_path"))

main()