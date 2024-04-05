from model import RNN, rnn_model, loss, optimizer
from utils import *
from config import TrainingConfig

def train():
    data_path = TrainingConfig.get("data_path")
    attribute = TrainingConfig.get("attribute")
    window_size = TrainingConfig.get("window_size")
    train_size = TrainingConfig.get("train_size")
    epochs = TrainingConfig.get("epochs")
    label_width = TrainingConfig.get("label_width")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rnn_model.to(device)

    windowed_data, X_train, y_train, X_test, y_test = get_windowed_data_splits(data_path, attribute, window_size, train_size, label_width=label_width)

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