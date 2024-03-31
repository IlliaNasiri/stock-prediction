from model import RNN
import argparse
import torch
import torch.nn as nn
from utils import *

# will be able to specify: epochs, data file, attribute, where to save the trained weights

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("-data_path", type=str, required=True, help="path to your csv data")
    parser.add_argument("-attr", type=str, required=True)
    parser.add_argument("-output_path", type=str, help="path to save trained model")
    return parser.parse_args()


def train(args):
    epochs, data_path, attr, output_path = args.epochs, args.data_path, args.attr, args.output_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    windowed_data, X_train, y_train, X_test, y_test = get_windowed_data_splits(data_path, attr, 5, 1)

    model = RNN(n_features=1, n_hidden=64).to(device)
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        # forward:
        y_pred = model(X_train)
        # compute loss
        l = loss(y_pred, y_train)
        print(l.item())
        # zero grad
        optimizer.zero_grad()
        # backward
        l.backward()
        # step
        optimizer.step()

    return model

def save_model(path, model):
    if path is not None:
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        torch.save(model.state_dict(), path)


def main():
    args = parse_arguments()
    model = train(args)
    save_model(args.output_path, model)

main()