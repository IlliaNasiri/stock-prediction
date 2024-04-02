import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class RNN(nn.Module):
    def __init__(self, n_features, n_hidden, num_rnn_layers=1, dense_layers=[(64, 64), (64, 1)]):
        super(RNN, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.num_rnn_layers = num_rnn_layers
        self.rnn = nn.RNN(input_size=n_features, hidden_size=n_hidden, num_layers=num_rnn_layers, bias=False, batch_first=True, nonlinearity="relu")
        self.ReLU = nn.ReLU()
        self.dense_layers = nn.ModuleList(
            [nn.Linear(*dense_layer) for dense_layer in dense_layers]
        )


    def forward(self, x):
        """
        :param x: the input to the RNN, assumed to have dimenstions: (batch_size, length_of_sequence, n_features)
        :return: the predicted price
        """
        h0 = torch.zeros(self.num_rnn_layers, x.shape[0], self.n_hidden)
        out, _ = self.rnn(x, h0)

        # y: last hidden state
        y = out[:, -1, :].squeeze()

        for i in range( len(self.dense_layers) ):
            layer = self.dense_layers[i]
            y = layer(y)
            if i < len(self.dense_layers) - 1:
                y = self.ReLU(y)

        return y

# parameters from config.py
n_features = ModelConfig.get("n_features")
n_hidden = ModelConfig.get("n_hidden")
num_rnn_layers = ModelConfig.get("num_rnn_layers")
dense_layers = ModelConfig.get("dense_layers")

rnn_model = RNN(n_features, n_hidden, num_rnn_layers, dense_layers)
loss = nn.L1Loss()
optimizer = torch.optim.Adam(rnn_model.parameters())