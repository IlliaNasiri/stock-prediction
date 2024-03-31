import torch
import torch.nn as nn

# TODO: play with the architecture by: Stacking RNNs, more deep and broad dense layers
class RNN(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(RNN, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.rnn = nn.RNN(input_size=n_features, hidden_size=n_hidden, bias=False, batch_first=True, nonlinearity="relu")
        self.LL1 = nn.Linear(n_hidden, 64)
        self.ReLU = nn.ReLU()
        self.LL2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        :param x: the input to the RNN, assumed to have dimenstions: (batch_size, length_of_sequence, n_features)
        :return: the predicted price
        """
        h0 = torch.zeros(1, x.shape[0], self.n_hidden)
        out, _ = self.rnn(x, h0)
        # hn: last hidden state
        hn = out[:, -1, :].squeeze()
        # print(hn.shape)

        y = self.LL1(hn)
        y = self.ReLU(y)
        y = self.LL2(y)

        return y