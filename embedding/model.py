import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Word2Vec, self).__init__()
        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.output = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=False)

    def forward(self, x):
        h = self.input(x)
        u = self.output(h)
        return u
