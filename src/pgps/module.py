import random
from utils import Configuration as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + Variable(self.pe[0:x.size(0)], requires_grad=False)


class Embedding(nn.Module):
    def __init__(self, vocab, d_model, padding=False):
        super(Embedding, self).__init__()
        self.d_model = d_model
        if padding:
            # 0 is the default padding character
            self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)
        else:
            self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)

    def forward(self, x):
        return self.emb(x) * torch.sqrt(self.d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.h = h
        self.linear = [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]  # 4 Linear

    def forward(self, x, mask=None):
        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        query, key, value = [l(x) for l in self.linear[0:3]]

        # decompose the d_model dimensional vector of QKV into h * d_k
        query, key, value = [x.view(-1, self.h, self.d_k).transpose(0, 1) for x in (query, key, value)]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(query.size(-1))  # attention score
        if mask is not None:  # masked attention
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k)
        x = self.linear[-1](x)

        return x


class TaskSpecificAttention(nn.Module):
    def __init__(self, h, d_model):
        super(TaskSpecificAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.h = h
        self.linear = [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]  # 4 Linear

    def forward(self, x, task):
        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        query = self.linear[0](task)
        key, value = [l(x) for l in self.linear[1:3]]

        # decompose the d_model dimensional vector of QKV into h * d_k
        query, key, value = [x.view(-1, self.h, self.d_k).transpose(0, 1) for x in (query, key, value)]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(query.size(-1))  # attention score
        scores = F.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k)
        x = self.linear[-1](x)

        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.input = nn.Linear(d_model, d_ff)  # Linear, has bias by default
        self.output = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.output(F.relu(self.input(x)))


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    p = PositionalEncoding(512, 10)
    print(p.pe.shape)
    print(p.pe.size(0))
