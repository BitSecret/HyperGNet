import random
from utils import Configuration as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor(10000.0)) / d_model))
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
            self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)  # 默认0是填充的字符
        else:
            self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)

    def forward(self, x):
        return self.emb(x) * torch.sqrt(self.d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # // 是取整除的意思
        self.h = h
        self.linear = [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]  # 4个线性层

    def forward(self, x, mask=None):
        # 把x分别经过一层Linear变换得到QKV，tensor size不变
        query, key, value = [l(x) for l in self.linear]

        # 将QKV的d_model维向量分解为h * d_k
        query, key, value = [x.view(-1, self.h, self.d_k).transpose(0, 1) for x in (query, key, value)]

        # Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(query.size(-1))  # 注意力评分
        scores = F.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)  # 结果

        # "Concat" using a view and apply a final linear.
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k)

        # final linear
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
        self.input = nn.Linear(d_model, d_ff)  # Linear 默认带bias
        self.output = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.output(F.relu(self.input(x)))
