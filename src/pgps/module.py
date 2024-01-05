import random
from utils import Configuration as config
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """Standard positional encoding from original transformer."""
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 'pe' will be retained when model saving and loading, but it will not be updated during the training.
        self.register_buffer('pe', pe)  # torch.Size([max_len, d_model])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :return result: torch.Size([batch_size, max_len, d_model])
        """
        return x + self.pe[0:x.size(0)]    # padding的地方也要加位置编码？


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        # 0 is the default padding character
        self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len])
        :return result: torch.Size([batch_size, max_len, d_model])
        """
        return self.emb(x) * torch.sqrt(self.d_model)  # the variance of `emb` distribution becomes 1


class SelfAttention(nn.Module):
    def __init__(self, h, d_model):
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.h = h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])  # 4 Linear

    def forward(self, x, mask=None):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :param mask: torch.Size([max_len, max_len])
        :return result: torch.Size([batch_size, max_len, d_model])
        """

        batch_size = x.size(0)

        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear[0:3], [x] * 3)]  # [batch_size, h, max_len, d_k]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.d_k)  # [batch_size, h, max_len, max_len]
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, max_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)

        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [batch_size, max_len, d_model]
        x = self.linear[-1](x)

        return x


class TaskSpecificAttention(nn.Module):
    def __init__(self, h, d_model):
        super(TaskSpecificAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.h = h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])  # 4 Linear

    def forward(self, x, task):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :param task: torch.Size([batch_size, 1, d_model])
        :return result: torch.Size([batch_size, 1, d_model])
        """

        batch_size = x.size(0)

        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        # [batch_size, h, 1, d_k]
        query = self.linear[0](task).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # [batch_size, h, max_len, d_k]
        key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l in self.linear[1:3]]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.d_k)  # [batch_size, h, 1, max_len]
        scores = F.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [batch_size, 1, d_model]
        x = self.linear[-1](x)

        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :return result: torch.Size([batch_size, max_len, d_model])
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.input = nn.Linear(d_model, d_ff)  # has bias by default
        self.output = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :return result: torch.Size([batch_size, max_len, d_model])
        """
        return self.output(F.relu(self.input(x)))


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    p = PositionalEncoding(512, 10)
    print(p.pe)
    print(p.pe.shape)
    print(p.pe.size(0))

    # # 假设每个序列的有效长度是5，剩余的位置是填充项
    # max_len = 8  # 序列的最大长度
    # valid_length = 5  # 有效的长度
    #
    # # 初始化一个全为1的掩码
    # mask = torch.ones((3, 1, max_len))
    #
    # # 将填充位置的掩码设置为0
    # mask[:, :, valid_length:] = 0
    #
    # print(mask)
