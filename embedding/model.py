from config import Config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(Config.seed)


class Embedding(nn.Module):
    def __init__(self, vocab, d_model, padding=False):
        super(Embedding, self).__init__()
        self.d_model = d_model
        if padding:
            self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)  # 默认0位置是填充的字符
        else:
            self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 谓词embedding预训练
class PredicateTrainer(nn.Module):
    def __init__(self, vocab, d_model):
        super(PredicateTrainer, self).__init__()
        self.predicate2vec = Embedding(vocab, d_model)
        self.output = nn.Linear(in_features=d_model, out_features=vocab, bias=False)

    def forward(self, x):
        return self.output(self.predicate2vec(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0  # 至于为什么要是2的倍数，参考位置编码的公式
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + Variable(self.pe[0:x.size(0)], requires_grad=False)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # // 是取整除的意思
        self.h = h
        self.linear = [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]  # 4个线性层

    def forward(self, q, k, v):
        # 把x分别经过一层Linear变换得到QKV，tensor size不变
        query, key, value = [l(x) for l, x in zip(self.linear, (q, k, v))]

        # 将QKV的d_model维向量分解为h * d_k
        query, key, value = [x.view(-1, self.h, self.d_k).transpose(0, 1) for x in (query, key, value)]

        # Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # 注意力评分
        scores = F.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)  # 结果

        # "Concat" using a view and apply a final linear.
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k)

        # final linear
        x = self.linear[-1](x)
        return x


class Encoder(nn.Module):

    def __init__(self, vocab, d_model, h, N):
        super(Encoder, self).__init__()
        self.embedding = nn.Sequential(Embedding(vocab, d_model, padding=True), PositionalEncoding(d_model))
        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]

    def forward(self, x):
        x = self.embedding(x)
        for attention in self.attentions:
            x = attention(x, x, x)
        s = torch.mean(x, dim=0)
        return s


class ReBuilderA(nn.Module):

    def __init__(self, vocab, d_model, padding_size):
        super(ReBuilderA, self).__init__()
        self.d_model = d_model
        self.padding_size = padding_size
        self.linear = nn.Sequential(
            nn.Linear(d_model, int(padding_size * d_model / 2), bias=True),
            nn.Linear(int(padding_size * d_model / 2), padding_size * d_model, bias=True)
        )  # 句子embedding到词列表embedding
        self.output = nn.Linear(d_model, vocab)  # 词embedding升维到vocab

    def forward(self, s):
        x = self.linear(s)
        x = x.contiguous().view(self.padding_size, self.d_model)
        x = self.output(x)
        return x


# 个体词embedding预训练 方案A
class SentenceTrainerA(nn.Module):

    def __init__(self, vocab, d_model, h, N, padding_size):
        super(SentenceTrainerA, self).__init__()
        self.sentence2vec = Encoder(vocab, d_model, h, N)
        self.rebuilder = ReBuilderA(vocab, d_model, padding_size)

    def forward(self, x):
        s = self.sentence2vec(x)
        x = self.rebuilder(s)
        return s, x


class ResNet(nn.Module):

    def __init__(self, d_model):
        super(ResNet, self).__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.a_2 = nn.Parameter(torch.ones(d_model))    # layer norm
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-6

    def forward(self, x):
        x = x + F.relu(self.linear(x))    # res connection
        mean = x.mean(-1, keepdim=True)    # layer norm
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class ReBuilderB(nn.Module):
    def __init__(self, d_model, vocab, N):
        super(ReBuilderB, self).__init__()
        self.input = nn.Linear(d_model * 2, d_model)
        # self.input = nn.Sequential(nn.Linear(d_model * 2, int(d_model * 1.5)),
        #                            nn.Linear(int(d_model * 1.5), d_model))
        self.hidden = [ResNet(d_model) for _ in range(N)]
        self.output = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = self.input(x)
        for resnet in self.hidden:
            x = resnet(x)
        x = self.output(x)
        return x


# 个体词embedding预训练 方案B
class SentenceTrainerB(nn.Module):

    def __init__(self, vocab, d_model, h, N_encoder, N_decoder):
        super(SentenceTrainerB, self).__init__()
        self.sentence2vec = Encoder(vocab, d_model, h, N_encoder)
        self.rebuilder = ReBuilderB(d_model, vocab, N_decoder)

    def forward(self, x):
        s_input = self.sentence2vec(x[0])
        s_output = self.sentence2vec(x[1])
        s = torch.concat((s_input, s_output), dim=0)    # 拼接起来
        w = self.rebuilder(s)
        return s_input, w
