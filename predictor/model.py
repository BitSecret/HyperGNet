import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(3407)    # 随机数种子
random.seed(3407)


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


class TheoremPredictor(nn.Module):

    def __init__(self, predicate2vec, sentence2vec, d_model, h, N, p_drop, d_ff, vocab_theo):
        super(TheoremPredictor, self).__init__()
        self.N = N
        self.predicate2vec = predicate2vec
        self.sentence2vec = sentence2vec

        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]    # attention
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]  # attention层 后的dropout
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]  # attention层 后的layer norm

        self.feedforwards = [FeedForward(d_model, d_ff) for _ in range(N)]  # ffd层
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]  # ffd层 后的dropout
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]  # ffd层 后的layer norm

        self.linear = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=vocab_theo, bias=False),
            nn.Linear(in_features=vocab_theo, out_features=vocab_theo, bias=False)
        )

    def forward(self, predicate, sentence):
        p_vec = self.predicate2vec(predicate)
        s_vec = self.sentence2vec(sentence)
        x = torch.cat((p_vec, s_vec), dim=0)

        for i in range(self.N):
            x = self.attentions[i](x[0], x[1:len(x)], x[1:len(x)])    # attention 层
            x = self.ln_attn[i](x + self.dp_attn[i](x))    # dropout、残差、layer norm
            x = self.feedforwards[i](x)    # 前馈层
            x = self.ln_ffd[i](x + self.dp_ffd[i](x))    # dropout、残差、layer norm

        x = self.linear(x)

        return x
