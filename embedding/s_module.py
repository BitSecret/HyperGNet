import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
torch.manual_seed(3407)  # 随机数种子
torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)  # tensor输出设置
debug = True


def clones(module, N):  # 这个方法得到的各模块权重都一样
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, embedding, layer, N, pooling, generator):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.layers = clones(layer, N)
        self.pooling = pooling
        self.generator = generator

    def forward(self, x):
        """Pass the input through each layer in turn."""
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pooling(x)
        x = self.generator(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, d_model, self_attn, feed_forward, p_drop):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 子层1 self-attention 模块
        self.dropout_attn = nn.Dropout(p_drop)  # 子层1 输出后的dropout
        self.norm1 = LayerNorm(d_model)  # 子层1之后的层归一化
        self.feed_forward = feed_forward  # 子层2 前馈模块
        self.dropout_fd = nn.Dropout(p_drop)  # 子层2 输出后的dropout
        self.norm2 = LayerNorm(d_model)  # 子层2之后的层归一化

    def forward(self, x):
        x = self.norm1(x + self.dropout_attn(self.self_attn(x)))  # 子层1输出、dropout、残差、layer norm
        x = self.norm2(x + self.dropout_fd(self.feed_forward(x)))  # 子层2输出、dropout、残差、layer norm
        return x


class LayerNorm(nn.Module):
    """Construct a layer_norm module (See citation for details)."""

    def __init__(self, d_model, eps=1e-6):
        """
        初始化函数

        :param d_model: 特征的数量，或者说编码的维度d_model
        :param eps: 为了计算时数值稳定所设的参数
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        show("子层输出-->dropout-->res-->layer_norm", x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, p_drop):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # // 是取整除的意思
        self.h = h
        self.linear = clones(nn.Linear(d_model, d_model, bias=False), 4)  # 4个线性层
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        batch_size = x.size(0)

        # 把x分别经过一层Linear变换得到QKV，tensor size不变
        query, key, value = [l(x)
                             for l, x in zip(self.linear, (x, x, x))]

        # 将QKV的d_model维向量分解为h * d_k
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)]

        # Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # 注意力评分
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)  # dropout
        x = torch.matmul(scores, value)  # 结果

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # final linear
        x = self.linear[-1](x)

        show("attention 子层", x)
        return x


class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, p_drop):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # Linear 默认带bias
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        show("feedforward 子层", x)
        return x


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        """
        token输入encoder和decoder之前的嵌入

        :param d_model: int, 嵌入的维度
        :param vocab: int, 词表中词汇的数量
        """
        super(Embedding, self).__init__()
        # nn.Embedding的作用: 随机初始化一个vocab*d_model的矩阵
        # 或许可以看作一个线性变换？将vocab维变换到d_model纬
        self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        """调用此函数得到x的embedding
        :param x: Tensor n, token的index
        :return: Tensor n*d_model, token的embedding
        """
        show("input_vec", x)
        x = self.lut(x) * math.sqrt(self.d_model)
        show("嵌入为{}维向量".format(self.d_model), x)
        return x


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, p_drop, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)
        show("加上位置编码", x)
        return x


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = F.softmax(self.linear(x), dim=-1)  # 注意，原code是log_softmax
        show("generator/output_vec", x)
        return x


class Pooling(nn.Module):

    def __init__(self, padding_size, d_model, p_drop):
        super(Pooling, self).__init__()
        self.padding_size = padding_size
        self.d_model = d_model
        self.pooling = nn.Linear(padding_size * d_model, d_model)  # 池化
        self.de_pooling = nn.Linear(d_model, padding_size * d_model)  # 复原
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 把x展开
        show("x_expand", x)
        pooling_x = self.dropout(self.pooling(x))
        show("pooling_x", pooling_x)
        x = self.de_pooling(pooling_x).view(batch_size, self.padding_size, -1)
        show("de_pooling_x", x)
        return x


def make_model(vocab, N=6, d_model=32, d_ff=128, h=4, p_drop=0.1, padding_size=64):
    model = Encoder(
        nn.Sequential(Embedding(d_model, vocab), PositionalEncoding(d_model, p_drop)),
        EncoderLayer(d_model, MultiHeadAttention(h, d_model, p_drop), FeedForward(d_model, d_ff, p_drop), p_drop),
        N,
        Pooling(padding_size, d_model, p_drop),
        Generator(d_model, vocab)
    )

    for p in model.parameters():  # 参数初始化， 但这个初始化会让padding不为0
        if p.dim() > 1:
            # print(p)
            nn.init.xavier_uniform_(p)

    return model


def show(des, data):
    if debug:
        print("{}: {}".format(des, data.size()))
        print(data)
        print()


if __name__ == '__main__':
    test_model = make_model(vocab=8, N=6, d_model=4, d_ff=8, h=2, p_drop=0.1, padding_size=4)
    vec = torch.tensor([[0, 1, 2, 3]])  # 第1批 2个维度为3的vec
    test_model(vec)




