import torch
import torch.nn as nn
import copy
import math
from s_module import clones, show
torch.manual_seed(3407)  # 随机数种子
torch.set_printoptions(precision=5, sci_mode=False, linewidth=1000)  # tensor输出设置


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # // 是取整除的意思
        self.h = h
        self.linear = clones(nn.Linear(d_model, d_model, bias=False), 4)  # 4个线性层
        for l in self.linear:    # 权重初始化
            torch.nn.init.xavier_uniform_(l.weight)

    def forward(self, x):
        batch_size = x.size(0)

        # 1.把x分别经过一层Linear变换得到Q, K, V，tensor size不变
        query = self.linear[0](x)
        key = self.linear[1](x)
        value = self.linear[2](x)

        # 2.将Q, K, V的d_model维向量分解为h * d_k
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 3.对所有的attention head计算注意力评分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # 注意力评分
        scores = torch.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)  # 结果

        # 4.拼接所有的attention head
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 5.最后一层线性变换
        x = self.linear[-1](x)

        return x


def test1():
    b = 1    # batches
    n = 4    # N
    d = 64    # dim
    h = 16    # attention head

    x = torch.randn((b, n, d))

    show("输入", x)
    for i in range(5):
        attn = MultiHeadAttention(h, d)
        x = attn(x)
        show("第{}次attention".format(i + 1), x)


def _compute(result, r, h, i, data):    # 计算data的方差并输出
    var = torch.sum(torch.var(data, dim=-2, unbiased=False), dim=-1).data[0]
    print("r={}, h={}, i={}, mean:{:.6f}".format(r, h, i, var))
    if (h, i) in result.keys():
        result[(h, i)] += var
    else:
        result[(h, i)] = var


def test2():
    b = 1  # batches
    n = 4  # N
    d = 120  # dim
    repeat = 20  # 重复实验次数
    result = {}

    for r in range(repeat):
        mat = torch.randn((b, n, d))
        for h in range(1, d + 1):
            if d % h == 0:    # 如果是合法的头的数量
                x = copy.deepcopy(mat)
                _compute(result, r + 1, h, 0, x)
                for i in range(4):
                    x = MultiHeadAttention(h, d)(x)
                    _compute(result, r + 1, h, i + 1, x)
                print()

    for key in result.keys():
        print("h={}, i={}, mean={:.6f}".format(key[0], key[1], result[key] / repeat))


def _self_attention(x):
    scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.size(-1))
    scores = torch.softmax(scores, dim=-1)
    return torch.matmul(scores, x)  # 结果


def _positional_embedding(d):
    pe = torch.zeros(1000, d)
    position = torch.arange(0, 1000).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


def test3(add_pe=False):
    n = 4    # N
    d = 8    # dim

    x = torch.randn((n, d))
    y = copy.deepcopy(x)
    y[2] = copy.deepcopy(x[3])
    y[3] = copy.deepcopy(x[2])
    show("输入", x)
    show("输入", y)

    if add_pe:
        pe = _positional_embedding(d)
        x = x + pe[0][0:4]
        y = y + pe[0][0:4]
        show("输入+PE", x)
        show("输入+PE", y)

    for i in range(6):
        x = _self_attention(x)
        y = _self_attention(y)
        show("第{}次attention, x".format(i + 1), x)
        show("第{}次attention, y".format(i + 1), y)


test3(add_pe=False)
