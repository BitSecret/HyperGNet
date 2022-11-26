import torch
import torch.nn as nn
import copy
import math
torch.manual_seed(3407)  # 随机数种子
torch.set_printoptions(precision=5, sci_mode=False, linewidth=1000)  # tensor输出设置


class Attention:

    def __init__(self, n, d, h):
        """
        测试Attention的类
        :param n: 样本的数量
        :param d: 样本嵌入的维度
        :param h: 多头注意力头的数量
        """
        self.n = n
        self.d = d
        self.h = h
        self.x = torch.randn((self.n, self.d))

    @staticmethod
    def show(text, data):
        print("{}: {}".format(text, data.size()))
        print(data)

    def simple_attn(self):
        print("\033[32msimple_attention:\033[0m")
        x = copy.deepcopy(self.x)
        Attention.show("x", x)
        for i in range(5):
            x = Attention._self_attn(x)
            Attention.show("after {} attn".format(i + 1), x)

    @staticmethod
    def _self_attn(x):
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.size(-1))
        scores = torch.softmax(scores, dim=-1)
        return torch.matmul(scores, x)  # 结果

    def multi_head_attn(self):
        print("\033[32mmulti_head_attention:\033[0m")
        x = copy.deepcopy(self.x)
        Attention.show("x", x)
        for i in range(5):
            linear = Attention._get_linear(self.d)    # 每次linear都用新的
            x = Attention._multi_self_attn(x, self.h, linear)
            Attention.show("after {} attn".format(i + 1), x)

    @staticmethod
    def _multi_self_attn(x, h, linear):
        d_k = x.size(1) // h

        # 1.把x分别经过一层Linear变换得到Q, K, V，tensor size不变
        query = linear[0](x)
        key = linear[1](x)
        value = linear[2](x)

        # 2.将Q, K, V的d_model维向量分解为h * d_k
        query = query.view(-1, h, d_k).transpose(0, 1)
        key = key.view(-1, h, d_k).transpose(0, 1)
        value = value.view(-1, h, d_k).transpose(0, 1)

        # 3.对所有的attention head计算注意力评分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # 注意力评分
        scores = torch.softmax(scores, dim=-1)
        x = torch.matmul(scores, value)  # 结果

        # 4.拼接所有的attention head
        x = x.transpose(0, 1).contiguous().view(-1, h * d_k)

        # 5.最后一层线性变换
        x = linear[-1](x)
        return x  # 结果

    @staticmethod
    def _get_linear(d):
        linear = [nn.Linear(d, d, bias=False) for _ in range(4)]
        for i in range(4):
            nn.init.xavier_uniform_(linear[i].weight)
        return linear

    def random_attn(self):
        print("\033[32mrandom_attention:\033[0m")
        x = copy.deepcopy(self.x)
        Attention.show("x", x)
        for i in range(8):
            x = Attention._random_score(x)
            Attention.show("after {} attn".format(i + 1), x)

    @staticmethod
    def _random_score(x):
        scores = torch.softmax(torch.randn((x.size(0), x.size(0))), dim=-1)
        return torch.matmul(scores, x)  # 结果

    def pe(self, add_pe=False):
        print("\033[32mpositional embedding:\033[0m")
        x = copy.deepcopy(self.x)
        y = copy.deepcopy(self.x)
        y[-1] = copy.deepcopy(x[-2])
        y[-2] = copy.deepcopy(x[-1])
        Attention.show("x", x)
        Attention.show("y", y)
        if add_pe:
            pe = Attention._positional_embedding(x.size(-1))
            x = x + pe[0][0:4]
            y = y + pe[0][0:4]
            Attention.show("x+PE", x)
            Attention.show("y+PE", y)

        for i in range(5):
            linear = Attention._get_linear(self.d)    # 每次linear都用新的
            x = Attention._multi_self_attn(x, self.h, linear)
            y = Attention._multi_self_attn(y, self.h, linear)
            Attention.show("after {} attn, x".format(i + 1), x)
            Attention.show("after {} attn, y".format(i + 1), y)

    @staticmethod
    def _positional_embedding(d):
        pe = torch.zeros(1000, d)
        position = torch.arange(0, 1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


def main():
    model = Attention(n=4, d=4, h=2)
    model.simple_attn()
    model.multi_head_attn()
    model.random_attn()
    model.pe(add_pe=False)
    model.pe(add_pe=True)

    # vec1 = torch.tensor([100, 101, 102, 104], dtype=torch.float32)   # 第1批 2个维度为3的vec
    # vec2 = torch.tensor([3, 4, 5, 7], dtype=torch.float32)   # 第1批 2个维度为3的vec
    # vec3 = torch.tensor([5, 6, 7, 9], dtype=torch.float32)   # 第1批 2个维度为3的vec
    # print(torch.softmax(vec1, dim=0))
    # print(torch.softmax(vec2, dim=0))
    # print(torch.softmax(vec3, dim=0))


if __name__ == '__main__':
    main()
