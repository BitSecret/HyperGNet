import os
import random
from utility import load_data, save_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import hiddenlayer as hl
import torch.utils.data as data
import numpy as np
import scipy

random.seed(3407)
torch.manual_seed(3407)  # 随机数种子
torch.set_printoptions(precision=4, sci_mode=False, linewidth=1000)  # tensor输出设置
sentence_word_list = ["padding", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
                      "r", "s", "t", "u", "v", "w", "x", "y", "z",
                      "+", "-", "*", "/", "^", "@", "#", "$", "(", ")",
                      "1", "2", "3", "4", "5", "6", "7"]

re_map = {"1": "nums", "2": "ll_", "3": "ma_", "4": "as_", "5": "pt_", "6": "at_", "7": "f_",
          "@": "sin", "#": "cos", "$": "tan"}    # 映射回原始的符号


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pos=True, max_len=5000):
        self.pos = pos
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pos:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # // 是取整除的意思
        self.h = h
        self.linear = [nn.Linear(d_model, d_model, bias=False) for _ in range(4)]  # 4个线性层

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
        x = torch.matmul(scores, value)  # 结果

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # final linear
        x = self.linear[-1](x)
        return x


class Generator(nn.Module):

    def __init__(self, padding_size, d_model, vocab):
        super(Generator, self).__init__()
        self.padding_size = padding_size
        self.d_model = d_model
        self.linear = nn.Sequential(
            nn.Linear(d_model, int(padding_size * d_model / 2), bias=True),
            nn.Linear(int(padding_size * d_model / 2), padding_size * d_model, bias=True)
        )  # 句子embedding到词列表embedding
        self.output = nn.Linear(d_model, vocab)  # 词embedding升维到vocab

    def forward(self, x):
        batch_size = x.size(0)
        s = torch.mean(x, dim=1)
        x = self.linear(s)
        x = x.contiguous().view(batch_size, self.padding_size, self.d_model)
        x = self.output(x)
        return s, x


class Encoder(nn.Module):

    def __init__(self, embedding, attentions, generator):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.attentions = attentions
        self.generator = generator

    def forward(self, x):
        x = self.embedding(x)
        for attention in self.attentions:
            x = attention(x)
        s, x = self.generator(x)
        return s, x


def make_model(vocab, h, N, padding_size, d_model, pos):
    model = Encoder(nn.Sequential(Embedding(vocab, d_model), PositionalEncoding(d_model, pos)),
                    [MultiHeadAttention(h, d_model) for _ in range(N)],
                    Generator(padding_size, d_model, vocab))
    return model


def main():
    model = make_model(vocab=16, h=2, N=3, padding_size=5, d_model=8, pos=True)
    vec = torch.tensor([[1, 2, 3, 0, 0],
                        [0, 1, 2, 3, 0]])  # 第1批 2个维度为6的vec
    s, x = model(vec)
    print(s)
    print(x)


def pretrain():
    raw_data = torch.tensor(load_data("STS-data/500_padding.pk"))
    data_loader = data.DataLoader(
        dataset=data.TensorDataset(raw_data, raw_data),
        batch_size=32,
        shuffle=True,
        num_workers=1
    )

    model = make_model(vocab=len(sentence_word_list), h=4, N=3, padding_size=36, d_model=32, pos=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失

    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    step_num = int(len(raw_data) / 32) + 1
    for epoch in range(100):  # 训练
        for step, (b_x, b_y) in enumerate(data_loader):
            s, x = model(b_x)
            loss = 0
            for i in range(len(x)):
                loss = loss + loss_func(x[i], b_y[i])
            optimizer.zero_grad()  # 梯度归0
            loss.backward()  # 损失反向传播
            optimizer.step()  # 参数优化

            history.log(epoch * step_num + step, step_loss=loss.item())  # 以下为训练可视化和模型保存
            canvas.draw_plot(history["step_loss"])
            canvas.save("./STS-train/pretrain_loss.png")
            print("epoch {}, step {}/{}, loss {}".format(epoch, step, step_num, loss.item()))

    save_data(model.state_dict(), "./train/STS-pretrain.model")


def train():
    raw_data = torch.tensor(load_data("STS-data/500_padding.pk"))
    data_loader = data.DataLoader(
        dataset=data.TensorDataset(raw_data, raw_data),
        batch_size=32,
        shuffle=True,
        num_workers=1
    )

    model = make_model(vocab=len(sentence_word_list), h=4, N=3, padding_size=36, d_model=32, pos=True)
    model.load_state_dict(load_data("STS-train/pretrain.model"))  # 载入预训练的模型

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    pre_loss = nn.CrossEntropyLoss()  # 交叉熵损失
    const_loss = nn.CosineEmbeddingLoss()  # cos损失
    w = 0.9  # 负类损失所占权重

    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    step_num = int(len(raw_data) / 32) + 1
    tgt = torch.Tensor([-1])
    for epoch in range(100):  # 训练
        for step, (b_x, b_y) in enumerate(data_loader):
            s, x = model(b_x)

            loss_p = 0    # 计算损失
            loss_c = 0
            for i in range(len(x)):
                loss_p = loss_p + pre_loss(x[i], b_y[i])
            for i in range(int(len(s) / 2)):
                loss_c = loss_c + const_loss(s[i * 2].view(1, 32), s[i * 2 + 1].view(1, 32), tgt)
            loss = w * loss_c + (1 - w) * loss_p

            optimizer.zero_grad()  # 梯度归0
            loss.backward()  # 损失反向传播
            optimizer.step()  # 参数优化

            history.log(epoch * step_num + step, step_loss=loss.item())  # 以下为训练可视化和模型保存
            canvas.draw_plot(history["step_loss"])
            canvas.save("./STS-train/train_loss.png")
            print("epoch {}, step {}/{}, loss {}".format(epoch, step, step_num, loss.item()))

    save_data(model.state_dict(), "STS-train/train.model")


def eval_rebuilt():
    # sim = load_data("./STS-data/500_sim.pk")

    raw_data = load_data("STS-data/500.pk")    # 原始数据
    data_vec = torch.tensor(load_data("STS-data/500_padding.pk"))    # 原始数据向量形式
    output_data = []
    model = make_model(vocab=len(sentence_word_list), h=4, N=3, padding_size=36, d_model=32, pos=True)
    model.load_state_dict(load_data("STS-train/train.model"))  # 载入预训练的模型
    s, x = model(data_vec)
    x = torch.softmax(x, dim=-1)
    for i in range(len(x)):
        output_data.append([])
        for j in range(len(x[i])):
            output_data[i].append(sentence_word_list[torch.max(x[i][j], 0)[1]])

    for i in range(len(raw_data)):    # 重新将符号替换回去
        for j in range(len(raw_data[i])):
            if raw_data[i][j] in ["1", "2", "3", "4", "5", "6", "7", "@", "#", "$"]:
                raw_data[i][j] = re_map[raw_data[i][j]]
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            if output_data[i][j] in ["1", "2", "3", "4", "5", "6", "7", "@", "#", "$"]:
                output_data[i][j] = re_map[output_data[i][j]]

    for i in range(len(raw_data)):
        print(raw_data[i])
        print(output_data[i])
        print()


def eval_similarity():
    sim = load_data("./STS-data/500_sim.pk")
    if "500_sim_output.pk" not in os.listdir("./STS-data/"):
        sim_output = np.zeros((500, 500))
        data_vec = torch.tensor(load_data("STS-data/500_padding.pk"))  # 原始数据向量形式
        model = make_model(vocab=len(sentence_word_list), h=4, N=3, padding_size=36, d_model=32, pos=True)
        model.load_state_dict(load_data("STS-train/train.model"))  # 载入预训练的模型
        s, x = model(data_vec)
        for i in range(len(s)):
            for j in range(len(s)):
                sim_output[i][j] = torch.cosine_similarity(s[i], s[j], dim=0).item()
        save_data(sim_output, "./STS-data/500_sim_output.pk")
    else:
        sim_output = load_data("./STS-data/500_sim_output.pk")

    # 以下是各种相关性度量
    corr_sum = 0
    for i in range(len(sim)):
        corr = scipy.stats.pearsonr(sim[i], sim_output[i])[0]
        corr_sum += corr
    print("Pearson corr: {}".format(corr_sum / len(sim)))

    corr_sum = 0
    for i in range(len(sim)):
        corr = scipy.stats.spearmanr(sim[i], sim_output[i])[0]
        corr_sum += corr
    print("Spearmanr corr: {}".format(corr_sum / len(sim)))

    corr_sum = 0
    for i in range(len(sim)):
        corr = scipy.stats.kendalltau(sim[i], sim_output[i])[0]
        corr_sum += corr
    print("Kendalltau corr: {}".format(corr_sum / len(sim)))


if __name__ == '__main__':
    eval_rebuilt()
