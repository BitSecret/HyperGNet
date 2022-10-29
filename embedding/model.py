from utility import save_data
from gen_data import one_hot_for_predicate, one_hot_for_sentence
from gen_data import predicate_word_list, sentence_word_list
import torch
import torch.utils.data as data
import torch.nn as nn
import hiddenlayer as hl
torch.manual_seed(3407)    # 随机数种子
dim = 64    # 谓词、定理和个体词的嵌入向量维度
p_one_hot_dim = len(predicate_word_list)
s_one_hot_dim = len(sentence_word_list)


class Word2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Word2Vec, self).__init__()
        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.output = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=False)

    def forward(self, x):
        h = self.input(x)
        u = self.output(h)
        return u


def train_predicate():
    x, y = one_hot_for_predicate()
    print("predicate: {}".format(len(x)))
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    data_loader = data.DataLoader(
        dataset=data.TensorDataset(x, y),
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    model = Word2Vec(p_one_hot_dim, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)    # 优化器
    loss_func = nn.CrossEntropyLoss()    # 损失函数，默认先softmax，不需要在模型中再添加softmax
    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    step_num = int(len(x) / 64) + 1
    for epoch in range(10):    # 训练
        for step, (b_x, b_y) in enumerate(data_loader):
            u = model(b_x)
            loss = loss_func(u, b_y)    # 损失
            optimizer.zero_grad()    # 梯度归0
            loss.backward()    # 损失反向传播
            optimizer.step()    # 参数优化
            history.log(epoch * step_num + step, step_loss=loss.item())    # 以下为训练可视化和模型保存
            canvas.draw_plot(history["step_loss"])
            canvas.save("./train/predicate_training_loss.png")
            print("epoch {}, step {}/{}, loss {}".format(epoch, step, step_num, loss.item()))

    save_data(model.state_dict(), "./train/predicate.model")


def train_sentence():
    x, y = one_hot_for_sentence()
    print("sentence: {}".format(len(x)))

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    data_loader = data.DataLoader(
        dataset=data.TensorDataset(x, y),
        batch_size=128,
        shuffle=True,
        num_workers=1
    )

    model = Word2Vec(s_one_hot_dim, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 损失函数，默认先softmax，不需要在模型中再添加softmax
    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    step_num = int(len(x) / 128) + 1
    for epoch in range(10):  # 训练
        for step, (b_x, b_y) in enumerate(data_loader):
            u = model(b_x)
            loss = loss_func(u, b_y)  # 损失
            optimizer.zero_grad()  # 梯度归0
            loss.backward()  # 损失反向传播
            optimizer.step()  # 参数优化
            history.log(epoch * step_num + step, step_loss=loss.item())  # 以下为训练可视化和模型保存
            canvas.draw_plot(history["step_loss"])
            canvas.save("./train/sentence_training_loss.png")
            print("epoch {}, step {}/{}, loss {}".format(epoch, step, step_num, loss.item()))

    save_data(model.state_dict(), "./train/sentence.model")
