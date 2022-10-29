import os
from utility import save_data, load_data
from gen_data import one_hot_for_predicate, one_hot_for_sentence
from gen_data import predicate_word_list, sentence_word_list
from model import Word2Vec
import torch
import torch.utils.data as data
import torch.nn as nn
import hiddenlayer as hl
torch.manual_seed(3407)    # 随机数种子
dim = 64    # 谓词、定理和个体词的嵌入向量维度
p_one_hot_dim = len(predicate_word_list)
s_one_hot_dim = len(sentence_word_list)
solution_data = "../reasoner/solution_data/g3k_normal/"


def train_predicate():
    x, y = one_hot_for_predicate(solution_data)
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


def get_predicate_embedding():
    if "predicate.emb" in os.listdir("./output/"):
        return load_data("./output/predicate.emb")

    predicate_embedding = {}
    state_dict = load_data("./train/predicate.model")
    weight = state_dict["input.weight"].T
    for i in range(len(predicate_word_list)):
        predicate_embedding[predicate_word_list[i]] = weight[i]
    save_data(predicate_embedding, "./output/predicate.emb")

    return predicate_embedding


def train_sentence():
    x, y = one_hot_for_sentence(solution_data)
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


def get_sentence_embedding():
    if "sentence.emb" in os.listdir("./output/"):
        return load_data("./output/sentence.emb")

    sentence_embedding = {}
    state_dict = load_data("./train/sentence.model")
    weight = state_dict["input.weight"].T
    for i in range(len(sentence_word_list)):
        sentence_embedding[sentence_word_list[i]] = weight[i]
    save_data(sentence_embedding, "./output/sentence.emb")

    return sentence_embedding


if __name__ == '__main__':
    p_e = get_predicate_embedding()
    for key in p_e.keys():
        print(key, end=": ")
        print(p_e[key])

    s_e = get_sentence_embedding()
    for key in s_e.keys():
        print(key, end=": ")
        print(s_e[key])
