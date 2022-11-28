import random
from model import SentenceTrainerA
from config import Config
from config import sentence_word_list
from utils import load_data, save_data, visualize_eb
from data_format import sentence_a_data_gen
import torch
import torch.nn as nn
import hiddenlayer as hl
import string

torch.manual_seed(Config.seed)
random.seed(Config.seed)


def make_model():
    model = SentenceTrainerA(vocab=Config.s_vocab, d_model=Config.s_eb_dim, h=Config.h,
                             N=Config.N_encoder, padding_size=Config.padding_size_a)
    return model


def train():
    sentence_a_data_gen()
    raw_data = load_data("./data/sentence2vec-A/train_vec({}).pk".format(Config.version))

    model = make_model()  # 模型

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失

    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    max_acc = 0  # 预测正确率，用于early-stop
    epoch_num = 10  # 训练周期数
    batch_size = 64  # batch大小
    total_step = int(len(raw_data) * 10 / batch_size + 1)  # 训练总步数
    step_count = 0  # 记录当前batch的训练信息
    count = 0
    loss = 0
    for epoch in range(epoch_num):
        random.shuffle(raw_data)  # 乱序
        sentences = torch.tensor(raw_data)  # 转化为tensor
        for sentence in sentences:
            _, x = model(sentence)
            loss = loss + loss_func(x, sentence)
            count += 1
            if count % batch_size == 0:  # batch大小为64
                optimizer.zero_grad()  # 梯度归0
                loss.backward()  # 损失反向传播
                optimizer.step()  # 参数优化

                step_count += 1
                history.log(step_count, loss=loss.item() / batch_size)  # 以下为训练可视化和模型保存
                canvas.draw_plot(history["loss"])
                canvas.save("./data/sentence2vec-A/training({}).png".format(Config.version))
                print("epoch {}/{}, step {}/{}, loss {}".format(epoch + 1, epoch_num,
                                                                step_count, total_step,
                                                                loss.item() / batch_size))

                loss = 0
                count = 0

        l_acc, s_acc, acc = eval_model(show_result=False, model=model)  # early-stop
        if s_acc > max_acc:  # 倾向于用结构预测正确率评估
            max_acc = s_acc
            print("epoch: {},  (l_acc, s_acc, acc): ({}, {}, {})".format(epoch + 1, l_acc, s_acc, acc))
            save_data(model, "./data/sentence2vec-A/model({}).pk".format(Config.version))


def eval_model(show_result=False, model=None):
    test_vec = load_data("./data/sentence2vec-A/test_vec({}).pk".format(Config.version))
    if model is None:
        model = load_data("./data/sentence2vec-A/model({}).pk".format(Config.version))

    ground_truth = []  # 真实值
    predict = []  # 预测值
    for i in range(len(test_vec)):
        ground_truth.append([])
        predict.append([])
        for w in test_vec[i]:  # 实值转化为字符
            ground_truth[i].append(sentence_word_list[w])

        _, x = model(torch.tensor(test_vec[i]))
        x = torch.softmax(x, dim=-1)
        for w in x:
            predict[i].append(sentence_word_list[torch.max(w, 0)[1]])

    letter = 0
    structure = 0
    letter_right = 0
    structure_right = 0
    letters = list(string.ascii_letters)
    for i in range(len(ground_truth)):
        j = 0
        while j < len(ground_truth[i]) and j < len(predict[i]) and ground_truth[i][j] != "padding":
            if ground_truth[i][j] not in letters:  # 结构预测正确率
                structure += 1
                if ground_truth[i][j] == predict[i][j]:
                    structure_right += 1
            else:  # 字母预测正确率
                letter += 1
                if ground_truth[i][j] == predict[i][j]:
                    letter_right += 1
            j += 1

        if show_result:
            print(ground_truth[i])
            print(predict[i])
            print()
    if show_result:
        print("Letter accuracy: {}({}/{})".format(letter_right / letter, letter_right, letter))
        print("Structure accuracy: {}({}/{})".format(structure_right / structure, structure_right, structure))
        print("Total accuracy: {}({}/{})".format((structure_right + letter_right) / (structure + letter),
                                                 structure_right + letter_right, structure + letter))

    return letter_right / letter, structure_right / structure, (structure_right + letter_right) / (structure + letter)


def eval_word_emb():
    model = load_data("./data/sentence2vec-A/model({}).pk".format(Config.version))
    embedding = []
    label = []
    for i in range(3, len(sentence_word_list)):
        embedding.append(model.sentence2vec.embedding[0](torch.tensor(i)).detach().numpy().tolist())
        label.append(sentence_word_list[i])

    visualize_eb(embedding, label, dim=2, use_pca=True)  # 可视化嵌入效果


if __name__ == '__main__':
    # train()
    eval_model(show_result=True)
    # eval_word_emb()
