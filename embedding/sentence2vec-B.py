import random
from model import SentenceTrainerB
from config import Config
from config import sentence_word_list
from utils import load_data, save_data, visualize_eb
from data_format import sentence_b_data_gen
import torch
import torch.nn as nn
import hiddenlayer as hl
import string
torch.manual_seed(Config.seed)
random.seed(Config.seed)


def make_model():
    model = SentenceTrainerB(vocab=Config.s_vocab, d_model=Config.s_eb_dim, h=Config.h,
                             N_encoder=Config.N_encoder, N_decoder=Config.N_decoder)
    return model


def train():
    sentence_b_data_gen()
    x = torch.tensor(load_data("./data/sentence2vec-B/train_vec_x({}).pk".format(Config.version)))
    y = torch.tensor(load_data("./data/sentence2vec-B/train_vec_y({}).pk".format(Config.version)))

    model = make_model()  # 模型

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失

    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    max_acc = 0  # 预测正确率，用于early-stop
    epoch_num = 10  # 训练周期数
    batch_size = 128    # batch大小
    total_step = int(len(x) * epoch_num / batch_size + 1)    # 训练总步数
    step_count = 0    # 记录当前batch的训练信息
    count = 0
    loss = 0
    for epoch in range(epoch_num):
        shuffle_ids = [i for i in range(len(x))]
        random.shuffle(shuffle_ids)    # 乱序
        for i in shuffle_ids:
            _, w = model(x[i])
            loss = loss + loss_func(w, y[i])
            count += 1
            if count % batch_size == 0:  # 到达batch大小
                optimizer.zero_grad()  # 梯度归0
                loss.backward()  # 损失反向传播
                optimizer.step()  # 参数优化

                step_count += 1
                history.log(step_count, loss=loss.item() / batch_size)  # 以下为训练可视化和模型保存
                canvas.draw_plot(history["loss"])
                canvas.save("./data/sentence2vec-B/training({}).png".format(Config.version))
                print("epoch {}/{}, step {}/{}, loss {}".format(epoch + 1, epoch_num,
                                                                step_count, total_step,
                                                                loss.item() / batch_size))

                loss = 0
                count = 0

        l_acc, s_acc, acc = eval_model(show_result=False, model=model)  # early-stop
        if s_acc > max_acc:  # 倾向于用结构预测正确率评估
            max_acc = s_acc
            print("epoch: {},  (l_acc, s_acc, acc): ({}, {}, {})".format(epoch + 1, l_acc, s_acc, acc))
            save_data(model, "./data/sentence2vec-B/model({}).pk".format(Config.version))


def eval_model(show_result=False, model=None):
    test_vec = load_data("./data/sentence2vec-B/test_vec({}).pk".format(Config.version))
    if model is None:
        model = load_data("./data/sentence2vec-B/model({}).pk".format(Config.version))

    ground_truth = []  # 真实值
    predict = []  # 预测值
    for i in range(len(test_vec)):
        ground_truth.append([])
        predict.append(["<start>"])
        for w in test_vec[i][0]:  # 实值转化为字符
            ground_truth[i].append(sentence_word_list[w])

        decoded = 0
        while True:    # 循环解码
            _, w = model(torch.tensor(test_vec[i]))
            w = torch.max(torch.softmax(w, dim=-1), 0)[1]    # 解码出的字符的实值表示
            test_vec[i][1][decoded + 1] = w
            predict[i].append(sentence_word_list[w])
            decoded += 1
            if w == sentence_word_list.index("<end>"):    # 解码出停止符了，停止解码
                break
            if decoded == Config.padding_size_b - 2:    # 到最大长度了，停止解码
                predict[i].append("<end>")
                break

    letter = 0
    structure = 0
    letter_right = 0
    structure_right = 0
    letters = list(string.ascii_letters)
    for i in range(len(ground_truth)):
        j = 1    # 从位置1开始，<start>不算
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
    model = load_data("./data/sentence2vec-B/model({}).pk".format(Config.version))
    embedding = []
    label = []
    for i in range(3, len(sentence_word_list)):
        embedding.append(model.sentence2vec.embedding[0](torch.tensor(i)).detach().numpy().tolist())
        label.append(sentence_word_list[i])

    visualize_eb(embedding, label, dim=2, use_pca=True)  # 可视化嵌入效果


if __name__ == '__main__':
    train()
    # eval_model(show_result=True)
    # eval_word_emb()
