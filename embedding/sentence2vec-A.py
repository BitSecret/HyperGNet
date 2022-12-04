import random
from model import SentenceTrainerA
from config import Config
from config import sentence_word_list
from utils import load_data, save_data, visualize_eb, eval_se_acc, eval_se_ed, log
from data_format import sentence_a_data_gen
import torch
import torch.nn as nn
import hiddenlayer as hl
torch.manual_seed(Config.seed)
random.seed(Config.seed)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


def make_model():
    model = SentenceTrainerA(vocab=Config.s_vocab, d_model=Config.s_eb_dim, h=Config.h,
                             N=Config.N, padding_size=Config.padding_size_a)
    model.apply(weights_init)
    return model


def train(path, note=""):
    log(path, note, time_append=True)
    sentence_a_data_gen()
    raw_data = load_data(path + "train_vec({}).pk".format(Config.version))

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
                canvas.save(path + "training({}).png".format(Config.version))
                print("epoch {}/{}, step {}/{}, loss {}".format(epoch + 1, epoch_num,
                                                                step_count, total_step,
                                                                loss.item() / batch_size))

                loss = 0
                count = 0

        l_acc, s_acc, acc, l_ed, s_ed, ed = eval_acc(path=path, show_result=False, model=model)  # early-stop
        msg = "epoch: {},  (l_acc, s_acc, acc, l_ed, s_ed, ed): ({}, {}, {}, {}, {}, {})".format(epoch + 1,
                                                                                                 l_acc, s_acc, acc,
                                                                                                 l_ed, s_ed, ed)
        log(path, msg)
        print(msg)
        if s_acc > max_acc:  # 倾向于用结构预测正确率评估
            max_acc = s_acc
            save_data(model, path + "model({}).pk".format(Config.version))


def eval_acc(path, show_result=False, model=None):
    test_vec = load_data(path + "test_vec({}).pk".format(Config.version))
    if model is None:
        model = load_data(path + "model({}).pk".format(Config.version))

    ground_truth = []  # 真实值
    predict = []  # 预测值
    for i in range(len(test_vec)):
        ground_truth.append([])  # 真实值
        j = 0
        while j < len(test_vec[i]) and test_vec[i][j] != sentence_word_list.index("padding"):
            ground_truth[i].append(sentence_word_list[test_vec[i][j]])
            j += 1

        predict.append([])  # 预测值
        _, x = model(torch.tensor(test_vec[i]))
        x = torch.softmax(x, dim=-1)
        for w in x:
            w = torch.max(w, 0)[1]
            if w != sentence_word_list.index("padding"):
                predict[i].append(sentence_word_list[w])

    l_acc, s_acc, acc = eval_se_acc(ground_truth, predict, show_result=show_result)    # 计算准确率
    l_ed, s_ed, ed = eval_se_ed(ground_truth, predict, show_result=show_result)    # 编辑距离

    return l_acc, s_acc, acc, l_ed, s_ed, ed


def eval_word_emb(path):
    model = load_data(path + "model({}).pk".format(Config.version))
    embedding = []
    label = []
    for i in range(3, len(sentence_word_list)):
        embedding.append(model.sentence2vec.embedding[0](torch.tensor(i)).detach().numpy().tolist())
        label.append(sentence_word_list[i])

    visualize_eb(embedding, label, dim=2, use_pca=False)  # 可视化嵌入效果


if __name__ == '__main__':
    data_path = "./data/sentence2vec-A/"
    # train(path=data_path, note="sentence2vec-A training start.")
    # eval_acc(path=data_path, show_result=True)
    eval_word_emb(path=data_path)
