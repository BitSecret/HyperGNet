import random
from model import PredicateTrainer
from config import Config
from config import predicate_word_list
from utils import load_data, save_data, visualize_eb
from data_format import predicate_data_gen
import hiddenlayer as hl
import torch
import torch.nn as nn
import torch.utils.data as data
random.seed(Config.seed)
torch.manual_seed(Config.seed)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


def make_model():
    model = PredicateTrainer(vocab=Config.p_vocab, d_model=Config.p_eb_dim)
    model.apply(weights_init)
    return model


def train(path):
    predicate_data_gen()    # 先生成数据
    x = torch.tensor(load_data(path + "predicate_vec_x({}).pk".format(Config.version)),
                     dtype=torch.long)
    y = torch.tensor(load_data(path + "predicate_vec_y({}).pk".format(Config.version)),
                     dtype=torch.long)
    data_loader = data.DataLoader(
        dataset=data.TensorDataset(x, y),
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
    history = hl.History()  # 训练损失可视化
    canvas = hl.Canvas()

    step_num = int(len(x) / 64) + 1
    for epoch in range(10):  # 训练
        for step, (b_x, b_y) in enumerate(data_loader):
            output = model(b_x)
            loss = loss_func(output, b_y)  # 损失
            optimizer.zero_grad()  # 梯度归0
            loss.backward()  # 损失反向传播
            optimizer.step()  # 参数优化

            history.log(epoch * step_num + step, step_loss=loss.item())  # 以下为训练可视化和模型保存
            canvas.draw_plot(history["step_loss"])
            canvas.save(path + "training({}).png".format(Config.version))
            print("epoch {}, step {}/{}, loss {}".format(epoch, step, step_num, loss.item()))

    save_data(model, path + "model({}).pk".format(Config.version))


def eval_model(path):
    embedding = []
    label = []

    for i in load_data(path + "predicate({}).pk".format(Config.version)):     # 筛选出训练过的谓词
        label.append(i[0])
    label = list(set(label))

    model = load_data(path + "model({}).pk".format(Config.version))
    for i in label:    # 得到label的嵌入表示
        output = model.predicate2vec(torch.tensor(predicate_word_list.index(i), dtype=torch.long))
        embedding.append(output.detach().numpy().tolist())

    visualize_eb(embedding, label, dim=2, use_pca=True)    # 可视化嵌入效果


if __name__ == '__main__':
    data_path = "./data/predicate2vec/"
    train(path=data_path)
    eval_model(path=data_path)

