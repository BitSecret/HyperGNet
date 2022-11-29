import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import string
import Levenshtein
import time


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def visualize_eb(eb, label, dim=2, use_pca=True):
    """
    :param eb: 嵌入的向量，np形式
    :param label: 标签
    :param dim: 可视化维度
    :param use_pca: 使用pca还是tsne
    """
    if use_pca:  # PCA 降维
        eb = PCA(n_components=dim).fit_transform(np.array(eb))
    else:  # TSNE 降维
        eb = TSNE(n_components=dim, learning_rate=10, init="random", random_state=3407).fit_transform(np.array(eb))

    if dim == 2:  # 可视化
        x = eb[:, 0]
        y = eb[:, 1]

        plt.figure(figsize=(12, 8))  # 可视化
        plt.scatter(x, y)
        for i in range(len(label)):
            plt.annotate(label[i], xy=(x[i], y[i]))

        plt.show()
    else:
        x = eb[:, 0]
        y = eb[:, 1]
        z = eb[:, 2]
        fig = plt.figure(figsize=(12, 8))
        ax = Axes3D(fig, auto_add_to_figure=False)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_zlim(min(z), max(z))

        fig.add_axes(ax)

        for i in range(len(label)):
            ax.text(x[i], y[i], z[i], label[i],
                    fontsize=8,
                    bbox=dict(boxstyle="round", alpha=0.7))

        plt.show()


def eval_se_acc(target, predict, show_result=True):
    letters = list(string.ascii_letters)  # 大小写字母

    letter = 0
    structure = 0
    for i in range(len(target)):
        for j in range(len(target[i])):
            if target[i][j] in letters:
                letter += 1
            else:
                structure += 1

    letter_right = 0
    structure_right = 0
    for i in range(len(target)):
        j = 0
        while j < len(target[i]) and j < len(predict[i]):
            if target[i][j] == predict[i][j]:
                if target[i][j] in letters:
                    letter_right += 1
                else:
                    structure_right += 1
            j += 1

    l_acc = letter_right / letter
    s_acc = structure_right / structure
    acc = (structure_right + letter_right) / (structure + letter)

    if show_result:
        for i in range(len(target)):
            print(target[i])
            print(predict[i])
            print()
        print("Letter accuracy: {}({}/{})".format(l_acc, letter_right, letter))
        print("Structure accuracy: {}({}/{})".format(s_acc, structure_right, structure))
        print("Total accuracy: {}({}/{})".format(acc, structure_right + letter_right, structure + letter))

    return l_acc, s_acc, acc


def eval_se_acc_(target, predict, show_result=True):
    replace_map = {"sin": "1", "cos": "2", "tan": "3", "nums": "4", "ll_": "5", "ma_": "6", "as_": "7", "pt_": "8",
                   "at_": "9", "f_": "0"}
    for i in range(len(target)):
        for j in range(len(target[i])):
            if target[i][j] in replace_map:
                target[i][j] = replace_map[target[i][j]]
        target[i] = "".join(target[i])
        for j in range(len(predict[i])):
            if predict[i][j] in replace_map:
                predict[i][j] = replace_map[predict[i][j]]
        predict[i] = "".join(predict[i])

    score = 0
    len_sum = 0
    for i in range(len(target)):
        score += Levenshtein.ratio(target[i], predict[i]) * len(target[i])
        len_sum += len(target[i])

    if show_result:
        print("Editing distance: {}".format(score / len_sum))

    return score / len_sum


def log(path, s, time_append=False):
    with open(path + "log.txt", "a") as f:
        if time_append:
            s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "  " + s
        f.write(s + "\n")
