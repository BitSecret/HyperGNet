import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# 将嵌入的效果，降维后可视化
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

    if dim == 2:    # 可视化
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
