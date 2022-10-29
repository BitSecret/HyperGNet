from gen_data import get_predicate_embedding, get_sentence_embedding, gen_for_sentence, gen_for_predicate
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eval_embedding(evaluate_pre=True, dim=2, use_pca=True):
    if evaluate_pre:
        raw_data = gen_for_predicate("../reasoner/solution_data/g3k_normal/")
        embedding_data = get_predicate_embedding()
    else:
        raw_data = gen_for_sentence("../reasoner/solution_data/g3k_normal/")
        embedding_data = get_sentence_embedding()

    trained_word_list = []    # 筛选出训练过的word
    for i in raw_data:
        if i[0] not in trained_word_list:
            trained_word_list.append(i[0])
        if i[1] not in trained_word_list:
            trained_word_list.append(i[1])
    label = []
    vec = []

    for i in embedding_data.keys():
        if i in trained_word_list:    # 只有训练过才纳入表示
            label.append(i)
            vec.append(embedding_data[i].tolist())
    label = np.array(label)
    vec = np.array(vec)

    if use_pca:  # PCA 降维
        vec = PCA(n_components=dim).fit_transform(vec.data)
    else:  # TSNE 降维
        vec = TSNE(n_components=dim, learning_rate=10, init="random", random_state=3407).fit_transform(vec.data)

    if dim == 2:    # 可视化
        x = vec[:, 0]
        y = vec[:, 1]

        plt.figure(figsize=(12, 8))  # 可视化
        plt.scatter(x, y)
        for i in range(len(label)):
            plt.annotate(label[i], xy=(x[i], y[i]))

        plt.show()
    else:
        x = vec[:, 0]
        y = vec[:, 1]
        z = vec[:, 2]
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
