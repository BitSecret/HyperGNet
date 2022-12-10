import random
import os
import re
from config import Path, Config
from config import predicate_word_list, expr_word_list, sentence_word_list
from utils import load_data, save_data
random.seed(Config.seed)


def get_train_pids():  # 训练集、测试集划分
    if "train_pids({}).pk".format(Config.version) not in os.listdir("./data/raw_data/"):
        train_pids = random.sample(range(0, Config.problem_count + 1), int(Config.problem_count * Config.train_ratio))
        train_pids = sorted(train_pids)
        save_data(train_pids, "./data/raw_data/train_pids({}).pk".format(Config.version))
    else:
        train_pids = load_data("./data/raw_data/train_pids({}).pk".format(Config.version))
    return train_pids


def get_theo_graph():  # 超图转化为普通图
    if "theo_graph({}).pk".format(Config.version) not in os.listdir("./data/raw_data/"):
        train_pids = get_train_pids()
        graph = {}
        for pid in train_pids:
            one_item = {}
            data = load_data(Path.solution_data + "{}_hyper.pk".format(pid))
            for j in data:
                heads, middle, tails = j
                for head in heads:
                    if head not in one_item:
                        one_item[head] = [middle]
                    else:
                        one_item[head].append(middle)
                one_item[middle] = tails
            graph[pid] = one_item
        save_data(graph, "./data/raw_data/theo_graph({}).pk".format(Config.version))
    else:
        graph = load_data("./data/raw_data/theo_graph({}).pk".format(Config.version))
    return graph


def node_format(current_node, next_node):  # 提取谓词和定理
    data_item = []
    if isinstance(current_node, tuple):  # start_node_predicate
        data_item.append(current_node[0])
    else:  # start_node_theorem
        data_item.append(current_node.split("_")[0])

    if isinstance(next_node, tuple):  # end_node_predicate
        data_item.append(next_node[0])
    else:  # end_node_theorem
        data_item.append(next_node.split("_")[0])

    return data_item


def graph_walking(root_node, current_node, graph, depth, data):  # 递归随机游走
    if depth == Config.walking_depth:  # 游走到到预计深度，返回
        return

    if current_node in graph.keys():
        for next_node in graph[current_node]:
            data_item = node_format(current_node, next_node)
            if data_item[0] in ["-1", "-2", "-3"] or data_item[1] in ["-1", "-2", "-3"]:  # 求解方程、自动扩展不算步长，继续游走
                graph_walking(root_node, next_node, graph, depth, data)
            else:
                data_item = node_format(root_node, next_node)
                data.append(tuple(data_item))
                graph_walking(root_node, next_node, graph, depth + 1, data)


# 运行此函数即可得到 predicate2vec.py 的训练数据
def predicate_data_gen():
    if "predicate({}).pk".format(Config.version) in os.listdir("./data/predicate2vec/"):
        return

    graphs = get_theo_graph()
    data = []
    for pid in graphs:
        graph = graphs[pid]
        one_problem = []
        for s_node in graph.keys():  # 从每一个node开始随机游走
            if isinstance(s_node, tuple):
                if s_node[0] not in ["Premise", "Target"]:
                    graph_walking(s_node, s_node, graph, 0, one_problem)
            elif not s_node.startswith("-1") and not s_node.startswith("-2") and not s_node.startswith("-3"):
                graph_walking(s_node, s_node, graph, 0, one_problem)
        data += list(set(one_problem))

    x = []  # 转化为实值并分开网络的输入和输出
    y = []
    for i in data:
        x.append(predicate_word_list.index(i[0]))
        y.append(predicate_word_list.index(i[1]))

    save_data(data, "./data/predicate2vec/predicate({}).pk".format(Config.version))
    save_data(x, "./data/predicate2vec/predicate_vec_x({}).pk".format(Config.version))
    save_data(y, "./data/predicate2vec/predicate_vec_y({}).pk".format(Config.version))


def get_sentence():  # 将个体词提取出来并分词
    if "sentence({}).pk".format(Config.version) not in os.listdir("./data/raw_data/"):
        graphs = get_theo_graph()
        words = []
        for pid in graphs:
            graph = graphs[pid]
            for key in graph.keys():
                if isinstance(key, tuple) and key[0] not in ["Premise", "Target", "Point"]:
                    words.append(sentence_format(key))  # 生成训练数据
                for item in graph[key]:
                    if isinstance(item, tuple) and item[0] not in ["Premise", "Target", "Point"]:
                        words.append(sentence_format(item))  # 生成训练数据

        train_pids = random.sample(range(0, len(words)), int(len(words) * 0.9))  # 划分训练集和测试集
        train = []
        test = []
        for i in range(len(words)):
            if i in train_pids:
                train.append(words[i])
            else:
                test.append(words[i])

        save_data(words, "./data/raw_data/sentence({}).pk".format(Config.version))
        save_data(train, "./data/raw_data/s_train({}).pk".format(Config.version))
        save_data(test, "./data/raw_data/s_test({}).pk".format(Config.version))
    else:
        words = load_data("./data/raw_data/sentence({}).pk".format(Config.version))
        train = load_data("./data/raw_data/s_train({}).pk".format(Config.version))
        test = load_data("./data/raw_data/s_test({}).pk".format(Config.version))

    return words, train, test


def equation_format(equation):
    # 空格替换
    equation = equation.replace(" ", "")
    # 简化三角函数表达，减小sentence长度
    for matched in re.findall(r"sin\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "sin({})".format(matched[7:13]))
    for matched in re.findall(r"cos\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "cos({})".format(matched[7:13]))
    for matched in re.findall(r"tan\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "tan({})".format(matched[7:13]))

    # 乘方符号替换
    equation = equation.replace("**", "^")

    # 实数识别与替换
    for matched in re.findall(r"\d+\.*\d*", equation):
        equation = equation.replace(matched, "nums", 1)
    print(equation)

    # 分词
    result = []
    while len(equation) > 0:
        length = len(equation)
        for c in expr_word_list:
            if equation.startswith(c):
                result.append(equation[0:len(c)])
                equation = equation[len(c):len(equation)]
        if length == len(equation):
            result.append(equation[0])
            equation = equation[1:len(equation)]

    return result


def sentence_format(sentence):
    if sentence[0] == "Equation":
        return equation_format(sentence[1])
    elif sentence[0] in ["Length", "Measure", "Area", "Perimeter", "Altitude", "Free"]:
        return list(sentence[1])
    else:
        result = []
        for i in range(1, len(sentence)):
            result += list(sentence[i])
            if i < len(sentence) - 1:
                result.append(",")
        return result


def check_sentence_len():  # 统计sentence的长度
    words, _, _ = get_sentence()
    len_s = {}
    for i in words:
        if len(i) not in len_s:
            len_s[len(i)] = 1
        else:
            len_s[len(i)] += 1
    for key in sorted(len_s):
        print("{}: {}".format(key, len_s[key]))


# 运行此函数即可得到 sentence2vec-A.py 的训练数据
def sentence_a_data_gen():
    if "train_vec({}).pk".format(Config.version) in os.listdir("./data/sentence2vec-A/"):
        return

    _, train, test = get_sentence()

    train_vec = []  # 转化为实值表示，并补全
    test_vec = []
    for i in range(len(train)):
        train_vec.append([])
        count = 0
        for w in train[i]:
            train_vec[i].append(sentence_word_list.index(w))
            count += 1
            if count == Config.padding_size_a:
                break
        while count < Config.padding_size_a:
            train_vec[i].append(0)
            count += 1
    for i in range(len(test)):
        test_vec.append([])
        count = 0
        for w in test[i]:
            test_vec[i].append(sentence_word_list.index(w))
            count += 1
            if count == Config.padding_size_a:
                break
        while count < Config.padding_size_a:
            test_vec[i].append(0)
            count += 1

    save_data(train_vec, "./data/sentence2vec-A/train_vec({}).pk".format(Config.version))
    save_data(test_vec, "./data/sentence2vec-A/test_vec({}).pk".format(Config.version))


# 运行此函数即可得到 sentence2vec-B.py 的训练数据
def sentence_b_data_gen():
    if "train_vec_x({}).pk".format(Config.version) in os.listdir("./data/sentence2vec-B/"):
        return

    _, train, test = get_sentence()

    train_vec_xi = []  # input seqs
    train_vec_xo = []  # output seqs
    train_vec_y = []  # next word

    test_vec = []   # input seqs

    for s in train:    # 训练集
        s.insert(0, "<start>")    # 添加起始符
        s.append("<end>")

        s_vec = []    # 转化为实值
        for i in s:
            s_vec.append(sentence_word_list.index(i))

        for i in range(len(s_vec) - 1):    # 生成训练数据
            train_vec_xi.append(s_vec)
            train_vec_xo.append(s_vec[0:i + 1])
            train_vec_y.append(s_vec[i + 1])

    for s in test:    # 测试集
        s.insert(0, "<start>")  # 添加起始符
        s.append("<end>")

        s_vec = []  # 转化为实值
        for i in s:
            s_vec.append(sentence_word_list.index(i))

        test_vec.append(s_vec)

    save_data(train_vec_xi, "./data/sentence2vec-B/train_vec_xi({}).pk".format(Config.version))
    save_data(train_vec_xo, "./data/sentence2vec-B/train_vec_xo({}).pk".format(Config.version))
    save_data(train_vec_y, "./data/sentence2vec-B/train_vec_y({}).pk".format(Config.version))
    save_data(test_vec, "./data/sentence2vec-B/test_vec({}).pk".format(Config.version))
