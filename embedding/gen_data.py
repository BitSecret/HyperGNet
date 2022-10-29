from utility import load_data, save_data
import os
import re
import numpy as np
walking_depth = 2  # 随机游走的深度
expr_word_list = ["+", "-", "*", "/", "^", "@", "#", "$", "(", ")",
                  "nums", "ll_", "ma_", "as_", "pt_", "at_", "f_"]  # 表达式词表
predicate_word_list = ["Shape", "Collinear", "Point", "Line", "Angle", "Triangle", "RightTriangle", "IsoscelesTriangle",
                       "EquilateralTriangle", "Polygon", "Length", "Measure", "Area", "Perimeter", "Altitude",
                       "Distance", "Midpoint", "Intersect", "Parallel", "DisorderParallel", "Perpendicular",
                       "PerpendicularBisector", "Bisector", "Median", "IsAltitude", "Neutrality", "Circumcenter",
                       "Incenter", "Centroid", "Orthocenter", "Congruent", "Similar", "MirrorCongruent",
                       "MirrorSimilar", "Equation", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                       "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
                       "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
                       "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
                       "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76",
                       "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90"]
sentence_word_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                      "T", "U", "V", "W", "X", "Y", "Z", ",", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                      "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "+", "-", "*", "/",
                      "^", "@", "#", "$", "(", ")", "nums", "ll_", "ma_", "as_", "pt_", "at_", "f_"]


# 谓词embedding
def gen_for_predicate(data_path):
    if "predicate.pk" in os.listdir("./output/"):
        return

    data = []
    for filename in os.listdir(data_path):
        if filename.endswith("graph.pk"):
            graph = load_data(data_path + filename)
            one_problem = []
            for s_node in graph.keys():  # 从每一个node开始随机游走
                if isinstance(s_node, tuple):
                    if s_node[0] not in ["Premise", "Target"]:
                        graph_walking(s_node, s_node, graph, 0, one_problem)
                elif not s_node.startswith("-1") and not s_node.startswith("-2") and not s_node.startswith("-3"):
                    graph_walking(s_node, s_node, graph, 0, one_problem)
            data += list(set(one_problem))

    save_data(data, "./output/predicate.pk")


# 提取谓词和定理
def node_format(current_node, next_node):
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


# 递归随机游走
def graph_walking(root_node, current_node, graph, depth, data):
    if depth == walking_depth:  # 游走到到预计深度，返回
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


def one_hot_for_predicate(data_path):
    if "predicate_x.vec" in os.listdir("./output/"):
        return load_data("./output/predicate_x.vec"), load_data("./output/predicate_y.vec")
    zero = np.zeros(len(predicate_word_list), dtype=int)
    predicate_x = []
    predicate_y = []

    gen_for_predicate(data_path)
    predicate = load_data("./output/predicate.pk")
    for x, y in predicate:
        item_x = zero.copy()
        item_x[predicate_word_list.index(x)] = 1
        item_y = zero.copy()
        item_y[predicate_word_list.index(y)] = 1
        predicate_x.append(item_x)
        predicate_y.append(item_y)

    save_data(predicate_x, "./output/predicate_x.vec")
    save_data(predicate_y, "./output/predicate_y.vec")
    return predicate_x, predicate_y


# 个体词embedding
def gen_for_sentence(data_path):
    if "sentence.pk" in os.listdir("./output/"):
        return

    words = []
    for filename in os.listdir(data_path):
        if filename.endswith("graph.pk"):
            one_problem = load_data(data_path + filename)
            for key in one_problem.keys():
                if isinstance(key, tuple) and key[0] not in ["Premise", "Target", "Point"]:
                    words.append(sentence_format(key))  # 生成训练数据
                for item in one_problem[key]:
                    if isinstance(item, tuple) and item[0] not in ["Premise", "Target", "Point"]:
                        words.append(sentence_format(item))  # 生成训练数据

    data = []
    for word in words:
        data += sentence_walking(word)    # 滑动窗口生成训练数据

    save_data(data, "./output/sentence.pk")


def equation_format(equation):
    # 空格替换
    equation = equation.replace(" ", "")
    # 简化三角函数表达，减小sentence长度
    for matched in re.findall(r"sin\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "@({})".format(matched[7:13]))
    for matched in re.findall(r"cos\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "#({})".format(matched[7:13]))
    for matched in re.findall(r"tan\(pi\*ma_\w+/180\)", equation):
        equation = equation.replace(matched, "$({})".format(matched[7:13]))

    # 乘方符号替换
    equation = equation.replace("**", "^")

    # 实数识别与替换
    for matched in re.findall(r"\d+\.*\d*", equation):
        equation = equation.replace(matched, "nums", 1)

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


def sentence_walking(sentence):
    result = []
    for i in range(len(sentence)):
        if i > 0:
            result.append((sentence[i], sentence[i - 1]))
        if i < len(sentence) - 1:
            result.append((sentence[i], sentence[i + 1]))
    return result


def one_hot_for_sentence(data_path):
    if "sentence_x.vec" in os.listdir("./output/"):
        return load_data("./output/sentence_x.vec"), load_data("./output/sentence_y.vec")

    zero = np.zeros(len(sentence_word_list), dtype=int)
    sentence_x = []
    sentence_y = []

    gen_for_sentence(data_path)
    sentence = load_data("./output/sentence.pk")
    for x, y in sentence:
        item_x = zero.copy()
        item_x[sentence_word_list.index(x)] = 1
        item_y = zero.copy()
        item_y[sentence_word_list.index(y)] = 1
        sentence_x.append(item_x)
        sentence_y.append(item_y)

    save_data(sentence_x, "./output/sentence_x.vec")
    save_data(sentence_y, "./output/sentence_y.vec")
    return sentence_x, sentence_y
