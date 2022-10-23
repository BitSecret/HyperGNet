from utility import load_data, save_data
import os
import re
walking_depth = 2    # 随机游走的深度
character = ["+", "-", "*", "/", "^", "@", "#", "$", "(", ")",
             "nums", "ll_", "ma_", "as_", "pt_", "at_", "f_"]    # 表达式词表


# 提取谓词和定理
def node_format(current_node, next_node, node_type):
    data_item = []
    if current_node in node_type["c"]:
        data_item.append(current_node[0])
    else:
        data_item.append(current_node.split("_")[0])
    if next_node in node_type["c"]:
        data_item.append(next_node[0])
    else:
        data_item.append(next_node.split("_")[0])

    return data_item


# 递归随机游走
def random_walking(root_node, current_node, graph, node_type, depth, data):
    if depth == walking_depth:    # 游走到到预计深度，返回
        return

    if current_node in graph.keys():
        for next_node in graph[current_node]:
            data_item = node_format(current_node, next_node, node_type)
            if data_item[0] in ["-2", "-3"] or data_item[1] in ["-2", "-3"]:  # 求解方程、自动扩展不算步长，继续游走
                random_walking(root_node, next_node, graph, node_type, depth, data)
            else:
                data_item = node_format(root_node, next_node, node_type)
                data.append(tuple(data_item))
                random_walking(root_node, next_node, graph, node_type, depth + 1, data)


# 谓词embedding
def gen_for_predicate(data_path):
    if "predicate.pk" in os.listdir("./output/"):
        return

    raw_data = []
    for i in range(106):
        raw_data.append([load_data(data_path + "{}_graph.pk".format(i)),
                         load_data(data_path + "{}_ntype.pk".format(i))])

    data = []
    for graph, node_type in raw_data:
        one_problem = []
        for s_node in graph.keys():  # 从每一个node开始随机游走
            if isinstance(s_node, tuple):
                random_walking(s_node, s_node, graph, node_type, 0, one_problem)
            elif not s_node.startswith("-2") and not s_node.startswith("-3"):
                random_walking(s_node, s_node, graph, node_type, 0, one_problem)
        for i in list(set(one_problem)):    # 快速去重
            data.append(i)
            print(i)
        print()

    save_data(data, "./output/predicate.pk")


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
        equation = equation.replace(matched, "nums")

    # 分词
    result = []
    while len(equation) > 0:
        length = len(equation)
        for c in character:
            if equation.startswith(c):
                result.append(equation[0:len(c)])
                equation = equation[len(c):len(equation)]
        if length == len(equation):
            result.append(equation[0])
            equation = equation[1:len(equation)]

    return result


def sentence_format(sentence):
    result = [sentence[0]]
    if sentence[0] == "Equation":
        result.append(equation_format(sentence[1]))
    elif sentence[0] in ["Length", "Measure", "Area", "Perimeter", "Altitude", "Free"]:
        result.append([])
        for item in sentence[1]:
            result[1].append(item)
    else:
        result.append([])
        for i in range(1, len(sentence)):
            for item in sentence[i]:
                result[1].append(item)
            if i < len(sentence) - 1:
                result[1].append(",")
    return result


# 个体词embedding
def gen_for_sentence(data_path):
    if "sentence.pk" in os.listdir("./output/"):
        return

    data = []
    for filename in os.listdir(data_path):
        if filename.endswith("steps.pk"):
            one_problem = load_data(data_path + filename)
            for step in one_problem.keys():
                for item in one_problem[step]:
                    data.append(sentence_format(item))

    save_data(data, "./output/sentence.pk")
