from utility import load_data, save_data
import os
walking_depth = 2    # 随机游走的深度


def node_format(current_node, next_node, node_type):    # 提取谓词和定理
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


def random_walking(root_node, current_node, graph, node_type, depth, data):    # 递归随机游走
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
        for i in list(set(one_problem)):
            data.append(i)
            print(i)
        print()

    return data


# 个体词embedding
def gen_for_sentence(data_path):
    data = []
    for filename in os.listdir(data_path):
        if filename.endswith("steps.pk"):
            data.append(load_data(data_path + filename))
    return data
