from graphviz import Digraph
from theorem import theorem_name_map
import string


class SolutionTree:

    def __init__(self, problem):
        self.dot = Digraph(name=str(problem.problem_index), format="png")  # 求解树
        self.used = []  # 使用到的条件
        self.problem = problem

    def generate_tree(self):  # 简单生成求解树，定理和条件都是题目序号
        for i in range(self.problem.target_count):  # 所有解题目标需要的条件
            target = self.problem.targets[i]
            if target.target_solved:
                self.dot.node(string.ascii_uppercase[i], str(target.target), style='filled', fillcolor='#43CD80',
                              shape='box')
                for premise in target.premise:
                    self.dot.node(str(premise), str(premise), style='filled', fillcolor='#FFDEAD', shape='box')
                    self.dot.edge(str(premise), string.ascii_uppercase[i], str(target.theorem))
                    self.used.append(premise)

        self.used = list(set(self.used))  # 快速去重
        visited = []  # 已经访问过的节点，防止重复访问

        while True:  # 向上遍历，寻找添加其他条件
            length = len(self.used)
            for index in self.used:
                if index not in visited:
                    for premise in self.problem.conditions.get_premise_by_index(index):
                        if premise == -1:
                            self.dot.node(str(index), str(index), style='filled', fillcolor='#40e0d0', shape='box')
                        else:
                            self.dot.node(str(premise), str(premise), style='filled', fillcolor='#FFDEAD', shape='box')
                            self.dot.edge(str(premise), str(index),
                                          str(self.problem.conditions.get_theorem_by_index(index)))
                            self.used.append(premise)
                    visited.append(index)
            self.used = list(set(self.used))  # 快速去重
            if len(self.used) == length:  # 如果没有更新，结束循环
                break

    def generate_tree_for_human(self):  # 条件和定理都是形式化语言
        for i in range(self.problem.target_count):  # 所有解题目标需要的条件
            target = self.problem.targets[i]
            if target.target_solved:
                node_name = str(target.target)    # 目标节点
                self.dot.node(string.ascii_uppercase[i], node_name, style='filled', fillcolor='#43CD80', shape='box')
                for premise in target.premise:
                    node_name = self.problem.anti_generate_one_fl_by_index(premise)
                    self.dot.node(str(premise), node_name, style='filled', fillcolor='#FFDEAD', shape='box')
                    edge_name = theorem_name_map[target.theorem]
                    self.dot.edge(str(premise), string.ascii_uppercase[i], edge_name)
                    self.used.append(premise)

        self.used = list(set(self.used))  # 快速去重
        visited = []  # 已经访问过的节点，防止重复访问

        while True:  # 向上遍历，寻找添加其他条件
            length = len(self.used)
            for index in self.used:
                if index not in visited:
                    for premise in self.problem.conditions.get_premise_by_index(index):
                        if premise == -1:
                            node_name = self.problem.anti_generate_one_fl_by_index(index)
                            self.dot.node(str(index), node_name, style='filled', fillcolor='#40e0d0', shape='box')
                        else:
                            node_name = self.problem.anti_generate_one_fl_by_index(premise)
                            self.dot.node(str(premise), node_name, style='filled', fillcolor='#FFDEAD', shape='box')
                            edge_name = theorem_name_map[self.problem.conditions.get_theorem_by_index(index)]
                            self.dot.edge(str(premise), str(index), edge_name)
                            self.used.append(premise)
                    visited.append(index)
            self.used = list(set(self.used))  # 快速去重
            if len(self.used) == length:  # 如果没有更新，结束循环
                break

    def show_tree(self):
        self.dot.render(directory="./solution_tree/", view=False)
