from graphviz import Digraph
from theorem import TheoremMap
import pickle


class SolutionTree:

    def __init__(self):
        self.dot = None
        self.nodes = []    # 点集 (node_name, node_type, c_index) 三元组
        self.node_count = 0    # 结点计数
        self.edges = []    # 边集 (node_index, node_index) 二元组
        self.used = []    # 解题过程中用到的 problem 条件的 index

    def generate_tree(self, problem):  # 条件和定理都是形式化语言
        self.dot = Digraph(name=str(problem.problem_index), format="png")  # 求解树
        visited = []    # 已经遍历过的节点

        for i in range(problem.target_count):  # 所有解题目标
            target = problem.targets[i]
            if target.target_solved:
                target_node_index = self.add_node(("Target", str(target.target)),
                                                  "c", -1)    # 目标
                visited.append(target_node_index)
                theorem_node_index = self.add_node(TheoremMap.get_theorem_name(target.theorem),
                                                   "t", -1)    # 定理
                visited.append(theorem_node_index)
                self.add_edge(theorem_node_index, target_node_index)
                for premise in target.premise:
                    premise_node_index = self.add_node(problem.anti_generate_one_fl_by_index(premise),
                                                       "c", premise)  # 前提
                    self.add_edge(premise_node_index, theorem_node_index)

        while True:  # 向上遍历，寻找添加其他条件
            length = self.node_count
            for target_node_index in range(length):
                if target_node_index not in visited:
                    visited.append(target_node_index)
                    n_name, n_type, c_index = self.nodes[target_node_index]    # 目标
                    theorem_number = problem.conditions.get_theorem_by_index(c_index)    # 定理标号
                    if theorem_number == -1:    # 如果是初始节点，就不用向上遍历了
                        continue
                    theorem_node_index = self.add_node(TheoremMap.get_theorem_name(theorem_number),
                                                       "t", -1)  # 定理
                    visited.append(theorem_node_index)
                    self.add_edge(theorem_node_index, target_node_index)
                    for premise in problem.conditions.get_premise_by_index(c_index):
                        premise_node_index = self.add_node(problem.anti_generate_one_fl_by_index(premise),
                                                           "c", premise)
                        self.add_edge(premise_node_index, theorem_node_index)
            if self.node_count == length:  # 没有添加新结点，结束循环
                break

    def add_node(self, node_name, node_type, c_index):
        if node_type == "t":    # node_type == "theorem"
            node_index = self.node_count
            self.node_count += 1
            self.nodes.append((node_name, node_type, c_index))
            self.dot.node(str(node_index), str(node_name))
            return node_index

        if (node_name, node_type, c_index) not in self.nodes:    # node_type == "condition"
            self.used.append(c_index)
            node_index = self.node_count
            self.node_count += 1
            self.nodes.append((node_name, node_type, c_index))
            self.dot.node(str(node_index), str(node_name), shape='box')
            return node_index
        return self.nodes.index((node_name, node_type, c_index))

    def add_edge(self, node1, node2):
        if (node1, node2) not in self.edges:
            self.edges.append((node1, node2))
            self.dot.edge(str(node1), str(node2))

    def save(self, file_dir):
        if self.dot is not None:
            self.dot.render(directory=file_dir, view=False)
            solution_hyper_graph = {}
            node_type = {"c": [], "t": []}
            for edge in self.edges:
                for i in range(2):
                    node = self.nodes[edge[i]]
                    if node[0] not in node_type[node[1]]:
                        node_type[node[1]].append(node[0])
                if self.nodes[edge[0]][0] not in solution_hyper_graph.keys():
                    solution_hyper_graph[self.nodes[edge[0]][0]] = [self.nodes[edge[1]][0]]
                else:
                    solution_hyper_graph[self.nodes[edge[0]][0]].append(self.nodes[edge[1]][0])

            with open(file_dir + "{}_graph.pk".format(self.dot.name), 'wb') as f:
                pickle.dump(solution_hyper_graph, f)
            with open(file_dir + "{}_ntype.pk".format(self.dot.name), 'wb') as f:
                pickle.dump(node_type, f)
        else:
            raise RuntimeError("Please generate the SolutionTree before showing.")

    def show_tree_data(self):
        print("nodes:")
        for node in self.nodes:
            print(node)
        print("edges:")
        for edge in self.edges:
            print(edge)
