# 推理部分，有两类模式四种组合
# 1.前向模式 、后向模式
# 2.交互式ITP、暴力搜索ATP
from definition.exception import RuntimeException
from definition.object import Problem
from aux_tools.parse import *


class Solver:

    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate = parse_predicate(predicate_GDL)
        self.theorem = parse_theorem(theorem_GDL)
        self.problem = None

    def load_problem(self, problem_CDL):
        self.problem = Problem(self.predicate, parse_problem(problem_CDL))

    def greedy_search(self, orientation="forward"):
        """实现ATP功能"""
        if self.problem is None:
            raise RuntimeException("<ProblemNotLoaded>", "Please run load_problem before run greedy_search.")

        pass

    def apply_theorem(self, theorem_name):
        """前向推理，应用定理，实现ITP功能"""
        if self.problem is None:
            raise RuntimeException("<ProblemNotLoaded>", "Please run load_problem before run apply_theorem.")

        pass

    def find_prerequisite(self, target_condition):
        """后向推理，应用定理，实现ITP功能"""
        if self.problem is None:
            raise RuntimeException("<ProblemNotLoaded>", "Please run load_problem before run find_prerequisite.")

        pass
