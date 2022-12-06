# 推理部分，有两类模式四种组合
# 1.前向模式 、后向模式
# 2.交互式ITP、暴力搜索ATP
from definition.exception import RuntimeException
from definition.object import Problem
from aux_tools.parse import parse_predicate, parse_theorem, parse_problem, get_equation_from_equal_tree_para
from aux_tools.logic import run
import time


class Solver:

    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate = parse_predicate(predicate_GDL)
        self.theorem = parse_theorem(theorem_GDL)
        self.problem = None

    def load_problem(self, problem_CDL):
        s_start_time = time.time()
        self.problem = Problem(self.predicate, parse_problem(problem_CDL))
        self._check_goal()
        time_cons = "\033[32mInit problem and first solving\033[0m time consuming:{:.6f}s"
        self.problem.goal["solving_msg"].append(time_cons.format(time.time() - s_start_time))

    def greedy_search(self, orientation="forward"):
        """实现ATP功能"""
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run greedy_search.")

        pass

    def apply_theorem(self, theorem_name):
        """前向推理，应用定理，实现ITP功能"""
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run apply_theorem.")

        if theorem_name not in self.theorem:
            raise RuntimeException("TheoremNotDefined", "Theorem {} not defined.".format(theorem_name))

        s_start_time = time.time()

        update = False
        premise_of_entity_relation = self.theorem[theorem_name]["premise"]["entity_relation"]
        premise_of_algebraic_relation = self.theorem[theorem_name]["premise"]["algebraic_relation"]
        conclusion_of_entity_relation = self.theorem[theorem_name]["conclusion"]["entity_relation"]
        conclusion_of_algebraic_relation = self.theorem[theorem_name]["conclusion"]["algebraic_relation"]

        relations = [self.problem.conditions[predicate](tuple(para_vars))
                     for predicate, para_vars in premise_of_entity_relation]

        ids, items, _ = run(tuple(relations), tuple([]))  # relational reasoning

        new_ids = []
        new_items = []
        for i in range(len(items)):
            passed = True
            id_extended = list(ids[i])
            for equal in premise_of_algebraic_relation:    # select item using algebraic relation
                equation = get_equation_from_equal_tree_para(
                    equal[1], self.problem.conditions["Equation"], True, items[i]
                )
                value, premise = self.problem.conditions["Equation"].solve_target(equation)
                if value is None or abs(value) > 0.0001:
                    passed = False
                    break
                id_extended += premise

            if passed:
                new_ids.append(tuple(id_extended))
                new_items.append(items[i])
        ids = new_ids
        items = new_items

        for i in range(len(items)):
            for predicate, para_vars in conclusion_of_entity_relation:    # add entity relation
                para = []
                for var in para_vars:
                    para.append(items[i][var])
                update = self.problem.add(predicate, tuple(para), ids[i], theorem_name) or update

            for equal in conclusion_of_algebraic_relation:    # add algebraic relation
                equation = get_equation_from_equal_tree_para(equal[1], self.problem.conditions["Equation"],
                                                             True, items[i])
                update = self.problem.add("Equation", equation, ids[i], theorem_name) or update

        if update:    # add theorem to problem theorem_applied list when update
            self.problem.conditions["Equation"].solve()
            self._check_goal()
            self.problem.theorems_applied.append(theorem_name)
            time_cons = "\033[32mApply and solve theorem <{}>\033[0m time consuming:{:.6f}s"
            self.problem.goal["solving_msg"].append(time_cons.format(theorem_name, time.time() - s_start_time))

        return update

    def find_prerequisite(self, target_condition):
        """后向推理，应用定理，实现ITP功能"""
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run find_prerequisite.")

        pass

    def _check_goal(self):
        if self.problem.goal["type"] == "value":
            equation = self.problem.conditions["Equation"]
            result, premise = equation.solve_target(self.problem.goal["item"])
            if result is not None:
                self.problem.goal["solved"] = True
                self.problem.goal["solved_answer"] = result
                self.problem.goal["premise"] = tuple(premise)
                self.problem.goal["theorem"] = "solve_eq"
        elif self.problem.goal["type"] == "equal":
            equation = self.problem.conditions["Equation"]
            result, premise = equation.solve_target(self.problem.goal["item"])
            if result is not None:
                if abs(result) < 0.001:
                    self.problem.goal["solved"] = True
                self.problem.goal["solved_answer"] = result
                self.problem.goal["premise"] = tuple(premise)
                self.problem.goal["theorem"] = "solve_eq"
        else:    # relation
            predicate, para = self.problem.goal["item"]
            if tuple(para) in self.problem.conditions[predicate].items:
                self.problem.goal["solved"] = True
                self.problem.goal["solved_answer"] = tuple(para)
                self.problem.goal["premise"] = self.problem.conditions[predicate].get_premise(para)
                self.problem.goal["theorem"] = self.problem.conditions[predicate].get_theorem(para)
