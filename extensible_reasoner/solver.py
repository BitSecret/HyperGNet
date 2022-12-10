from definition.exception import RuntimeException
from definition.object import Problem
from aux_tools.parse import parse_predicate, parse_theorem, parse_problem
from aux_tools.parse import replace_free_vars_with_letters, build_vars_from_algebraic_relation
from aux_tools.logic import run
import time


class Solver:

    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate_GDL = parse_predicate(predicate_GDL)
        self.theorem_GDL = parse_theorem(theorem_GDL)
        self.problem = None

    def load_problem(self, problem_CDL):
        s_start_time = time.time()  # timing
        self.problem = Problem(self.predicate_GDL, parse_problem(problem_CDL))  # init problem
        self.problem.conditions["Equation"].solve()  # Solve the equations after initialization
        self.problem.applied("init_problem")  # save applied theorem and update step
        self.problem.goal["solving_msg"].append(
            "\033[32mInit problem and first solving\033[0m:{:.6f}s".format(time.time() - s_start_time))

    def greedy_search(self, orientation="forward"):
        """实现ATP功能"""
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run greedy_search.")

        pass

    def apply_theorem(self, theorem_name):
        """
        Forward reasoning.
        Apply theorem and return whether condition update or not.
        """
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run apply_theorem.")

        if theorem_name not in self.theorem_GDL:
            raise RuntimeException("TheoremNotDefined", "Theorem {} not defined.".format(theorem_name))

        s_start_time = time.time()

        update = False
        premise_of_entity_relation = self.theorem_GDL[theorem_name]["premise"]["entity_relation"]
        premise_of_algebraic_relation = self.theorem_GDL[theorem_name]["premise"]["algebraic_relation"]
        conclusion_of_entity_relation = self.theorem_GDL[theorem_name]["conclusion"]["entity_relation"]
        conclusion_of_algebraic_relation = self.theorem_GDL[theorem_name]["conclusion"]["algebraic_relation"]

        relations = [self.problem.conditions[predicate](tuple(para_vars))
                     for predicate, para_vars in premise_of_entity_relation]

        ids, items, _ = run(tuple(relations))  # relational reasoning

        new_ids = []
        new_items = []
        for i in range(len(items)):
            passed = True
            id_extended = list(ids[i])
            for equal in premise_of_algebraic_relation:  # select item using algebraic relation
                value, premise = self.problem.conditions["Equation"].solve_target(
                    self.problem.conditions["Equation"].get_equation_from_tree(equal[1], True, items[i])
                )
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
            for predicate, para_vars in conclusion_of_entity_relation:  # add entity relation
                para = []
                for var in para_vars:
                    para.append(items[i][var])
                update = self.problem.add(predicate, tuple(para), ids[i], theorem_name) or update

            for equal in conclusion_of_algebraic_relation:  # add algebraic relation
                update = self.problem.add(
                    "Equation",
                    self.problem.conditions["Equation"].get_equation_from_tree(equal[1], True, items[i]),
                    ids[i],
                    theorem_name
                ) or update

        if update:  # add theorem to problem theorem_applied list when update
            self.problem.conditions["Equation"].solve()
            self.problem.applied(theorem_name)  # save applied theorem and update step
            self.problem.goal["solving_msg"].append(
                "\033[32mApply theorem <{}>\033[0m:{:.6f}s".format(theorem_name, time.time() - s_start_time))

        return update

    def find_prerequisite(self, target_predicate, target_item):
        """
        Backward reasoning. Find prerequisite of given condition.
        :param target_predicate: condition's predicate. Such as 'Triangle', 'Equation'.
        :param target_item: condition para. Such as (‘A’, 'B', 'C'), a - b + c.
        """
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run load_problem before run find_prerequisite.")

        options = {}
        options_count = 0

        if target_predicate == "Equation":  # algebraic target
            equation = self.problem.conditions["Equation"]
            _, _, _, sym_set = equation.get_minimum_equations(target_item)

            unsolved_sym = []  # only need to find unsolved symbol
            for sym in sym_set:
                if equation.value_of_sym[sym] is not None and equation.attr_of_sym[sym][1] != "Free":
                    unsolved_sym.append(sym)

            for sym in unsolved_sym:  # 查找algebraic中包含这些sym的
                attr = self.problem.conditions["Equation"].attr_of_sym[sym]
                for theorem_name in self.theorem_GDL:
                    all_vars = self.theorem_GDL[theorem_name]["vars"]  # all vars in current theorem
                    for equal_tree in self.theorem_GDL[theorem_name]["conclusion"]["algebraic_relation"]:
                        for possible_free_vars in build_vars_from_algebraic_relation(all_vars, equal_tree, attr):
                            all_para_combination = replace_free_vars_with_letters(
                                possible_free_vars,
                                [point[0] for point in self.problem.conditions["Point"].get_item_by_id.values()],
                                self.theorem_GDL[theorem_name]["mutex_points"]
                            )

                            for paras in all_para_combination:
                                one_option = []
                                for p, p_vars in self.theorem_GDL[theorem_name]["premise"]["entity_relation"]:
                                    one_option.append([p, [paras[i] for i in p_vars]])
                                for p, p_vars in self.theorem_GDL[theorem_name]["premise"]["algebraic_relation"]:
                                    one_option.append(
                                        ["Equation",
                                         self.problem.conditions["Equation"].get_equation_from_tree(p_vars, True,
                                                                                                    paras)]
                                    )
                                options[options_count] = {"theorem": theorem_name, "option": one_option}
                                options_count += 1
        else:  # entity target
            for theorem_name in self.theorem_GDL:
                all_vars = self.theorem_GDL[theorem_name]["vars"]  # all vars in current theorem

                for predicate, t_vars in self.theorem_GDL[theorem_name]["conclusion"]["entity_relation"]:
                    if predicate == target_predicate:
                        all_para_combination = replace_free_vars_with_letters(  # all possible para combination
                            [target_item[t_vars.index(all_vars[i])] if all_vars[i] in t_vars else all_vars[i]
                             for i in range(len(all_vars))],
                            [point[0] for point in self.problem.conditions["Point"].get_item_by_id.values()],
                            self.theorem_GDL[theorem_name]["mutex_points"]
                        )

                        for paras in all_para_combination:
                            one_option = []
                            for p, p_vars in self.theorem_GDL[theorem_name]["premise"]["entity_relation"]:
                                one_option.append([p, [paras[i] for i in p_vars]])
                            for p, p_vars in self.theorem_GDL[theorem_name]["premise"]["algebraic_relation"]:
                                one_option.append(
                                    ["Equation",
                                     self.problem.conditions["Equation"].get_equation_from_tree(p_vars, True, paras)]
                                )
                            options[options_count] = {"theorem": theorem_name, "option": one_option}
                            options_count += 1
        return options

    def check_goal(self):
        s_start_time = time.time()  # timing

        if self.problem.goal["type"] == "value":
            equation = self.problem.conditions["Equation"]
            result, premise = equation.solve_target(self.problem.goal["item"])
            if result is not None:
                if abs(result - self.problem.goal["answer"]) < 0.001:
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
        else:  # relation
            if self.problem.goal["answer"] in self.problem.conditions[self.problem.goal["item"]].get_id_by_item:
                self.problem.goal["solved"] = True
                self.problem.goal["solved_answer"] = self.problem.goal["answer"]
                self.problem.goal["premise"] = self.problem.conditions[self.problem.goal["item"]].premises[
                    self.problem.goal["answer"]]
                self.problem.goal["theorem"] = self.problem.conditions[self.problem.goal["item"]].theorems[
                    self.problem.goal["answer"]]

        self.problem.goal["solving_msg"].append(
            "\033[32mChecking goal\033[0m:{:.6f}s".format(time.time() - s_start_time))
