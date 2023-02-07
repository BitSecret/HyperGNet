from definition.exception import RuntimeException
from problem import Problem
from aux_tools.parse import FLParser
from aux_tools.parse import EqParser
from aux_tools.parse import replace_free_vars_with_letters, build_vars_from_algebraic_relation
from aux_tools.logic import run
import time
from sympy import symbols, solve, Float, Integer
from func_timeout import func_set_timeout


class Solver:

    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate_GDL = FLParser.parse_predicate(predicate_GDL)
        self.theorem_GDL = FLParser.parse_theorem(theorem_GDL)
        self.problem = None

    def load_problem(self, problem_CDL):
        s_start_time = time.time()  # timing
        self.problem = Problem(self.predicate_GDL, FLParser.parse_problem(problem_CDL))  # init problem
        self.solve()  # Solve the equations after initialization
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

        for branch in self.theorem_GDL[theorem_name]:
            b_vars = self.theorem_GDL[theorem_name][branch]["vars"]
            b_premise = self.theorem_GDL[theorem_name][branch]["premise"]
            b_conclusion = self.theorem_GDL[theorem_name][branch]["conclusion"]

            for normal_form in b_premise:
                results = self.problem.conditions[normal_form[0][0]](normal_form[0][1])  # (ids, items, vars)
                for i in range(1, len(normal_form)):
                    if len(results[0]) == 0:
                        break
                    if normal_form[i][0] == "Equal":
                        results = self.algebra_and(results, normal_form[i])
                    else:
                        results = self.logic_and(results, normal_form[i])

                r_ids, r_items, r_vars = results    # add satisfied results to conclusion
                for i in range(len(r_items)):
                    for predicate, para in b_conclusion:
                        if predicate != "Equal":    # logic relation
                            item = [r_items[i][r_vars.index(j)] for j in para]
                            update = self.problem.add(predicate, tuple(item), r_ids[i], theorem_name) or update
                        else:    # algebra relation
                            equation = EqParser.get_equation_from_tree(self.problem, para, True, r_items[i])
                            update = self.problem.add("Equation", equation, r_ids[i], theorem_name) or update

        if update:  # add theorem to problem theorem_applied list when update
            self.solve()
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
            result, premise = self.solve_target(self.problem.goal["item"])
            if result is not None:
                if abs(result - self.problem.goal["answer"]) < 0.001:
                    self.problem.goal["solved"] = True
                self.problem.goal["solved_answer"] = result
                self.problem.goal["premise"] = tuple(premise)
                self.problem.goal["theorem"] = "solve_eq"
        elif self.problem.goal["type"] == "equal":
            equation = self.problem.conditions["Equation"]
            result, premise = self.solve_target(self.problem.goal["item"])
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

    """-----------Underlying implementation of <relational reasoning>-----------"""

    def logic_and(self, results, logic):
        negate = False    # Distinguishing operation ‘&’ and '&~'
        if logic[0].startswith("~"):
            negate = True
            logic[0] = logic[0].replace("~", "")

        r1_ids, r1_items, r1_vars = results
        r2_ids, r2_items, r2_vars = self.problem.conditions[logic[0]](logic[1])
        r_ids, r_items = [], []

        r_vars = tuple(set(r1_vars) | set(r2_vars))  # union
        inter = list(set(r1_vars) & set(r2_vars))  # intersection
        for i in range(len(inter)):
            inter[i] = (r1_vars.index(inter[i]), r2_vars.index(inter[i]))  # change to index
        difference = list(set(r2_vars) - set(r1_vars))  # difference
        for i in range(len(difference)):
            difference[i] = r2_vars.index(difference[i])  # change to index

        if not negate:    # &
            for i in range(len(r1_items)):
                r1_data = r1_items[i]
                for j in range(len(r2_items)):
                    r2_data = r2_items[j]
                    correspondence = True
                    for r1_i, r2_i in inter:
                        if r1_data[r1_i] != r2_data[r2_i]:  # the corresponding points are inconsistent.
                            correspondence = False
                            break
                    if correspondence:
                        item = list(r1_data)
                        for dif in difference:
                            item.append(r2_data[dif])
                        r_items.append(tuple(item))
                        r_ids.append(tuple(set(list(r1_ids[i]) + list(r2_ids[j]))))
        else:    # &~
            r_vars = r1_vars
            for i in range(len(r1_items)):
                r1_data = r1_items[i]
                valid = True
                for j in range(len(r2_items)):
                    r2_data = r2_items[j]
                    correspondence = True
                    for r1_i, r2_i in inter:
                        if r1_data[r1_i] != r2_data[r2_i]:  # the corresponding points are inconsistent.
                            correspondence = False
                            break
                    if correspondence:
                        valid = False
                        break
                if valid:
                    r_items.append(r1_items[i])
                    r_ids.append(r1_ids[i])

        return r_ids, r_items, r_vars

    def algebra_and(self, results, equal):
        negate = False    # Distinguishing operation ‘&’ and '&~'
        if equal[0].startswith("~"):
            negate = True
            equal[0] = equal[0].replace("~", "")

        r1_ids, r1_items, r1_vars = results
        r_ids, r_items = [], []

        if not negate:
            for i in range(len(r1_items)):
                equation = EqParser.get_equation_from_tree(self.problem, equal[1], True, r1_items[i])
                if equation is None:
                    continue
                result, premise = self.solve_target(equation)

                if result is not None and abs(result) < 0.001:
                    r_items.append(r1_items[i])
                    r_ids.append(tuple(set(list(r1_ids[i]) + premise)))
        else:
            for i in range(len(r1_items)):
                equation = EqParser.get_equation_from_tree(self.problem, equal[1], True, r1_items[i])
                if equation is None:
                    r_items.append(r1_items[i])
                    r_ids.append(r1_ids[i])
                    continue
                result, premise = self.solve_target(equation)

                if result is None or abs(result) > 0.001:
                    r_items.append(r1_items[i])
                    r_ids.append(tuple(set(list(r1_ids[i]) + premise)))

        return r_ids, r_items, r1_vars

    """-------------Underlying implementation of <equation solving>-------------"""

    @func_set_timeout(10)  # 限时10s
    def solve(self):
        """Solve the equation contained in the question."""
        eq = self.problem.conditions["Equation"]  # class <Equation>

        if eq.solved:
            return

        update = True
        while update:
            update = False
            self._simplify_equations()  # simplify equations before solving

            sym_set = []
            for equation in list(eq.equations.values()):  # all symbols that have not been solved
                sym_set += equation.free_symbols
            sym_set = list(set(sym_set))  # quickly remove redundancy

            resolved_sym_set = set()
            for sym in sym_set:
                if sym in resolved_sym_set:  # skip already solved sym
                    continue

                equations, _, premise, mini_sym_set = self.get_minimum_equations(sym)
                resolved_sym_set.union(mini_sym_set)

                result = solve(equations)  # solve equations

                if len(result) > 0:
                    if isinstance(result, list):
                        result = result[0]
                    for key in result.keys():  # save solved value
                        if eq.value_of_sym[key] is None \
                                and (isinstance(result[key], Float) or isinstance(result[key], Integer)):
                            self.problem.set_value_of_sym(key, float(result[key]), tuple(premise), "solve_eq")
                            update = True

        eq.solved = True

    def solve_target(self, target_expr):
        """
        Solve target expression of symbolic form.
        >> equations = [a - b, b - c, c - 1]
        >> solve_target(a)
        1
        """
        eq = self.problem.conditions["Equation"]  # class <Equation>

        if target_expr in eq.get_item_by_id.values() or \
                -target_expr in eq.get_item_by_id.values():  # It is a known equation and does not need to solve it.
            return 0.0, [eq.get_id_by_item[target_expr]]

        premise = []
        for sym in target_expr.free_symbols:  # Solved only using value replacement
            if eq.value_of_sym[sym] is not None:
                premise.append(eq.get_id_by_item[sym - eq.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, eq.value_of_sym[sym])
        if len(target_expr.free_symbols) == 0:
            return float(target_expr), premise

        # Need to solve. Construct minimum solution equations.
        equations, target_expr, eq_premise, _ = self.get_minimum_equations(target_expr)
        premise += eq_premise
        equations = self._high_level_simplify(equations, target_expr)  # high level simplify
        target_sym = symbols("t_s")
        equations[-1] = target_sym - equations[-1]
        solved_result = solve(equations)

        if len(solved_result) > 0 and isinstance(solved_result, list):  # Multi answer. Choose the first.
            solved_result = solved_result[0]

        if len(solved_result) > 0 and \
                target_sym in solved_result.keys() and \
                (isinstance(solved_result[target_sym], Float) or
                 isinstance(solved_result[target_sym], Integer)):
            return float(solved_result[target_sym]), list(set(premise))  # Only return real solution.

        return None, None  # unsolvable

    def get_minimum_equations(self, target_expr):  # 返回求解target_expr依赖的最小方程组
        eq = self.problem.conditions["Equation"]  # class <Equation>
        sym_set = target_expr.free_symbols  # 方程组涉及到的符号
        min_equations = []  # 最小方程组
        premise = []  # 每个方程的index，作为target_expr求解结果的premise

        self._simplify_equations()  # simplify equations before return minimum equations

        # 循环添加依赖方程，得到最小方程组
        update = True
        while update:
            update = False
            for sym in sym_set:
                if eq.value_of_sym[sym] is None:  # 如果sym的值未求出，需要添加包含sym的依赖方程
                    for key in eq.equations:  # 添加简单依赖方程
                        if sym in eq.equations[key].free_symbols and eq.equations[key] not in min_equations:
                            min_equations.append(eq.equations[key])
                            premise.append(eq.get_id_by_item[key])
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号(未化简的原方程的所有符号)
                            update = True

        for sym in sym_set:
            if eq.value_of_sym[sym] is not None:
                premise.append(eq.get_id_by_item[sym - eq.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, eq.value_of_sym[sym])  # 替换target_expr中的已知sym

        return min_equations, target_expr, premise, sym_set  # 返回化简的target_expr、最小依赖方程组和前提

    def _simplify_equations(self):
        """Simplify all equations based on value replaced."""
        eq = self.problem.conditions["Equation"]  # class <Equation>
        update = True
        while update:
            update = False
            remove_lists = []  # 要删除的 basic equation 列表
            for key in eq.equations.keys():
                for sym in eq.equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if eq.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        eq.equations[key] = eq.equations[key].subs(sym, eq.value_of_sym[sym])
                        update = True

                if len(eq.equations[key].free_symbols) == 0:  # 没有未知符号：删除方程
                    remove_lists.append(key)

                if len(eq.equations[key].free_symbols) == 1:  # 只剩一个符号：求得符号值，然后删除方程
                    target_sym = list(eq.equations[key].free_symbols)[0]
                    value = solve(eq.equations[key])[0]
                    premise = [eq.get_id_by_item[key]]
                    for sym in key.free_symbols:
                        if eq.value_of_sym[sym] is not None:
                            premise.append(eq.get_id_by_item[sym - eq.value_of_sym[sym]])
                    self.problem.set_value_of_sym(target_sym, value, tuple(premise), "solve_eq")
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                eq.equations.pop(remove_eq)

    @staticmethod
    def _high_level_simplify(equations, target_expr):
        """ High level simplify based on symbol replacement."""
        update = True
        while update:
            update = False
            for equation in equations:  # 替换符号
                if len(equation.free_symbols) == 2:
                    result = solve(equation)
                    if len(result) > 0:  # 有解
                        if isinstance(result, list):  # 若解不唯一，选择第一个
                            result = result[0]
                        sym = list(result.keys())[0]
                        target_expr = target_expr.subs(sym, result[sym])  # 符号替换
                        for i in range(len(equations)):
                            equations[i] = equations[i].subs(sym, result[sym])  # 符号替换
                        update = True

        equations.append(target_expr)

        return equations
