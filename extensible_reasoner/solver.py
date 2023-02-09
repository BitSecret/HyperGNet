from definition.exception import RuntimeException
from definition.problem import Problem
from aux_tools.parse import FLParser
from aux_tools.parse import EqParser
import copy
import time
from sympy import symbols, solve, Float, Integer
from func_timeout import func_set_timeout


class Solver:

    def __init__(self, predicate_GDL, theorem_GDL):
        self.predicate_GDL = FLParser.parse_predicate(predicate_GDL)
        self.theorem_GDL = FLParser.parse_theorem(theorem_GDL)
        self.problem = None

    def load_problem(self, problem_CDL):
        """Load problem through problem_CDL."""
        s_start_time = time.time()  # timing
        self.problem = Problem(self.predicate_GDL, FLParser.parse_problem(problem_CDL))  # init problem
        self.solve()  # Solve the equations after initialization
        self.problem.applied("init_problem")  # save applied theorem and update step
        self.problem.goal["solving_msg"].append(
            "\033[32mInit problem and first solving\033[0m:{:.6f}s".format(time.time() - s_start_time))

    def greedy_search(self, orientation="forward"):
        """Breadth-first search."""
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run <load_problem> before run <greedy_search>.")

        if orientation == "forward":    # forward search
            pass
        else:    # backward search
            pass

    def apply_theorem(self, theorem_name):
        """
        Forward reasoning.
        :param theorem_name: theorem's name. Such as 'pythagorean'.
        :return update: whether condition update or not.
        """
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run <load_problem> before run <apply_theorem>.")

        if theorem_name not in self.theorem_GDL:
            raise RuntimeException("TheoremNotDefined", "Theorem {} not defined.".format(theorem_name))

        s_start_time = time.time()
        update = False

        for branch in self.theorem_GDL[theorem_name]:
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

                r_ids, r_items, r_vars = results  # add satisfied results to conclusion
                for i in range(len(r_items)):
                    for predicate, para in b_conclusion:
                        if predicate != "Equal":  # logic relation
                            item = [r_items[i][r_vars.index(j)] for j in para]
                            update = self.problem.add(predicate, tuple(item), r_ids[i], theorem_name) or update
                        else:  # algebra relation
                            equation = EqParser.get_equation_from_tree(self.problem, para, True, r_items[i])
                            update = self.problem.add("Equation", equation, r_ids[i], theorem_name) or update

        if update:  # add theorem to problem theorem_applied list when update
            self.solve()
            self.problem.applied(theorem_name)  # save applied theorem and update step
            self.problem.goal["solving_msg"].append(
                "\033[32mApply theorem <{}>\033[0m:{:.6f}s".format(theorem_name, time.time() - s_start_time))

        return update

    def logic_and(self, results, logic):
        """
        Underlying implementation of <relational reasoning>: logic part.
        Note that logic[0] may start with '~'.
        :param results: triplet, (r1_ids, r1_items, r1_vars).
        :param logic: predicate and vars.
        :return: triplet, reasoning result.
        >> self.problem.conditions['Line']([1, 2])
        ([(3,), (4,)], [('B', 'C'), ('D', 'E')], [1, 2])
        >> logic_and(([(1,), (2,)], [('A', 'B'), ('C', 'D')], [0, 1]), ['Line', [1, 2]])
        ([(1, 3), (2, 4)], [('A', 'B', 'C'), ('C', 'D', 'E')], [0, 1, 2])
        """
        negate = False  # Distinguishing operation ‘&’ and '&~'
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

        if not negate:  # &
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
        else:  # &~
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
        """
        Underlying implementation of <relational reasoning>: algebra part.
        Note that equal[0] may start with '~'.
        :param results: triplet, (r1_ids, r1_items, r1_vars).
        :param equal: equal tree.
        :return: triplet, reasoning result.
        >> self.problem.conditions['Equation'].value_of_sym
        {ll_ab: 1, ll_cd: 2}
        >> logic_and(([(1,), (2,)], [('A', 'B'), ('C', 'D')], [0, 1]), ['Equal', [['Line', [0, 1]], 1]])
        ([(1, N)], [('A', 'B')], [0, 1])
        """
        negate = False  # Distinguishing operation ‘&’ and '&~'
        if equal[0].startswith("~"):
            negate = True
            equal[0] = equal[0].replace("~", "")

        r1_ids, r1_items, r1_vars = results
        r_ids, r_items = [], []

        if not negate:  # &
            for i in range(len(r1_items)):
                equation = EqParser.get_equation_from_tree(self.problem, equal[1], True, r1_items[i])
                if equation is None:
                    continue
                result, premise = self.solve_target(equation)

                if result is not None and abs(result) < 0.001:
                    r_items.append(r1_items[i])
                    r_ids.append(tuple(set(list(r1_ids[i]) + premise)))
        else:  # &~
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

    @func_set_timeout(10)
    def solve(self):
        """Solve the equation contained in the <Problem.condition["Equation"].equations>."""
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
        >> problem.conditions['Equation'].equations
        [a - b, b - c, c - 1]
        >> solve_target(a)
        1
        >> solve_target(b)
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

    def get_minimum_equations(self, target_expr):
        """Return the minimum equation set required to solve target_expr."""
        eq = self.problem.conditions["Equation"]  # class <Equation>
        sym_set = target_expr.free_symbols
        min_equations = []
        premise = []  # equation's id

        self._simplify_equations()  # simplify equations before return minimum equations

        update = True
        while update:
            update = False
            for sym in sym_set:
                if eq.value_of_sym[sym] is None:
                    for key in eq.equations:  # add Dependency Equation
                        if sym in eq.equations[key].free_symbols and eq.equations[key] not in min_equations:
                            min_equations.append(eq.equations[key])
                            premise.append(eq.get_id_by_item[key])
                            sym_set = sym_set.union(key.free_symbols)  # add new sym
                            update = True

        for sym in sym_set:
            if eq.value_of_sym[sym] is not None:
                premise.append(eq.get_id_by_item[sym - eq.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, eq.value_of_sym[sym])  # replace sym with value when value solved

        return min_equations, target_expr, premise, sym_set

    def _simplify_equations(self):
        """Simplify all equations based on value replaced."""
        eq = self.problem.conditions["Equation"]  # class <Equation>
        update = True
        while update:
            update = False
            remove_lists = []  # equation to be deleted
            for key in eq.equations.keys():
                for sym in eq.equations[key].free_symbols:
                    if eq.value_of_sym[sym] is not None:  # replace sym with value when the value solved
                        eq.equations[key] = eq.equations[key].subs(sym, eq.value_of_sym[sym])
                        update = True

                if len(eq.equations[key].free_symbols) == 0:  # remove equation when all the sym of equation solved
                    remove_lists.append(key)

                if len(eq.equations[key].free_symbols) == 1:  # only one sym unsolved, then solved
                    target_sym = list(eq.equations[key].free_symbols)[0]
                    value = solve(eq.equations[key])[0]
                    premise = [eq.get_id_by_item[key]]
                    for sym in key.free_symbols:
                        if eq.value_of_sym[sym] is not None:
                            premise.append(eq.get_id_by_item[sym - eq.value_of_sym[sym]])
                    self.problem.set_value_of_sym(target_sym, value, tuple(premise), "solve_eq")
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # remove useless equation
                eq.equations.pop(remove_eq)

    @staticmethod
    def _high_level_simplify(equations, target_expr):
        """ High level simplify based on symbol replacement."""
        update = True
        while update:
            update = False
            for equation in equations:
                if len(equation.free_symbols) == 2:
                    result = solve(equation)
                    if len(result) > 0:
                        if isinstance(result, list):
                            result = result[0]
                        sym = list(result.keys())[0]
                        target_expr = target_expr.subs(sym, result[sym])
                        for i in range(len(equations)):
                            equations[i] = equations[i].subs(sym, result[sym])
                        update = True

        equations.append(target_expr)

        return equations

    def find_prerequisite(self, target_predicate, target_item):
        """
        Backward reasoning.
        Find prerequisite of given condition.
        :param target_predicate: condition's predicate. Such as 'Triangle', 'Equation'.
        :param target_item: condition para. Such as (‘A’, 'B', 'C'), a - b + c.
        """
        if self.problem is None:
            raise RuntimeException("ProblemNotLoaded", "Please run <load_problem> before run <find_prerequisite>.")

        results = []

        if target_predicate == "Equation":  # algebraic target
            equation = self.problem.conditions["Equation"]
            _, _, _, sym_set = self.get_minimum_equations(target_item)

            unsolved_sym = []  # only need to find unsolved symbol
            for sym in sym_set:
                if equation.value_of_sym[sym] is None and equation.attr_of_sym[sym][1] != "Free":
                    unsolved_sym.append(sym)

            for sym in unsolved_sym:  # find algebraic conclusion containing unsolved_sym
                attr_para, attr_name = equation.attr_of_sym[sym]  # such as ('A', 'B'), 'Length'
                # 一个sym可能对应多个attr_para，这里还要再考虑一下
                for theorem_name in self.theorem_GDL:
                    for branch in self.theorem_GDL[theorem_name]:
                        one_theorem = self.theorem_GDL[theorem_name][branch]
                        for conclusion in one_theorem["conclusion"]:
                            if conclusion[0] == "Equal":
                                attr_vars = self.find_vars_from_equal_tree(conclusion[1][0], attr_name) + \
                                            self.find_vars_from_equal_tree(conclusion[1][1], attr_name)
                                replaced = []
                                for attr_var in list(set(attr_vars)):  # fast redundancy removal and ergodic
                                    replaced.append([attr_para[attr_var.index(v)] if v in attr_var else v
                                                     for v in one_theorem["vars"]])
                                pres = self.prerequisite_generation(replaced, one_theorem["premise"])
                                for pre in pres:
                                    results.append((theorem_name, pre))  # add to results
        else:  # entity target
            for theorem_name in self.theorem_GDL:
                for branch in self.theorem_GDL[theorem_name]:
                    one_theorem = self.theorem_GDL[theorem_name][branch]
                    for conclusion in one_theorem["conclusion"]:
                        if conclusion[0] == target_predicate:
                            replaced = [[target_item[conclusion[1].index(v)] if v in conclusion[1] else v
                                         for v in one_theorem["vars"]]]
                            pres = self.prerequisite_generation(replaced, one_theorem["premise"])
                            for pre in pres:
                                results.append((theorem_name, pre))  # add to results

        unique = []   # redundancy removal
        for result in results:
            if result not in unique:
                unique.append(result)
        return unique

    def find_vars_from_equal_tree(self, tree, attr_name):
        """
        Called by <find_prerequisite>.
        Recursively find attr in equal tree.
        :param tree: equal tree, such as ['Length', [0, 1]].
        :param attr_name: attribution name, such as 'Length'.
        :return results: searching result, such as [[0, 1]].
        >> find_vars_from_equal_tree(['Add', [['Length', [0, 1]], '2*x-14', ['Length', [2, 3]]], 'Length')
        [[0, 1], [2, 3]]
        >> get_expr_from_tree(['Sin', [['Measure', ['0', '1', '2']]]], 'Measure')
        [[0, 1, 2]]
        """
        if not isinstance(tree, list):  # expr
            return []

        if tree[0] in self.predicate_GDL["Attribution"]:  # attr
            if tree[0] == attr_name:
                return [tuple(tree[1])]
            else:
                return []

        if tree[0] in ["Add", "Mul", "Sub", "Div", "Pow", "Sin", "Cos", "Tan"]:  # operate
            result = []
            for item in tree[1]:
                result += self.find_vars_from_equal_tree(item, attr_name)
            return result
        else:
            raise RuntimeException("OperatorNotDefined",
                                   "No operation {}, please check your expression.".format(tree[0]))

    def prerequisite_generation(self, replaced, premise):
        """
        Called by <find_prerequisite>.
        :param replaced: points set, contain points and vars, such as ('A', 1, 'C').
        :param premise: one normal form of current theorem's premise.
        :return results: prerequisite, such as [('incenter_property_intersect', (('Incenter', ('D', 'A', 'B', 'C')),))].
        """
        replaced = self.theorem_vars_completion(replaced)
        results = []
        for premise_normal in premise:
            selected = self.theorem_vars_selection(replaced, premise_normal)
            for para in selected:
                result = []
                for p in premise_normal:
                    if p[0] == "Equal":  # algebra premise
                        result.append(("Equation", EqParser.get_equation_from_tree(self.problem, p[1], True, para)))
                    else:  # logic premise
                        item = [para[i] for i in p[1]]
                        result.append((p[0], tuple(item)))
                results.append(tuple(result))
        return results

    def theorem_vars_completion(self, replaced):
        """
        Called by <prerequisite_generation>.
        Replace free vars with points. Suppose there are four points ['A', 'B', 'C', 'D'].
        >> replace_free_vars_with_letters([['A', 'B', 2]])
        >> [['A', 'B', 'C'], ['A', 'B', 'D']]
        >> replace_free_vars_with_letters([['A', 'B', 2, 3]])
        >> [['A', 'B', 'C', 'D'], ['A', 'B', 'D', 'C']]
        """
        update = True
        while update:
            update = False
            for i in range(len(replaced)):
                for j in range(len(replaced[i])):
                    if isinstance(replaced[i][j], int):  # replace var with letter
                        for point, in self.problem.conditions["Point"].get_id_by_item:
                            replaced.append(copy.copy(replaced[i]))
                            replaced[-1][j] = point
                            update = True
                        replaced.pop(i)  # delete being replaced para
                        break
        return replaced

    def theorem_vars_selection(self, replaced, premise_normal):
        """
        Called by <prerequisite_generation>.
        Select vars by theorem's premise..
        >> theorem_vars_selection([['B', 'A', 'A'], ['B', 'A', 'B'], ['B', 'A', 'C']], [['Triangle', [0, 1, 2]]])
        >> [['B', 'A', 'C']]
        """
        selected = []
        for para in replaced:
            valid = True
            for p in premise_normal:
                if p[0] == "Equal":  # algebra premise
                    if EqParser.get_equation_from_tree(self.problem, p[1], True, para) is None:
                        valid = False
                        break
                else:  # logic premise
                    item = [para[i] for i in p[1]]
                    if not self.problem.item_is_valid(p[0], tuple(item)):
                        valid = False
                        break
            if valid:
                selected.append(para)
        return selected

    def check_goal(self):
        """Check whether the solution is completed."""
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
