# 定义一些数据结构
# 如 Condition、Relation、Equation
# 再比如 Target ...
from definition.exception import RuntimeException
from aux_tools.logic import run
from sympy import symbols, solve, Float, Integer
from func_timeout import func_set_timeout


class Condition:
    _id = 0

    def __init__(self, name):
        """
        a set of conditions.
        :param name: <str> type of condition, one-to-one correspondence with predicate.
        self.items: list of item
        self.ids: key:item, value: id
        self.premises: key:item or id, value: premise
        self.theorems: key:item or id, value: theorem
        """
        self.name = name
        self.items = []
        self.ids = {}
        self.premises = {}
        self.theorems = {}

    def add(self, item, premise, theorem):  # 添加条件
        """
        add item and guarantee no redundancy.
        :param item: relation or equation
        :param premise: <tuple> of <int>
        :param theorem: <int>
        :return: ddd successfully or not, item id
        """
        if item not in self.items:
            self.items.append(item)  # item
            _id = Condition._id
            Condition._id += 1
            self.ids[item] = _id  # id
            self.premises[item] = premise  # premise
            self.premises[_id] = premise
            self.theorems[item] = theorem  # theorem
            self.theorems[_id] = theorem  # theorem
            return True, _id
        return False, None

    def get_id(self, item):
        """return id by item. if item not in items, raise ItemNotFound error"""
        if item not in self.items:  # 错误的调用会报错
            raise RuntimeException("ItemNotFound",
                                   "No {} found in {}.".format(str(item), self.name))
        return self.ids[self.items.index(item)]

    def get_premise(self, item_or_id):
        """
        return premise by item or id. if item not in items, raise ItemNotFound error
        :param item_or_id: item <relation or equation> or id <int>
        """
        if item_or_id not in self.premises:
            raise RuntimeException("ItemNotFound",
                                   "No {}(item or id) found in {}.".format(str(item_or_id), self.name))
        return self.premises[item_or_id]

    def get_theorem(self, item_or_id):
        """
        return theorem by item or id. if item not in items, raise ItemNotFound error
        :param item_or_id: item <relation or equation> or id <int>
        """
        if item_or_id not in self.theorems:
            raise RuntimeException("ItemNotFound",
                                   "No {}(item or id) found in {}.".format(str(item_or_id), self.name))
        return self.theorems[item_or_id]

    def __str__(self):
        return self.name + " <Condition> with {} items".format(len(self.items))


class Construction(Condition):

    def __init__(self, name):
        super(Construction, self).__init__(name)

    def __call__(self, variables):
        """generate a function to get items, premise and variables when reasoning"""

        def get_data():
            items = []
            ids = []
            expected_len = len(variables)
            for item in self.items:
                if len(item) == expected_len:
                    items.append(items)
                    ids.append((self.ids[item],))
            return ids, items, variables

        return get_data


class Relation(Condition):

    def __init__(self, name):
        super(Relation, self).__init__(name)

    def __call__(self, variables):
        """generate a function to get items, premise and variables when reasoning"""

        def get_data():
            ids = []
            for item in self.items:
                ids.append((self.ids[item],))
            return ids, self.items, variables

        return get_data


class Equation(Condition):

    def __init__(self, name, attr_GDL):
        """
        self.sym_of_attr = {}  # Symbolic representation of attribute values.
        >> {(('A', 'B'), 'll'): ll_ab}
        self.value_of_sym = {}  # Value of symbol.
        >> {ll_ab: 3.0}
        self.equations = {}    # Simplified equations. Replace sym with value of symbol's value already known.
        >> {a + b - c: a -5}    # Suppose that b, c already known.
        self.solved = True   # If not solved, then solve.
        """
        super(Equation, self).__init__(name)
        self.attr_GDL = attr_GDL

        self.sym_of_attr = {}
        self.value_of_sym = {}
        self.equations = {}
        self.solved = True

    def add(self, item, premise, theorem):
        """reload add() of parent class <Condition> to adapt equation's operation."""
        added, _id = super().add(item, premise, theorem)
        if added:
            self.equations[item] = item
            self.solved = False
        return added, _id

    def get_sym_of_attr(self, item, attr=None):
        """
        Get symbolic representation of item's attribution.
        :param item: tuple, such as ('A', 'B')
        :param attr: attr's name, such as Length
        :return:
        """
        if attr is None:    # middle sym, no need to storage.
            return symbols("m_m")

        if (item, attr) not in self.sym_of_attr:    # No symbolic representation, initialize one.
            if self.attr_GDL[attr]["negative"] == "True":    # Judge whether sym can be negative.
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower())
            else:
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower(), positive=True)

            self.value_of_sym[sym] = None     # init symbol's value
            self.sym_of_attr[(item, attr)] = sym    # add sym

            if self.attr_GDL[attr]["multi_rep"] == "True":    # Judge whether sym has multi representation.
                l = len(item)
                for bias in range(l):
                    extended_item = []
                    for i in range(l):
                        extended_item.append(item[(i + bias) % l])
                    self.sym_of_attr[(tuple(extended_item), attr)] = sym
            return sym
        else:
            return self.sym_of_attr[(item, attr)]

    def set_value_of_sym(self, sym, value, premise, theorem):
        """
        Set value of sym.
        Add equation to record the premise and theorem of solving the symbol's value at the same time.
        :param sym: <symbol>
        :param value: <float>
        :param premise: tuple of <int>, premise of getting value.
        :param theorem: <str>, theorem of getting value.
        """
        if self.value_of_sym[sym] is None:
            self.value_of_sym[sym] = value
            added, _id = super().add(sym - value, premise, theorem)
            return added
        return False

    def solve_target(self, target_expr):  # 求解target_expr的值
        # 无需求解的情况
        if target_expr in self.items or -target_expr in self.items:  # 如果是已知方程，那么target_expr=0
            return 0.0, [self.ids[target_expr]]

        # 简单替换就可求解的情况
        premise = []
        for sym in target_expr.free_symbols:  # 替换掉target_expr中符号值已知的符号
            if self.value_of_sym[sym] is not None:
                premise.append(self.ids[sym - self.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])
        if len(target_expr.free_symbols) == 0:
            return float(target_expr), premise

        # 需要求解的情况
        equations, eq_premise = self.get_minimum_equations(target_expr)  # 最小依赖方程组
        premise += eq_premise  # 上面替换掉的符号的premise
        # print("高级化简之前:", end="  ")
        # print(equations, end=",  ")
        # print(target_expr)
        equations = self.high_level_simplify(equations, target_expr)  # 高级化简
        target_sym = symbols("t_s")
        equations[-1] = target_sym - equations[-1]  # 添加目标方程
        # print("高级化简之后:", end="  ")
        # print(equations)
        solved_result = solve(equations)  # 求解最小方程组
        # print(solved_result)
        # print()

        if len(solved_result) > 0 and isinstance(solved_result, list):  # 若解不唯一，选择第一个
            solved_result = solved_result[0]

        if len(solved_result) > 0 and \
                target_sym in solved_result.keys() and \
                (isinstance(solved_result[target_sym], Float) or
                 isinstance(solved_result[target_sym], Integer)):
            return float(solved_result[target_sym]), list(set(premise))  # 有实数解，返回解

        return None, None  # 无解，返回None

    def get_minimum_equations(self, target_expr):  # 返回求解target_expr依赖的最小方程组
        sym_set = target_expr.free_symbols  # 方程组涉及到的符号
        min_equations = []  # 最小方程组
        premise = []  # 每个方程的index，作为target_expr求解结果的premise

        # 循环添加依赖方程，得到最小方程组
        update = True
        while update:
            update = False
            for sym in sym_set:
                if self.value_of_sym[sym] is None:  # 如果sym的值未求出，需要添加包含sym的依赖方程
                    for key in self.equations.keys():  # 添加简单依赖方程
                        if sym in self.equations[key].free_symbols and \
                                self.equations[key] not in min_equations:
                            min_equations.append(self.equations[key])
                            premise.append(self.ids[key])
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号(未化简的原方程的所有符号)
                            update = True
                    for key in self.equations.keys():  # 添加复杂依赖方程
                        if sym in self.equations[key].free_symbols and \
                                self.equations[key] not in min_equations:
                            min_equations.append(self.equations[key])
                            premise.append(self.ids[key])
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号
                            update = True
        # 化简最小方程组
        for sym in sym_set:
            if self.value_of_sym[sym] is not None:
                premise.append(self.ids[sym - self.value_of_sym[sym]])
                for i in range(len(min_equations)):  # 替换方程中的已知sym
                    min_equations[i] = min_equations[i].subs(sym, self.value_of_sym[sym])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])  # 替换target_expr中的已知sym

        return min_equations, premise  # 返回化简的target_expr、最小依赖方程组和前提

    def simplify_equations(self):  # 化简 basic、complex equations
        update = True
        while update:
            update = False
            remove_lists = []  # 要删除的 basic equation 列表
            for key in self.equations.keys():
                for sym in self.equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if self.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        self.equations[key] = self.equations[key].subs(sym, self.value_of_sym[sym])
                        update = True

                if len(self.equations[key].free_symbols) == 0:  # 没有未知符号：删除方程
                    remove_lists.append(key)

                if len(self.equations[key].free_symbols) == 1:  # 只剩一个符号：求得符号值，然后删除方程
                    target_sym = list(self.equations[key].free_symbols)[0]
                    value = solve(self.equations[key])[0]
                    premise = [self.ids[key]]
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.ids[sym - self.value_of_sym[sym]])
                    self.set_value_of_sym(target_sym, value, tuple(premise), "solve_eq")
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                self.equations.pop(remove_eq)

    @staticmethod
    def high_level_simplify(equations, target_expr):    # 基于替换的高级化简
        update = True
        while update:
            update = False
            for equation in equations:    # 替换符号
                if len(equation.free_symbols) == 2:
                    result = solve(equation)
                    if len(result) > 0:  # 有解
                        if isinstance(result, list):  # 若解不唯一，选择第一个
                            result = result[0]
                        sym = list(result.keys())[0]
                        target_expr = target_expr.subs(sym, result[sym])    # 符号替换
                        for i in range(len(equations)):
                            equations[i] = equations[i].subs(sym, result[sym])    # 符号替换
                        update = True

        equations.append(target_expr)

        return equations

    @func_set_timeout(10)  # 限时10s
    def solve(self):  # 求解basic、complex equations
        if self.solved:  # equations没有更新，不用重复求解
            return

        update = True
        while update:
            update = False
            self.simplify_equations()    # solve前先化简方程

            sym_set = []    # 得到所有值未知的符号
            for equation in list(self.equations.values()):
                sym_set += equation.free_symbols
            sym_set = list(set(sym_set))  # 快速去重

            for sym in sym_set:    # 方程求解
                equations, premise = self.get_minimum_equations(sym)
                result = solve(equations)  # 求解最小方程组

                if len(result) > 0:  # 有解
                    if isinstance(result, list):  # 若解不唯一，选择第一个
                        result = result[0]
                    for key in result.keys():  # 遍历并保存所有解
                        if self.value_of_sym[key] is None \
                                and (isinstance(result[key], Float) or isinstance(result[key], Integer)):
                            self.set_value_of_sym(key, float(result[key]), tuple(premise), "solve_eq")
                            update = True

        self.solved = True

    def gen_expr_from_tree_data(self, tree):
        return tree


class Problem:
    def __init__(self, predicate_GDL, problem_msg):
        """
        initialize a problem.
        :param predicate_GDL: parsed predicate_GDL.
        :param problem_msg: parsed problem_CDL
        """
        self.msg = problem_msg   # 用于输出的，调试完了删掉就行
        self.predicate_GDL = predicate_GDL  # problem predicate definition

        self.id = problem_msg["id"]  # problem id

        self.fl = problem_msg["fls"]  # problem fls
        self.fl["step_fls_id"] = {}

        self.condition_list = []  # list of item: (item, predicate), row number is id start from 0.

        self.conditions = {  # dict, key:predicate_name, value: children of <Condition>.
            "Shape": Construction("Shape"),
            "Collinear": Construction("Collinear"),
            "Equation": Equation("Equation", self.predicate_GDL["Attribution"])
        }
        for p in self.predicate_GDL["Entity"]:
            self.conditions[p] = Relation(p)
        for p in self.predicate_GDL["Relation"]:
            self.conditions[p] = Relation(p)

        for p, item in problem_msg["parsed_fls"]["construction_fls"]:    # conditions of construction
            self.add(p, item, (-1,), "prerequisite")

        for k in self.conditions:
            print(self.conditions[k])

        for i in self.condition_list:
            print(i)

        self.construction_init()   # start construction

        for p, item in problem_msg["parsed_fls"]["text_and_image_fls"]:    # conditions of text_and_image
            self.add(p, item, (-1,), "prerequisite")

        self.target = problem_msg["parsed_fls"]["target"]  # set target

    def construction_init(self, ):
        pass

    def add(self, predicate, item, premise, theorem):
        """
        Add item to condition of specific predicate category.
        Also consider condition expansion and equation construction.
        :param predicate: Construction, Entity, Relation or Equation.
        :param item: item.
        :param premise: tuple of <int>, premise of item.
        :param theorem: <str>, theorem of item.
        :return:
        """
        # print(predicate)
        # print(item)
        # print(premise)
        # print(theorem)
        # print()
        if predicate not in self.conditions:
            raise RuntimeException("PredicateNotDefined",
                                   "Predicate '{}': not defined in current problem.".format(predicate))

        if predicate in ["Shape", "Collinear"]:
            return self._add_construction(predicate, item, premise, theorem)
        elif predicate == "Equation":
            return self._add_equation(item, premise, theorem)
        elif predicate in self.predicate_GDL["Entity"]:
            return self._add_entity(predicate, item, premise, theorem)
        else:    # Relation
            return self._add_relation(predicate, item, premise, theorem)

    def _add_construction(self, predicate, item, premise, theorem):
        if predicate == "Shape":
            added, _id = self.conditions["Shape"].add(tuple(item), premise, theorem)
            if added:    # if added successful
                self.condition_list.append((tuple(item), "Shape"))
                l = len(item)
                for bias in range(l):
                    extended_shape = []    # extend Shape
                    for i in range(l):
                        extended_shape.append(item[(i + bias) % l])
                    added, _ = self.conditions["Shape"].add(tuple(extended_shape), (_id,), "extended")
                    if added:
                        self.condition_list.append((tuple(extended_shape), "Shape"))
                    extended_angle = [item[0 + bias], item[(1 + bias) % l], item[(2 + bias) % l]]    # extend Angle
                    self.add("Angle", tuple(extended_angle), (_id,), "extended")
                return True
        else:    # Collinear
            added, _id = self.conditions["Collinear"].add(tuple(item), premise, theorem)
            if added:
                self.condition_list.append((tuple(item), "Collinear"))
                self.conditions["Collinear"].add(tuple(item[::-1]), (_id,), "extended")    # extend Collinear
                for point in item:
                    added, _ = self.conditions["Point"].add((point,), (_id,), "extended")    # extend Point
                    if added:
                        self.condition_list.append(((point,), "Point"))
                return True
        return False

    def _add_equation(self, item, premise, theorem):
        left_expr = self.conditions["Equation"].gen_expr_from_tree_data(item[0])
        right_expr = self.conditions["Equation"].gen_expr_from_tree_data(item[1])
        added, _id = self.conditions["Equation"].add(left_expr - right_expr, premise, theorem)
        if added:
            self.condition_list.append((left_expr - right_expr, "Equation"))
            return True

    def _add_entity(self, predicate, item, premise, theorem):
        added, _id = self.conditions[predicate].add(tuple(item), premise, theorem)
        if added:
            self.condition_list.append((tuple(item), predicate))
            for extended_predicate, para_list in self.predicate_GDL["Entity"][predicate]["extend"]:    # extended
                para = []
                for i in para_list:
                    para.append(item[i])
                self.add(extended_predicate, tuple(para), (_id,), "extended")
            return True
        return False

    def _add_relation(self, predicate, item, premise, theorem):
        added, _id = self.conditions[predicate].add(tuple(item), premise, theorem)
        if added:
            self.condition_list.append((tuple(item), predicate))
            for extended_predicate, para_list in self.predicate_GDL["Relation"][predicate]["extend"]:  # extended
                para = []
                for i in para_list:
                    para.append(item[i])
                self.add(extended_predicate, tuple(para), (_id,), "extended")
            return True
        return False


if __name__ == '__main__':
    shape = Relation("shape")
    shape.add(("A", "B", "C"), (-1), -1)
    shape.add(("B", "C", "D"), (-1), -1)
    shape.add(("E", "F", "G"), (-1), -1)
    shape.add(("F", "G", "H"), (-1), -1)

    r1_ids, r1_items, r1_vars = run((shape((0, 1, 2)), shape((1, 2, 3))), ("and",))
    print(r1_ids)
    print(r1_items)
    print(r1_vars)
