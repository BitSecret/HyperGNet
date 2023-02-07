from definition.exception import RuntimeException
from sympy import symbols, solve, Float, Integer
from func_timeout import func_set_timeout
from sympy import sin, cos, tan, pi
from aux_tools.parse import idt_expr, operator, stack_priority, outside_priority


class Condition:
    id = 0
    step = 0

    def __init__(self, name):
        """
        a set of conditions.
        :param name: <str> type of condition, one-to-one correspondence with predicate.
        self.items: key:id, value:item
        self.ids: key:item, value: id
        self.premises: key:item or id, value: premise
        self.theorems: key:item or id, value: theorem
        """
        self.name = name
        self.get_item_by_id = {}
        self.get_id_by_item = {}
        self.premises = {}
        self.theorems = {}
        self.step_msg = []  # (0, 2)  item 2 adding in step 0

    def add(self, item, premise, theorem):
        """
        add item and guarantee no redundancy.
        :param item: relation or equation
        :param premise: <tuple> of <int>
        :param theorem: <int>
        :return: ddd successfully or not, item id
        """
        if item not in self.get_item_by_id.values():
            _id = Condition.id
            self.get_item_by_id[_id] = item  # item
            Condition.id += 1
            self.get_id_by_item[item] = _id  # id
            self.premises[item] = premise  # premise
            self.premises[_id] = premise
            self.theorems[item] = theorem  # theorem
            self.theorems[_id] = theorem  # theorem
            self.step_msg.append((Condition.step, _id))  # step_msg
            return True, _id
        return False, None

    def __str__(self):
        return self.name + " <Condition> with {} items".format(len(self.get_item_by_id))


class Construction(Condition):

    def __init__(self, name):
        super(Construction, self).__init__(name)

    def __call__(self, variables):
        """generate a function to get items, premise and variables when reasoning"""

        def get_data():
            items = []
            ids = []
            expected_len = len(variables)
            for item in self.get_item_by_id.values():
                if len(item) == expected_len:
                    items.append(item)
                    ids.append((self.get_id_by_item[item],))
            return ids, items, variables

        return get_data


class Relation(Condition):

    def __init__(self, name):
        super(Relation, self).__init__(name)

    def __call__(self, variables):
        """generate a function to get items, premise and variables when reasoning"""

        def get_data():
            ids = []
            for item in self.get_item_by_id.values():
                ids.append((self.get_id_by_item[item],))
            return ids, list(self.get_item_by_id.values()), variables

        return get_data


class Equation(Condition):

    def __init__(self, name, attr_GDL):
        """
        self.sym_of_attr = {}  # Symbolic representation of attribute values.
        >> {(('A', 'B'), 'Length'): l_ab}
        self.value_of_sym = {}  # Value of symbol.
        >> {l_ab: 3.0}
        self.equations = {}    # Simplified equations. Replace sym with value of symbol's value already known.
        >> {a + b - c: a -5}    # Suppose that b, c already known.
        self.solved = True   # If not solved, then solve.
        """
        super(Equation, self).__init__(name)
        self.attr_GDL = attr_GDL
        self.solved = True
        self.sym_of_attr = {}
        self.attr_of_sym = {}
        self.value_of_sym = {}
        self.equations = {}

    def add(self, item, premise, theorem):
        """reload add() of parent class <Condition> to adapt equation's operation."""
        if item not in self.get_item_by_id.values() and -item not in self.get_item_by_id.values():
            added, _id = super().add(item, premise, theorem)
            self.equations[item] = item
            self.solved = False
            return added, _id
        return False, None

    def get_sym_of_attr(self, item, attr):
        """
        Get symbolic representation of item's attribution.
        :param item: tuple, such as ('A', 'B')
        :param attr: attr's name, such as Length
        :return: sym
        """
        if (item, attr) not in self.sym_of_attr:  # No symbolic representation, initialize one.
            if self.attr_GDL[attr]["negative"] == "True":  # Judge whether sym can be negative.
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower())
            else:
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower(), positive=True)

            self.value_of_sym[sym] = None  # init symbol's value
            self.sym_of_attr[(item, attr)] = sym  # add sym
            self.attr_of_sym[sym] = (item, attr)

            if self.attr_GDL[attr]["attr_multi"] == "True":
                l = len(item)
                for bias in range(1, l):
                    extended_item = []  # extend item
                    for i in range(l):
                        extended_item.append(item[(i + bias) % l])
                    self.sym_of_attr[(tuple(extended_item), attr)] = sym  # multi representation
                    self.attr_of_sym[sym] = (tuple(extended_item), attr)

            return sym

        return self.sym_of_attr[(item, attr)]

    def _set_value_of_sym(self, sym, value, premise, theorem):
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

    def get_minimum_equations(self, target_expr):  # 返回求解target_expr依赖的最小方程组
        sym_set = target_expr.free_symbols  # 方程组涉及到的符号
        min_equations = []  # 最小方程组
        premise = []  # 每个方程的index，作为target_expr求解结果的premise

        self._simplify_equations()  # simplify equations before return minimum equations

        # 循环添加依赖方程，得到最小方程组
        update = True
        while update:
            update = False
            for sym in sym_set:
                if self.value_of_sym[sym] is None:  # 如果sym的值未求出，需要添加包含sym的依赖方程
                    for key in self.equations:  # 添加简单依赖方程
                        if sym in self.equations[key].free_symbols and self.equations[key] not in min_equations:
                            min_equations.append(self.equations[key])
                            premise.append(self.get_id_by_item[key])
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号(未化简的原方程的所有符号)
                            update = True

        for sym in sym_set:
            if self.value_of_sym[sym] is not None:
                premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])  # 替换target_expr中的已知sym

        return min_equations, target_expr, premise, sym_set  # 返回化简的target_expr、最小依赖方程组和前提

    def _simplify_equations(self):
        """Simplify all equations based on value replaced."""
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
                    premise = [self.get_id_by_item[key]]
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                    self._set_value_of_sym(target_sym, value, tuple(premise), "solve_eq")
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                self.equations.pop(remove_eq)

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

    @func_set_timeout(10)  # 限时10s
    def solve(self):
        """Solve the equation contained in the question."""
        if self.solved:
            return

        update = True
        while update:
            update = False
            self._simplify_equations()  # simplify equations before solving

            sym_set = []
            for equation in list(self.equations.values()):    # all symbols that have not been solved
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
                    for key in result.keys():    # save solved value
                        if self.value_of_sym[key] is None \
                                and (isinstance(result[key], Float) or isinstance(result[key], Integer)):
                            self._set_value_of_sym(key, float(result[key]), tuple(premise), "solve_eq")
                            update = True

        self.solved = True

    def solve_target(self, target_expr):
        """
        Solve target expression of symbolic form.
        >> equations = [a - b, b - c, c - 1]
        >> solve_target(a)
        1
        """
        if target_expr in self.get_item_by_id.values() or \
                -target_expr in self.get_item_by_id.values():  # It is a known equation and does not need to solve it.
            return 0.0, [self.get_id_by_item[target_expr]]

        premise = []
        for sym in target_expr.free_symbols:  # Solved only using value replacement
            if self.value_of_sym[sym] is not None:
                premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])
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

    def get_expr_from_tree(self, tree, replaced=False, letters=None):
        """
        Recursively trans expr_tree to symbolic algebraic expression.
        :param tree: An expression in the form of a list tree.
        :param replaced: Optional. Set True when tree's item is expressed by vars.
        :param letters: Optional. Letters that will replace vars.
        >> get_expr_from_tree(['Length', ['T', 'R']], equation)
        l_tr
        >> get_expr_from_tree(['Add', [['Length', ['Z', 'X']], '2*x-14']], equation)
        2.0*f_x + l_zx - 14.0
        >> get_expr_from_tree(['Sin', [['Measure', ['Z', 'X', 'Y']]]], equation)
        sin(pi*m_zxy/180)
        """
        if not isinstance(tree, list):  # expr
            return self._parse_expr(tree)

        if tree[0] in self.attr_GDL:  # attr
            if not replaced:
                return self.get_sym_of_attr(tuple(tree[1]), tree[0])
            else:
                replaced_item = [letters[i] for i in tree[1]]
                return self.get_sym_of_attr(tuple(replaced_item), tree[0])

        if tree[0] == "Add":  # operate
            result = 0
            for item in tree[1]:
                result += self.get_expr_from_tree(item, replaced, letters)
            return result
        elif tree[0] == "Sub":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) -\
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Mul":
            result = 1
            for item in tree[1]:
                result *= self.get_expr_from_tree(item, replaced, letters)
            return result
        elif tree[0] == "Div":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) / \
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Pow":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) **\
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Sin":
            return sin(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        elif tree[0] == "Cos":
            return cos(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        elif tree[0] == "Tan":
            return tan(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        else:
            raise RuntimeException("OperatorNotDefined",
                                   "No operation {}, please check your expression.".format(tree[0]))

    def get_equation_from_tree(self, tree, replaced=False, letters=None):
        """Refer to function <get_expr_from_tree>."""
        left_expr = self.get_expr_from_tree(tree[0], replaced, letters)
        right_expr = self.get_expr_from_tree(tree[1], replaced, letters)
        return left_expr - right_expr

    def _parse_expr(self, expr):
        """Parse the expression in <str> form into <symbolic> form"""
        expr_list = idt_expr.parseString(expr + "~").asList()
        expr_stack = []
        operator_stack = ["~"]  # 栈底元素

        i = 0
        while i < len(expr_list):
            unit = expr_list[i]
            if unit in operator:  # 运算符
                if stack_priority[operator_stack[-1]] < outside_priority[unit]:
                    operator_stack.append(unit)
                    i = i + 1
                else:
                    operator_unit = operator_stack.pop()
                    if operator_unit == "+":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 + expr_2)
                    elif operator_unit == "-":
                        expr_2 = expr_stack.pop()
                        expr_1 = 0 if len(expr_stack) == 0 else expr_stack.pop()
                        expr_stack.append(expr_1 - expr_2)
                    elif operator_unit == "*":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 * expr_2)
                    elif operator_unit == "/":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 / expr_2)
                    elif operator_unit == "^":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 ** expr_2)
                    elif operator_unit == "{":  # 只有unit为"}"，才能到达这个判断
                        i = i + 1
                    elif operator_unit == "@":  # sin
                        expr_1 = expr_stack.pop()
                        expr_stack.append(sin(expr_1))
                    elif operator_unit == "#":  # cos
                        expr_1 = expr_stack.pop()
                        expr_stack.append(cos(expr_1))
                    elif operator_unit == "$":  # tan
                        expr_1 = expr_stack.pop()
                        expr_stack.append(tan(expr_1))
                    elif operator_unit == "~":  # 只有unit为"~"，才能到达这个判断，表示表达式处理完成
                        break
            else:  # 实数或符号
                unit = self.get_sym_of_attr((unit,), "Free") if unit.isalpha() else float(unit)
                expr_stack.append(unit)
                i = i + 1

        return expr_stack.pop()

    def angle_alignment(self, angles, collinear):
        for angle in angles:
            if (angle, "Measure") in self.sym_of_attr:
                continue
            sym = self.get_sym_of_attr(angle, "Measure")

            a, v, b = angle
            a_points = []    # Points collinear with a and on the same side with a
            b_points = []
            for coll in collinear:
                if v in coll and a in coll:
                    if coll.index(v) < coll.index(a):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            a_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            a_points.append(coll[i])
                            i += 1
                    break
            if len(a_points) == 0:
                a_points.append(a)
            for coll in collinear:
                if v in coll and b in coll:
                    if coll.index(v) < coll.index(b):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            b_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            b_points.append(coll[i])
                            i += 1
                    break
            if len(b_points) == 0:
                b_points.append(b)

            if len(a_points) == 1 and len(b_points) == 1:  # 角只有一种表示
                continue

            same_angles = []
            for a_point in a_points:
                for b_point in b_points:
                    same_angles.append((a_point, v, b_point))  # 相同的角设置一样的符号

            for same_angle in same_angles:
                self.sym_of_attr[(same_angle, "Measure")] = sym
                self.attr_of_sym[sym] = (same_angle, "Measure")


class NewEquation(Condition):

    def __init__(self, name, attr_GDL):
        """
        self.sym_of_attr = {}  # Symbolic representation of attribute values.
        >> {(('A', 'B'), 'Length'): l_ab}
        self.value_of_sym = {}  # Value of symbol.
        >> {l_ab: 3.0}
        self.equations = {}    # Simplified equations. Replace sym with value of symbol's value already known.
        >> {a + b - c: a -5}    # Suppose that b, c already known.
        self.solved = True   # If not solved, then solve.
        """
        super(NewEquation, self).__init__(name)
        self.attr_GDL = attr_GDL
        self.solved = True
        self.sym_of_attr = {}
        self.attr_of_sym = {}
        self.value_of_sym = {}
        self.equations = {}

    def add(self, item, premise, theorem):
        """reload add() of parent class <Condition> to adapt equation's operation."""
        if item not in self.get_item_by_id.values() and -item not in self.get_item_by_id.values():
            added, _id = super().add(item, premise, theorem)
            self.equations[item] = item
            self.solved = False
            return added, _id
        return False, None

    def get_sym_of_attr(self, item, attr):
        """
        Get symbolic representation of item's attribution.
        :param item: tuple, such as ('A', 'B')
        :param attr: attr's name, such as Length
        :return: sym
        """
        if (item, attr) not in self.sym_of_attr:  # No symbolic representation, initialize one.
            if self.attr_GDL[attr]["negative"] == "True":  # Judge whether sym can be negative.
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower())
            else:
                sym = symbols(self.attr_GDL[attr]["sym"] + "_" + "".join(item).lower(), positive=True)

            self.value_of_sym[sym] = None  # init symbol's value
            self.sym_of_attr[(item, attr)] = sym  # add sym
            self.attr_of_sym[sym] = (item, attr)

            if self.attr_GDL[attr]["attr_multi"] == "True":
                l = len(item)
                for bias in range(1, l):
                    extended_item = []  # extend item
                    for i in range(l):
                        extended_item.append(item[(i + bias) % l])
                    self.sym_of_attr[(tuple(extended_item), attr)] = sym  # multi representation
                    self.attr_of_sym[sym] = (tuple(extended_item), attr)

            return sym

        return self.sym_of_attr[(item, attr)]

    def _set_value_of_sym(self, sym, value, premise, theorem):
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

    def get_minimum_equations(self, target_expr):  # 返回求解target_expr依赖的最小方程组
        sym_set = target_expr.free_symbols  # 方程组涉及到的符号
        min_equations = []  # 最小方程组
        premise = []  # 每个方程的index，作为target_expr求解结果的premise

        self._simplify_equations()  # simplify equations before return minimum equations

        # 循环添加依赖方程，得到最小方程组
        update = True
        while update:
            update = False
            for sym in sym_set:
                if self.value_of_sym[sym] is None:  # 如果sym的值未求出，需要添加包含sym的依赖方程
                    for key in self.equations:  # 添加简单依赖方程
                        if sym in self.equations[key].free_symbols and self.equations[key] not in min_equations:
                            min_equations.append(self.equations[key])
                            premise.append(self.get_id_by_item[key])
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号(未化简的原方程的所有符号)
                            update = True

        for sym in sym_set:
            if self.value_of_sym[sym] is not None:
                premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])  # 替换target_expr中的已知sym

        return min_equations, target_expr, premise, sym_set  # 返回化简的target_expr、最小依赖方程组和前提

    def _simplify_equations(self):
        """Simplify all equations based on value replaced."""
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
                    premise = [self.get_id_by_item[key]]
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                    self._set_value_of_sym(target_sym, value, tuple(premise), "solve_eq")
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                self.equations.pop(remove_eq)

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

    @func_set_timeout(10)  # 限时10s
    def solve(self):
        """Solve the equation contained in the question."""
        if self.solved:
            return

        update = True
        while update:
            update = False
            self._simplify_equations()  # simplify equations before solving

            sym_set = []
            for equation in list(self.equations.values()):    # all symbols that have not been solved
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
                    for key in result.keys():    # save solved value
                        if self.value_of_sym[key] is None \
                                and (isinstance(result[key], Float) or isinstance(result[key], Integer)):
                            self._set_value_of_sym(key, float(result[key]), tuple(premise), "solve_eq")
                            update = True

        self.solved = True

    def solve_target(self, target_expr):
        """
        Solve target expression of symbolic form.
        >> equations = [a - b, b - c, c - 1]
        >> solve_target(a)
        1
        """
        if target_expr in self.get_item_by_id.values() or \
                -target_expr in self.get_item_by_id.values():  # It is a known equation and does not need to solve it.
            return 0.0, [self.get_id_by_item[target_expr]]

        premise = []
        for sym in target_expr.free_symbols:  # Solved only using value replacement
            if self.value_of_sym[sym] is not None:
                premise.append(self.get_id_by_item[sym - self.value_of_sym[sym]])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])
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

    def get_expr_from_tree(self, tree, replaced=False, letters=None):
        """
        Recursively trans expr_tree to symbolic algebraic expression.
        :param tree: An expression in the form of a list tree.
        :param replaced: Optional. Set True when tree's item is expressed by vars.
        :param letters: Optional. Letters that will replace vars.
        >> get_expr_from_tree(['Length', ['T', 'R']], equation)
        l_tr
        >> get_expr_from_tree(['Add', [['Length', ['Z', 'X']], '2*x-14']], equation)
        2.0*f_x + l_zx - 14.0
        >> get_expr_from_tree(['Sin', [['Measure', ['Z', 'X', 'Y']]]], equation)
        sin(pi*m_zxy/180)
        """
        if not isinstance(tree, list):  # expr
            return self._parse_expr(tree)

        if tree[0] in self.attr_GDL:  # attr
            if not replaced:
                return self.get_sym_of_attr(tuple(tree[1]), tree[0])
            else:
                replaced_item = [letters[i] for i in tree[1]]
                return self.get_sym_of_attr(tuple(replaced_item), tree[0])

        if tree[0] == "Add":  # operate
            result = 0
            for item in tree[1]:
                result += self.get_expr_from_tree(item, replaced, letters)
            return result
        elif tree[0] == "Sub":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) -\
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Mul":
            result = 1
            for item in tree[1]:
                result *= self.get_expr_from_tree(item, replaced, letters)
            return result
        elif tree[0] == "Div":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) / \
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Pow":
            return self.get_expr_from_tree(tree[1][0], replaced, letters) **\
                   self.get_expr_from_tree(tree[1][1], replaced, letters)
        elif tree[0] == "Sin":
            return sin(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        elif tree[0] == "Cos":
            return cos(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        elif tree[0] == "Tan":
            return tan(self.get_expr_from_tree(tree[1][0], replaced, letters) * pi / 180)
        else:
            raise RuntimeException("OperatorNotDefined",
                                   "No operation {}, please check your expression.".format(tree[0]))

    def get_equation_from_tree(self, tree, replaced=False, letters=None):
        """Refer to function <get_expr_from_tree>."""
        left_expr = self.get_expr_from_tree(tree[0], replaced, letters)
        right_expr = self.get_expr_from_tree(tree[1], replaced, letters)
        return left_expr - right_expr

    def _parse_expr(self, expr):
        """Parse the expression in <str> form into <symbolic> form"""
        expr_list = idt_expr.parseString(expr + "~").asList()
        expr_stack = []
        operator_stack = ["~"]  # 栈底元素

        i = 0
        while i < len(expr_list):
            unit = expr_list[i]
            if unit in operator:  # 运算符
                if stack_priority[operator_stack[-1]] < outside_priority[unit]:
                    operator_stack.append(unit)
                    i = i + 1
                else:
                    operator_unit = operator_stack.pop()
                    if operator_unit == "+":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 + expr_2)
                    elif operator_unit == "-":
                        expr_2 = expr_stack.pop()
                        expr_1 = 0 if len(expr_stack) == 0 else expr_stack.pop()
                        expr_stack.append(expr_1 - expr_2)
                    elif operator_unit == "*":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 * expr_2)
                    elif operator_unit == "/":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 / expr_2)
                    elif operator_unit == "^":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 ** expr_2)
                    elif operator_unit == "{":  # 只有unit为"}"，才能到达这个判断
                        i = i + 1
                    elif operator_unit == "@":  # sin
                        expr_1 = expr_stack.pop()
                        expr_stack.append(sin(expr_1))
                    elif operator_unit == "#":  # cos
                        expr_1 = expr_stack.pop()
                        expr_stack.append(cos(expr_1))
                    elif operator_unit == "$":  # tan
                        expr_1 = expr_stack.pop()
                        expr_stack.append(tan(expr_1))
                    elif operator_unit == "~":  # 只有unit为"~"，才能到达这个判断，表示表达式处理完成
                        break
            else:  # 实数或符号
                unit = self.get_sym_of_attr((unit,), "Free") if unit.isalpha() else float(unit)
                expr_stack.append(unit)
                i = i + 1

        return expr_stack.pop()

    def angle_alignment(self, angles, collinear):
        for angle in angles:
            if (angle, "Measure") in self.sym_of_attr:
                continue
            sym = self.get_sym_of_attr(angle, "Measure")

            a, v, b = angle
            a_points = []    # Points collinear with a and on the same side with a
            b_points = []
            for coll in collinear:
                if v in coll and a in coll:
                    if coll.index(v) < coll.index(a):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            a_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            a_points.append(coll[i])
                            i += 1
                    break
            if len(a_points) == 0:
                a_points.append(a)
            for coll in collinear:
                if v in coll and b in coll:
                    if coll.index(v) < coll.index(b):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            b_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            b_points.append(coll[i])
                            i += 1
                    break
            if len(b_points) == 0:
                b_points.append(b)

            if len(a_points) == 1 and len(b_points) == 1:  # 角只有一种表示
                continue

            same_angles = []
            for a_point in a_points:
                for b_point in b_points:
                    same_angles.append((a_point, v, b_point))  # 相同的角设置一样的符号

            for same_angle in same_angles:
                self.sym_of_attr[(same_angle, "Measure")] = sym
                self.attr_of_sym[sym] = (same_angle, "Measure")
