from problem import Problem
from theorem import Theorem
from facts import AttributionType as aType
from facts import TargetType as tType
from facts import EquationType as eType
from facts import FormalLanguage
from sympy import sin, cos, tan, pi
from utility import PreParse as pp
import time

"""后面改进
2022.09.07
1.float、integer、symbols，sympy求解的结果形式不一样（详见test），看看如何统一(不如就统一为实数)
2.要转化的数据集：①interGPS的  ②几何瑰宝1有关三角形的
2022.09.08
1.去掉多于代码，只考虑三角形的
2.编写完所有定理
3.logic 转化为 FL 部分代码 编写
4.parse 部分 改写代码更易读
"""


class Solver:

    def __init__(self):
        self.last_time = time.time()
        self.problem = None
        self.problem_define_map = None
        self.theorem_map = {
            1: Theorem.nous_1_extend_shape,
            2: Theorem.nous_2_extend_shape_and_area_addition,
            3: Theorem.nous_3_extend_line_addition,
            4: Theorem.nous_4_extend_angle_addition,
            5: Theorem.nous_5_extend_flat_angle,
            21: Theorem.theorem_21_triangle_property_angle_sum,
            22: Theorem.theorem_22_right_triangle_pythagorean
            }

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):  #
        self.last_time = time.time()

        if self.problem is None:  # 第一次，初始化
            self.problem = Problem(problem_index,
                                   construction_fls, text_fls, image_fls, target_fls,
                                   theorem_seqs, answer)
        else:    # 更新题目
            self.problem.new_problem(problem_index,
                                     construction_fls, text_fls, image_fls, target_fls,
                                     theorem_seqs, answer)

        self.parse()  # 解析形式化语句到logic形式
        self.time_cons("init_problem, parse_and_expand_fl, first_solve")  # 初始化问题、解析语句、扩充语句和求解方程耗时

    def parse(self):
        """------构图语句------"""
        for fl in pp.pre_parse_fls(self.problem.fl.construction_fls):
            if fl[0] == "Shape":  # 基本构图图形
                self.problem.define_shape(fl[1], [-1], -1)
            elif fl[0] == "Collinear":  # 共线
                self.problem.define_collinear(fl[1], [-1], -1)
            else:
                print("Unknown predicate: {}".format(fl[0]))    # 之后可以改成 throw error

        """------角一致化------"""
        self.problem.angle_representation_alignment()  # 使角的符号表示一致

        """------题目条件------"""
        for fl in pp.pre_parse_fls(self.problem.fl.text_fls + self.problem.fl.image_fls):
            if fl[0] == "Equal":    # Equal
                self.problem.define_equation(self._generate_expr(fl), eType.basic, [-1], -1)
            elif fl[0] == "Point":    # Entity
                self.problem.define_point(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "Line":
                self.problem.define_line(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "Angle":
                self.problem.define_angle(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "Triangle":
                self.problem.define_triangle(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "RightTriangle":
                self.problem.define_right_triangle(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "IsoscelesTriangle":
                self.problem.define_isosceles_triangle(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "EquilateralTriangle":
                self.problem.define_equilateral_triangle(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "Polygon":
                self.problem.define_polygon(fl[1], eType.basic, [-1], -1)
            elif fl[0] == "Midpoint":    # Entity Relation
                self.problem.define_midpoint(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Intersect":
                self.problem.define_intersect(([fl[1][1], fl[2][1], fl[3][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Parallel":
                self.problem.define_parallel(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Perpendicular":
                self.problem.define_perpendicular(([fl[1][1], fl[2][1], fl[3][1]]), eType.basic, [-1], -1)
            elif fl[0] == "PerpendicularBisector":
                self.problem.define_perpendicular_bisector(([fl[1][1], fl[2][1], fl[3][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Bisector":
                self.problem.define_bisector(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Median":
                self.problem.define_median(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "IsAltitude":
                self.problem.define_is_altitude(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Neutrality":
                self.problem.define_neutrality(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Circumcenter":
                self.problem.define_circumcenter(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Incenter":
                self.problem.define_incenter(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Centroid":
                self.problem.define_centroid(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Orthocenter":
                self.problem.define_orthocenter(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Congruent":
                self.problem.define_congruent(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            elif fl[0] == "Similar":
                self.problem.define_similar(([fl[1][1], fl[2][1]]), eType.basic, [-1], -1)
            else:
                print("Unknown predicate: {}".format(fl[0]))

        """------解题目标------"""
        target_fls = pp.pre_parse_fls(self.problem.fl.target_fls)
        for fl in target_fls:
            if fl[0] == "Find":
                self._parse_find(fl[1])
            else:
                print("Unknown predicate: {}".format(fl[0]))

    def _generate_expr(self, fl):  # 将FL解析成代数表达式
        if fl[0] == "Length":  # 生成属性的符号表示
            self.problem.define_line(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.LL, fl[1][1]))
        elif fl[0] == "Measure":
            self.problem.define_angle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.MA, fl[1][1]))
        elif fl[0] == "Area":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.AS, fl[1][1]))
        elif fl[0] == "Perimeter":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.PT, fl[1][1]))
        elif fl[0] == "Altitude":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.AT, fl[1][1]))
        elif fl[0] == "Add":  # 生成运算的符号表示
            return self._generate_expr(fl[1]) + self._generate_expr(fl[2])
        elif fl[0] == "Sub":
            return self._generate_expr(fl[1]) - self._generate_expr(fl[2])
        elif fl[0] == "Mul":
            return self._generate_expr(fl[1]) * self._generate_expr(fl[2])
        elif fl[0] == "Div":
            return self._generate_expr(fl[1]) / self._generate_expr(fl[2])
        elif fl[0] == "Pow":
            return self._generate_expr(fl[1]) ** self._generate_expr(fl[2])
        elif fl[0] == "Sum":
            para_length = len(fl)
            expr = self._generate_expr(fl[1])
            for i in range(2, para_length):
                expr += self._generate_expr(fl[i])
            return expr
        elif fl[0] == "Avg":
            para_length = len(fl)
            expr = self._generate_expr(fl[1])
            for i in range(2, para_length):
                expr += self._generate_expr(fl[i])
            return expr / (para_length - 1)
        elif fl[0] == "Sin":
            return sin(self._generate_expr(fl[1]) * pi / 180)
        elif fl[0] == "Cos":
            return cos(self._generate_expr(fl[1]) * pi / 180)
        elif fl[0] == "Tan":
            return tan(self._generate_expr(fl[1]) * pi / 180)
        elif fl[0] == "Equal":
            return self._generate_expr(fl[1]) - self._generate_expr(fl[2])
        else:
            return self._parse_expr(fl)

    def _parse_expr(self, expr):  # 解析表达式字符串为list，然后在组装成sym体系下的表示
        expr_list = pp.parse_expr(expr + "~")
        expr_stack = []
        operator_stack = ["~"]  # 栈底元素

        i = 0
        while i < len(expr_list):
            unit = expr_list[i]
            if unit in pp.operator:  # 运算符
                if pp.stack_priority[operator_stack[-1]] < pp.outside_priority[unit]:
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
                unit = self.problem.get_sym_of_attr((aType.F.name, unit)) if unit.isalpha() else float(unit)
                expr_stack.append(unit)
                i = i + 1

        return expr_stack.pop()

    def _generate_tuple(self, fl):
        pass

    def _parse_find(self, fl):  # 解析find
        self.problem.target_count += 1  # 新目标
        self.problem.target_solved.append("unsolved")  # 初始化题目求解情况

        if fl[0] == "Equal":   # 数值-验证型 解题目标
            self.problem.target_type.append(tType.equal)
            target = self.problem.get_sym_of_attr((aType.T.name, str(format(self.problem.target_count))))
            self.problem.target.append([target, target - self._generate_expr(fl), None, None])  # 求解目标 辅助方程 目标值 前提
        elif fl[0] in FormalLanguage.entity_predicates:    # 关系型解题目标(实体)
            self.problem.target_type.append(tType.entity)
            self.problem.target.append([fl[0], self._generate_tuple(fl)])
        elif fl[0] in FormalLanguage.entity_relation_predicates:    # 关系型解题目标(实体关系)
            self.problem.target_type.append(tType.relation)
            self.problem.target.append([fl[0], self._generate_tuple(fl)])
        else:   # 数值-求解型 解题目标
            self.problem.target_type.append(tType.value)
            target = self.problem.get_sym_of_attr((aType.T.name, str(format(self.problem.target_count))))
            self.problem.target.append([target, target - self._generate_expr(fl[0]), None, None])


    def solve(self):
        self.problem.solve_equations()    # 求解初始方程

        for theorem in self.problem.theorem_seqs:  # 应用定理序列
            self.theorem_map[theorem](self.problem)
            self.problem.solve_equations()  # 求解定理添加的方程
            self.time_cons("apply and solve theorem {}".format(theorem))  # 耗时

        for i in range(self.problem.target_count):
            if self.problem.target_type[i] is tType.relation:  # 关系
                if self.problem.target[i][1] in self.problem.relations[self.problem.target[i][0]].items:
                    self.problem.target_solved[i] = "solved"
            elif self.problem.target_type[i] is tType.entity:  # 实体
                if self.problem.target[i][1] in self.problem.entities[self.problem.target[i][0]].items:
                    self.problem.target_solved[i] = "solved"
            else:  # 数量型
                self.problem.target[i][2], self.problem.target[i][3] = \
                    self.problem.solve_equations(self.problem.target[i][0], self.problem.target[i][1])
                if self.problem.target[i][2] is not None:
                    if self.problem.target_type[i] is tType.value and \
                            abs(self.problem.target[i][2] - self._parse_expr(self.problem.answer[i])) < 0.01:  # 有解
                        self.problem.target_solved[i] = "solved"
                    elif self.problem.target[i][2] == 0:  # 验证型，且解为0
                        self.problem.target_solved[i] = "solved"
            self.time_cons("solve target {}".format(i))  # 求解目标耗时

    """------------auxiliary function------------"""

    def time_cons(self, keyword):
        self.problem.solve_time_list.append(
            "\033[32m{}\033[0m time consuming:{:.6f}s".format(keyword, time.time() - self.last_time))
        self.last_time = time.time()
