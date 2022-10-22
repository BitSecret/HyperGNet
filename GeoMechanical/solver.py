from problem import Problem
from theorem import TheoremMap
from facts import AttributionType as aType
from facts import TargetType as tType
from facts import EquationType as eType
from facts import Condition
from facts import FormalLanguage
from facts import Target
from sympy import sin, cos, tan, pi
from utility import PreParse as pp
from utility import Utility as util
import time

"""
"""


class Solver:

    def __init__(self):
        self.last_time = time.time()
        self.problem = None
        self.problem_define_map = None

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):  #
        self.last_time = time.time()
        TheoremMap.count = 0

        if self.problem is None:  # 第一次，初始化
            self.problem = Problem(problem_index,
                                   construction_fls, text_fls, image_fls, target_fls,
                                   theorem_seqs, answer)
        else:  # 更新题目
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
                raise RuntimeError("Unknown predicate <{}> when parse construction FL. Please check FL.".format(fl[0]))

        """------构造图形------"""
        self.problem.construct_all_shape()    # 构造所有图形
        self.problem.construct_all_line()  # 构造所有线段
        self.problem.angle_representation_alignment()  # 使角的符号表示一致
        self.problem.flat_angle()  # 赋予平角180°

        """------题目条件------"""
        for fl in pp.pre_parse_fls(self.problem.fl.text_fls + self.problem.fl.image_fls):
            if fl[0] == "Equal":  # Equal
                self.problem.define_equation(self._generate_expr(fl), eType.basic, [-1], -1)
            elif fl[0] == "Point":  # Entity
                self.problem.define_point(fl[1], [-1], -1)
            elif fl[0] == "Line":
                self.problem.define_line(fl[1], [-1], -1)
            elif fl[0] == "Angle":
                self.problem.define_angle(fl[1], [-1], -1)
            elif fl[0] == "Triangle":
                self.problem.define_triangle(fl[1], [-1], -1)
            elif fl[0] == "RightTriangle":
                self.problem.define_right_triangle(fl[1], [-1], -1)
            elif fl[0] == "IsoscelesTriangle":
                self.problem.define_isosceles_triangle(fl[1], [-1], -1)
            elif fl[0] == "EquilateralTriangle":
                self.problem.define_equilateral_triangle(fl[1], [-1], -1)
            elif fl[0] == "Polygon":
                self.problem.define_polygon(fl[1], [-1], -1)
            elif fl[0] == "Midpoint":  # Entity Relation
                self.problem.define_midpoint((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Intersect":
                self.problem.define_intersect((fl[1][1], fl[2][1], fl[3][1]), [-1], -1)
            elif fl[0] == "Parallel":
                self.problem.define_parallel((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "DisorderParallel":
                self.problem.define_parallel((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Perpendicular":
                self.problem.define_perpendicular((fl[1][1], fl[2][1], fl[3][1]), [-1], -1)
            elif fl[0] == "PerpendicularBisector":
                self.problem.define_perpendicular_bisector((fl[1][1], fl[2][1], fl[3][1]), [-1], -1)
            elif fl[0] == "Bisector":
                self.problem.define_bisector((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Median":
                self.problem.define_median((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "IsAltitude":
                self.problem.define_is_altitude((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Neutrality":
                self.problem.define_neutrality((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Circumcenter":
                self.problem.define_circumcenter((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Incenter":
                self.problem.define_incenter((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Centroid":
                self.problem.define_centroid((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Orthocenter":
                self.problem.define_orthocenter((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Congruent":
                self.problem.define_congruent((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "Similar":
                self.problem.define_similar((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "MirrorCongruent":
                self.problem.define_mirror_congruent((fl[1][1], fl[2][1]), [-1], -1)
            elif fl[0] == "MirrorSimilar":
                self.problem.define_mirror_similar((fl[1][1], fl[2][1]), [-1], -1)
            else:
                raise RuntimeError("Unknown predicate <{}> when parse text/image FL. Please check FL.".format(fl[0]))

        """------解题目标------"""
        target_fls = pp.pre_parse_fls(self.problem.fl.target_fls)
        for fl in target_fls:
            if fl[0] == "Find":
                self._parse_find(fl[1])
            else:
                raise RuntimeError("Unknown predicate <{}> when parse target FL. Please check FL.".format(fl[0]))

    def _generate_expr(self, fl):  # 将FL解析成代数表达式
        if fl[0] == "Length":  # 生成属性的符号表示
            self.problem.define_line(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr(fl[1][1], aType.LL)
        elif fl[0] == "Measure":
            self.problem.define_angle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr(fl[1][1], aType.MA)
        elif fl[0] == "Area":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr(fl[1][1], aType.AS)
        elif fl[0] == "Perimeter":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr(fl[1][1], aType.PT)
        elif fl[0] == "Altitude":
            self.problem.define_triangle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr(fl[1][1], aType.AT)
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
            result = self._parse_expr(fl)
            if isinstance(result, float):
                return float(str(result)[0:10])
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
                unit = self.problem.get_sym_of_attr(unit, aType.F) if unit.isalpha() else float(unit)
                expr_stack.append(unit)
                i = i + 1

        return expr_stack.pop()

    def _parse_find(self, fl):  # 解析find
        self.problem.target_count += 1  # 新目标
        target = Target()

        if fl[0] == "Equal":  # 数值-验证型 解题目标
            target.target_type = tType.equal
            target.target = self._generate_expr(fl)
        elif fl[0] in FormalLanguage.entity_predicates:  # 关系型解题目标(实体)
            target.target_type = tType.entity
            target.target = Condition.entity_list[FormalLanguage.entity_predicates.index(fl[0])]    # cType
            target.target_solved = fl[1]    # tuple
        elif fl[0] in FormalLanguage.entity_relation_predicates:  # 关系型解题目标(实体关系)
            target_list = [fl[1][1], fl[2][1]]
            if fl[0] in ["Intersect", "Perpendicular", "PerpendicularBisector"]:   # 三元关系
                target_list.append(fl[3][1])
            target.target_type = tType.relation
            target.target = Condition.entity_relation_list[FormalLanguage.entity_relation_predicates.index(fl[0])]
            target.target_solved = tuple(target_list)
        else:  # 数值-求解型 解题目标
            target.target_type = tType.value
            target.target = self._generate_expr(fl)

        self.problem.targets.append(target)    # 添加到解题目标

    def solve(self):
        self.problem.solve_equations()  # 求解初始方程
        self.problem.anti_generate_fl_using_logic()    # 通过logic反向生成FL，以供embedding

        for theorem in self.problem.theorem_seqs:  # 应用定理序列
            TheoremMap.theorem_map[theorem](self.problem)
            self.problem.solve_equations()  # 求解定理添加的方程
            self.problem.anti_generate_fl_using_logic()  # 通过logic反向生成FL，以供embedding
            self.time_cons("apply and solve theorem {}".format(theorem))  # 耗时

        for i in range(self.problem.target_count):
            target = self.problem.targets[i]
            if target.target_type in [tType.entity, tType.relation]:  # 实体、实体关系
                if target.solved_answer in self.problem.conditions.item[target.target]:
                    target.target_solved = True
                    target.premise = self.problem.conditions.get_premise(target.solved_answer, target.target)
                    target.theorem = self.problem.conditions.get_theorem(target.solved_answer, target.target)
            else:  # 数量型
                target.solved_answer, target.premise = self.problem.solve_target(target.target)  # 求解并保存求解结果
                target.theorem = -3
                if target.solved_answer is not None:
                    if target.target_type is tType.value and \
                            util.equal(target.solved_answer, self._parse_expr(self.problem.answer[i])):  # 有解
                        target.target_solved = True
                    elif target.target_type is tType.equal and util.equal(target.solved_answer, 0):  # 验证型，且解为0
                        target.target_solved = True

            self.time_cons("solve target {}".format(i))  # 求解目标耗时

        self.problem.s_tree.generate_tree(self.problem)  # 求解结束后生成求解树

    """------------auxiliary function------------"""

    def time_cons(self, keyword):
        self.problem.solve_time_list.append(
            "\033[32m{}\033[0m time consuming:{:.6f}s".format(keyword, time.time() - self.last_time))
        self.last_time = time.time()
