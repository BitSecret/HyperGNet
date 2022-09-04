import copy
from problem import Problem
from theorem import Theorem
from facts import AttributionType as aType
from facts import TargetType as tType
from facts import EquationType as eType
from sympy import sin, cos, tan, pi
from utility import PreParse as pp
import time

"""后面改进
1.float、integer、symbols，sympy求解的结果形式不一样（详见test），看看如何统一(不如就统一为实数)
2.Theorem部分还没重构完成，这个与题目的整理同步进行"""
"""
1.把 extend 移植到theorem上
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
            21: Theorem.theorem_21_pythagorean,
            22: Theorem.theorem_22_pythagorean_inverse,
            23: Theorem.theorem_23_right_triangle_determine,
            24: Theorem.theorem_24_transitivity_of_parallel,
            25: Theorem.theorem_25_transitivity_of_perpendicular,
            26: Theorem.theorem_26_similar_triangle,
            27: Theorem.theorem_27_similar_triangle_determine,
            28: Theorem.theorem_28_congruent_triangle,
            29: Theorem.theorem_29_congruent_triangle_determine,
            30: Theorem.theorem_30_triangle,
            31: Theorem.theorem_31_isosceles_triangle,
            32: Theorem.theorem_32_isosceles_triangle_determine,
            33: Theorem.theorem_33_tangent_radius,
            34: Theorem.theorem_34_center_and_circumference_angle,
            35: Theorem.theorem_35_parallel,
            36: Theorem.theorem_36_parallel_inverse,
            37: Theorem.theorem_37_flat_angle,
            38: Theorem.theorem_38_intersecting_chord,
            39: Theorem.theorem_39_polygon,
            40: Theorem.theorem_40_angle_bisector,
            41: Theorem.theorem_41_sine,
            42: Theorem.theorem_42_cosine,
            43: Theorem.theorem_43_perimeter_of_tri,
            44: Theorem.theorem_44_perimeter_of_shape}

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):  #
        self.last_time = time.time()

        if self.problem is None:  # 第一次，初始化
            self.problem = Problem(problem_index,
                                   construction_fls, text_fls, image_fls, target_fls,
                                   theorem_seqs, answer)  # 题目
            self.problem_define_map = {"Point": self.problem.define_point,
                                       "Line": self.problem.define_line,
                                       "Angle": self.problem.define_angle,
                                       "Arc": self.problem.define_arc,
                                       "Shape": self.problem.define_shape,
                                       "Circle": self.problem.define_circle,
                                       "Sector": self.problem.define_sector,
                                       "Triangle": self.problem.define_triangle,
                                       "RightTriangle": self.problem.define_right_triangle,
                                       "IsoscelesTriangle": self.problem.define_isosceles_triangle,
                                       "RegularTriangle": self.problem.define_regular_triangle,
                                       "Quadrilateral": self.problem.define_quadrilateral,
                                       "Trapezoid": self.problem.define_trapezoid,
                                       "IsoscelesTrapezoid": self.problem.define_isosceles_trapezoid,
                                       "Parallelogram": self.problem.define_parallelogram,
                                       "Rectangle": self.problem.define_rectangle,
                                       "Rhombus": self.problem.define_rhombus,
                                       "Kite": self.problem.define_kite,
                                       "Square": self.problem.define_square,
                                       "Polygon": self.problem.define_polygon,
                                       "RegularPolygon": self.problem.define_regular_polygon,
                                       "Collinear": self.problem.define_collinear,
                                       "PointOnLine": self.problem.define_point_on_line,
                                       "PointOnArc": self.problem.define_point_on_arc,
                                       "PointOnCircle": self.problem.define_point_on_circle,
                                       "Midpoint": self.problem.define_midpoint,
                                       "Circumcenter": self.problem.define_circumcenter,
                                       "Incenter": self.problem.define_incenter,
                                       "Centroid": self.problem.define_centroid,
                                       "Orthocenter": self.problem.define_orthocenter,
                                       "Parallel": self.problem.define_parallel,
                                       "BisectsAngle": self.problem.define_bisects_angle,
                                       "DisjointLineCircle": self.problem.define_disjoint_line_circle,
                                       "DisjointCircleCircle": self.problem.define_disjoint_circle_circle,
                                       "Median": self.problem.define_median,
                                       "HeightOfTriangle": self.problem.define_height_triangle,
                                       "HeightOfTrapezoid": self.problem.define_height_trapezoid,
                                       "Contain": self.problem.define_contain,
                                       "CircumscribedToTriangle": self.problem.define_circumscribed_to_triangle,
                                       "Congruent": self.problem.define_congruent,
                                       "Similar": self.problem.define_similar,
                                       "Chord": self.problem.define_chord,
                                       "Perpendicular": self.problem.define_perpendicular,
                                       "PerpendicularBisector": self.problem.define_perpendicular_bisector,
                                       "TangentLineCircle": self.problem.define_tangent_line_circle,
                                       "TangentCircleCircle": self.problem.define_tangent_circle_circle,
                                       "IntersectLineLine": self.problem.define_intersect_line_line,
                                       "InternallyTangent": self.problem.define_internally_tangent,
                                       "IntersectLineCircle": self.problem.define_intersect_line_circle,
                                       "IntersectCircleCircle": self.problem.define_intersect_circle_circle,
                                       "InscribedInTriangle": self.problem.define_inscribed_in_triangle
                                       }
        else:
            self.problem.new_problem(problem_index,
                                     construction_fls, text_fls, image_fls, target_fls,
                                     theorem_seqs, answer)

        self.parse()  # 解析形式化语句到logic形式
        self.time_cons("init_problem, parse_and_expand_fl, first_solve")  # 初始化问题、解析语句、扩充语句和求解方程耗时

    def parse(self):
        """------构图语句------"""
        construction_fls = pp.pre_parse_fls(copy.copy(self.problem.fl.construction_fls))
        for fl in construction_fls:
            if fl[0] == "Shape":  # 基本构图图形
                self.problem.define_shape(fl[1], [-1], -1)
            else:  # 共线
                self.problem.define_collinear(fl[1], [-1], -1)

        self.problem.angle_representation_alignment()    # 使角的符号表示一致

        """------题目条件------"""
        text_fls = pp.pre_parse_fls(copy.copy(self.problem.fl.text_fls))
        image_fls = pp.pre_parse_fls(copy.copy(self.problem.fl.image_fls))
        for fl in text_fls + image_fls:
            if fl[0] == "Equal":  # 数量关系
                self.problem.define_equation(self._generate_expr(fl), eType.basic, [-1], -1)
            else:  # 实体定义、位置关系定义
                self.problem_define_map[fl[0]](fl[1], [-1], -1)

        """------解题目标------"""
        target_fls = pp.pre_parse_fls(copy.copy(self.problem.fl.target_fls))
        for fl in target_fls:
            self._parse_find(fl[1])

        """------求解初始方程------"""
        self.problem.solve_equations()

    def _generate_expr(self, fl):  # 将FL解析成代数表达式
        if fl[0] == "Length":  # 生成属性的符号表示
            if fl[1][0] == "Line":
                self.problem.define_line(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.LL.name, fl[1][1]))
            else:
                self.problem.define_arc(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.LA.name, fl[1][1]))
        elif fl[0] == "Degree":
            if fl[1][0] == "Angle":
                self.problem.define_angle(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.DA.name, fl[1][1]))
            else:
                self.problem.define_sector(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.DS.name, fl[1][1]))
        elif fl[0] == "Radius":
            if fl[1][0] == "Arc":
                self.problem.define_arc(fl[1][1], [-1], -1)
            elif fl[1][0] == "Circle":
                self.problem.define_circle(fl[1][1], [-1], -1)
            else:
                self.problem.define_sector(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.R.name, fl[1][1]))
        elif fl[0] == "Diameter":
            self.problem.define_circle(fl[1][1], [-1], -1)
            return self.problem.get_sym_of_attr((aType.DC.name, fl[1][1]))
        elif fl[0] == "Perimeter":
            if fl[1][0] == "Triangle":
                self.problem.define_triangle(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.P.name, fl[1][1]))
            elif fl[1][0] == "Circle":
                self.problem.define_circle(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.P.name, fl[1][1]))
            elif fl[1][0] == "Sector":
                self.problem.define_sector(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.PS.name, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                self.problem.define_quadrilateral(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.P.name, fl[1][1]))
            else:
                self.problem.define_polygon(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.P.name, fl[1][1]))
        elif fl[0] == "Area":
            if fl[1][0] == "Triangle":
                self.problem.define_triangle(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.A.name, fl[1][1]))
            elif fl[1][0] == "Circle":
                self.problem.define_circle(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.A.name, fl[1][1]))
            elif fl[1][0] == "Sector":
                self.problem.define_sector(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.AS.name, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                self.problem.define_quadrilateral(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.A.name, fl[1][1]))
            else:
                self.problem.define_polygon(fl[1][1], [-1], -1)
                return self.problem.get_sym_of_attr((aType.A.name, fl[1][1]))
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

    def _parse_find(self, fl):  # 解析find
        self.problem.target_count += 1  # 新目标
        self.problem.target_solved.append("unsolved")  # 初始化题目求解情况

        if fl[0] == "Equal":  # 验证型解题目标
            self.problem.target_type.append(tType.equal)
            target = self.problem.get_sym_of_attr((aType.T.name, str(format(self.problem.target_count))))
            self.problem.target.append([target, target - self._generate_expr(fl), None, None])  # 求解目标 辅助方程 目标值 前提
        elif fl[0] == "Value":  # 求值型解题目标
            self.problem.target_type.append(tType.value)
            target = self.problem.get_sym_of_attr((aType.T.name, str(format(self.problem.target_count))))
            self.problem.target.append([target, target - self._generate_expr(fl[1]), None, None])  # 求解目标 辅助方程 目标值 前提
        else:  # 关系型解题目标
            if fl[0] in self.problem.entities.keys():
                self.problem.target_type.append(tType.entity)
            else:
                self.problem.target_type.append(tType.relation)
            self.problem.target.append([fl[0], fl[1]])

    def solve(self):
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
        print("\033[32m{}\033[0m time consuming:{:.6f}s".format(keyword, time.time() - self.last_time))
        self.last_time = time.time()
