from problem import Problem
from theorem import Theorem
from facts import AttributionType as aType
from facts import TargetType as tType
from facts import EquationType as eType
from facts import Condition
from facts import FormalLanguage
from sympy import sin, cos, tan, pi
from utility import PreParse as pp
import time

"""
1.完成定理 标注题目 标注到75了  ok
2.68  74 没标注  ok
"""


class Solver:

    def __init__(self):
        self.last_time = time.time()
        self.problem = None
        self.problem_define_map = None
        self.theorem_map = {
            1: Theorem.nous_1_area_addition,
            2: Theorem.nous_2_line_addition,
            3: Theorem.nous_3_angle_addition,
            4: Theorem.nous_4_,
            5: Theorem.nous_5_,
            6: Theorem.nous_6_,
            7: Theorem.nous_7_,
            8: Theorem.nous_8_,
            9: Theorem.nous_9_,
            10: Theorem.nous_10_,
            11: Theorem.auxiliary_11_,
            12: Theorem.auxiliary_12_,
            13: Theorem.auxiliary_13_,
            14: Theorem.auxiliary_14_,
            15: Theorem.auxiliary_15_,
            16: Theorem.auxiliary_16_,
            17: Theorem.auxiliary_17_,
            18: Theorem.auxiliary_18_,
            19: Theorem.auxiliary_19_,
            20: Theorem.auxiliary_20_,
            21: Theorem.theorem_21_triangle_property_angle_sum,
            22: Theorem.theorem_22_triangle_property_equal_line_angle,
            23: Theorem.theorem_23_pythagorean,
            24: Theorem.theorem_24_right_triangle_property_rt,
            25: Theorem.theorem_25_right_triangle_property_special_rt,
            26: Theorem.theorem_26_pythagorean_inverse,
            27: Theorem.theorem_27_right_triangle_judgment,
            28: Theorem.theorem_28_isosceles_triangle_property_angle_equal,
            29: Theorem.theorem_29_isosceles_triangle_property_side_equal,
            30: Theorem.theorem_30_isosceles_triangle_property_line_coincidence,
            31: Theorem.theorem_31_isosceles_triangle_judgment_angle_equal,
            32: Theorem.theorem_32_isosceles_triangle_judgment_side_equal,
            33: Theorem.theorem_33_equilateral_triangle_property_angle_equal,
            34: Theorem.theorem_34_equilateral_triangle_property_side_equal,
            35: Theorem.theorem_35_equilateral_triangle_judgment_angle_equal,
            36: Theorem.theorem_36_equilateral_triangle_judgment_side_equal,
            37: Theorem.theorem_37_equilateral_triangle_judgment_isos_and_angle60,
            38: Theorem.theorem_38_intersect_property,
            39: Theorem.theorem_39_parallel_property,
            40: Theorem.theorem_40_parallel_judgment,
            41: Theorem.theorem_41_perpendicular_judgment,
            42: Theorem.theorem_42_parallel_perpendicular_combination,
            43: Theorem.theorem_43_midpoint_judgment,
            44: Theorem.theorem_44_perpendicular_bisector_property_distance_equal,
            45: Theorem.theorem_45_perpendicular_bisector_judgment,
            46: Theorem.theorem_46_bisector_property_line_ratio,
            47: Theorem.theorem_47_bisector_property_angle_equal,
            48: Theorem.theorem_48_bisector_judgment_angle_equal,
            49: Theorem.theorem_49_median_property,
            50: Theorem.theorem_50_median_judgment,
            51: Theorem.theorem_51_altitude_property,
            52: Theorem.theorem_52_altitude_judgment,
            53: Theorem.theorem_53_neutrality_property_similar,
            54: Theorem.theorem_54_neutrality_property_angle_equal,
            55: Theorem.theorem_55_neutrality_property_line_ratio,
            56: Theorem.theorem_56_neutrality_judgment,
            57: Theorem.theorem_57_circumcenter_property_line_equal,
            58: Theorem.theorem_58_circumcenter_property_intersect,
            59: Theorem.theorem_59_circumcenter_judgment,
            60: Theorem.theorem_60_incenter_property_line_equal,
            61: Theorem.theorem_61_incenter_property_intersect,
            62: Theorem.theorem_62_incenter_property_judgment,
            63: Theorem.theorem_63_centroid_property_line_equal,
            64: Theorem.theorem_64_centroid_property_intersect,
            65: Theorem.theorem_65_centroid_property_judgment,
            66: Theorem.theorem_66_orthocenter_property_line_equal,
            67: Theorem.theorem_67_orthocenter_property_intersect,
            68: Theorem.theorem_68_orthocenter_property_judgment,
            69: Theorem.theorem_69_congruent_property_line_equal,
            70: Theorem.theorem_70_congruent_property_angle_equal,
            71: Theorem.theorem_71_congruent_property_area_equal,
            72: Theorem.theorem_72_congruent_judgment_sss,
            73: Theorem.theorem_73_congruent_judgment_sas,
            74: Theorem.theorem_74_congruent_judgment_asa,
            75: Theorem.theorem_75_congruent_judgment_aas,
            76: Theorem.theorem_76_congruent_judgment_hl,
            77: Theorem.theorem_77_similar_property_angle_equal,
            78: Theorem.theorem_78_similar_property_line_ratio,
            79: Theorem.theorem_79_similar_property_perimeter_ratio,
            80: Theorem.theorem_80_similar_property_area_square_ratio,
            81: Theorem.theorem_81_similar_judgment_sss,
            82: Theorem.theorem_82_similar_judgment_sas,
            83: Theorem.theorem_83_similar_judgment_aa,
            84: Theorem.theorem_84_triangle_perimeter_formula,
            85: Theorem.theorem_85_triangle_area_formula_common,
            86: Theorem.theorem_86_triangle_area_formula_heron,
            87: Theorem.theorem_87_triangle_area_formula_sine,
            88: Theorem.theorem_88_sine,
            89: Theorem.theorem_89_cosine
        }

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):  #
        self.last_time = time.time()

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
        self.problem.target_solved.append("unsolved")  # 初始化题目求解情况

        if fl[0] == "Equal":  # 数值-验证型 解题目标
            self.problem.target_type.append(tType.equal)
            self.problem.target.append([self._generate_expr(fl), None, None])  # 求解目标 求解值 前提
        elif fl[0] in FormalLanguage.entity_predicates:  # 关系型解题目标(实体)
            self.problem.target_type.append(tType.entity)
            self.problem.target.append(
                [Condition.entity_list[FormalLanguage.entity_predicates.index(fl[0])], fl[1], None])
        elif fl[0] in FormalLanguage.entity_relation_predicates:  # 关系型解题目标(实体关系)
            target_list = [fl[1][1], fl[2][1]]
            if fl[0] in ["Intersect", "Perpendicular", "PerpendicularBisector"]:
                target_list.append(fl[3][1])
            self.problem.target_type.append(tType.relation)
            self.problem.target.append(
                [Condition.entity_relation_list[FormalLanguage.entity_relation_predicates.index(fl[0])],
                 tuple(target_list), None])
        else:  # 数值-求解型 解题目标
            self.problem.target_type.append(tType.value)
            self.problem.target.append([self._generate_expr(fl), None, None])

    def solve(self):
        self.problem.solve_equations()  # 求解初始方程
        self.problem.anti_generate_fl_using_logic()    # 通过logic反向生成FL，以供embedding

        for theorem in self.problem.theorem_seqs:  # 应用定理序列
            self.theorem_map[theorem](self.problem)
            self.problem.solve_equations()  # 求解定理添加的方程
            self.problem.anti_generate_fl_using_logic()  # 通过logic反向生成FL，以供embedding
            self.time_cons("apply and solve theorem {}".format(theorem))  # 耗时

        for i in range(self.problem.target_count):
            if self.problem.target_type[i] in [tType.entity, tType.relation]:  # 实体、实体关系
                if self.problem.target[i][1] in self.problem.conditions.item[self.problem.target[i][0]]:
                    self.problem.target_solved[i] = "solved"
            else:  # 数量型
                self.problem.target[i][1], self.problem.target[i][2] =\
                    self.problem.solve_target(self.problem.target[i][0])  # 求解并保存至result
                if self.problem.target[i][1] is not None:
                    if self.problem.target_type[i] is tType.value and \
                            abs(float(self.problem.target[i][1]) - self._parse_expr(self.problem.answer[i])) < 0.01:  # 有解
                        self.problem.target_solved[i] = "solved"
                    elif self.problem.target_type[i] is tType.equal and self.problem.target[i][1] == 0:  # 验证型，且解为0
                        self.problem.target_solved[i] = "solved"
            self.time_cons("solve target {}".format(i))  # 求解目标耗时

    """------------auxiliary function------------"""

    def time_cons(self, keyword):
        self.problem.solve_time_list.append(
            "\033[32m{}\033[0m time consuming:{:.6f}s".format(keyword, time.time() - self.last_time))
        self.last_time = time.time()
