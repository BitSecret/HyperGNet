from pyparsing import alphanums, Forward, Group, Word, Literal, ZeroOrMore
from problem import Problem
from theorem import Theorem
from facts import AttributionType, TargetType
from sympy import sin, cos, tan, solve, symbols
from utility import pre_parse


class Solver:

    def __init__(self, problem_index, formal_languages, theorem_seqs=None):
        self.problem = Problem(problem_index, formal_languages, theorem_seqs)  # 题目

        # 使用字典映射parse函数，代替if-else，加速查找。字典采用hash映射，查找复杂读为O(1)。
        self.parse_func_map = {"Point": self._parse_entity,
                               "Line": self._parse_entity,
                               "Angle": self._parse_entity,
                               "Arc": self._parse_entity,
                               "Shape": self._parse_entity,
                               "Circle": self._parse_entity,
                               "Sector": self._parse_entity,
                               "Triangle": self._parse_entity,
                               "RightTriangle": self._parse_entity,
                               "IsoscelesTriangle": self._parse_entity,
                               "RegularTriangle": self._parse_entity,
                               "Quadrilateral": self._parse_entity,
                               "Trapezoid": self._parse_entity,
                               "IsoscelesTrapezoid": self._parse_entity,
                               "Parallelogram": self._parse_entity,
                               "Rectangle": self._parse_entity,
                               "Rhombus": self._parse_entity,
                               "Kite": self._parse_entity,
                               "Square": self._parse_entity,
                               "Polygon": self._parse_entity,
                               "RegularPolygon": self._parse_entity,
                               "PointOnLine": self._parse_binary_relation,
                               "PointOnArc": self._parse_binary_relation,
                               "PointOnCircle": self._parse_binary_relation,
                               "Midpoint": self._parse_binary_relation,
                               "Circumcenter": self._parse_binary_relation,
                               "Incenter": self._parse_binary_relation,
                               "Centroid": self._parse_binary_relation,
                               "Orthocenter": self._parse_binary_relation,
                               "Parallel": self._parse_binary_relation,
                               "Perpendicular": self._parse_ternary_relation,
                               "PerpendicularBisector": self._parse_ternary_relation,
                               "BisectsAngle": self._parse_binary_relation,
                               "DisjointLineCircle": self._parse_binary_relation,
                               "DisjointCircleCircle": self._parse_binary_relation,
                               "TangentLineCircle": self._parse_ternary_relation,
                               "TangentCircleCircle": self._parse_ternary_relation,
                               "IntersectLineLine": self._parse_ternary_relation,
                               "IntersectLineCircle": self._parse_quaternion_relation,
                               "IntersectCircleCircle": self._parse_quaternion_relation,
                               "Median": self._parse_binary_relation,
                               "HeightOfTriangle": self._parse_binary_relation,
                               "HeightOfTrapezoid": self._parse_binary_relation,
                               "InternallyTangent": self._parse_ternary_relation,
                               "Contain": self._parse_binary_relation,
                               "CircumscribedToTriangle": self._parse_binary_relation,
                               "InscribedInTriangle": self._parse_five_relation,
                               "Congruent": self._parse_binary_relation,
                               "Similar": self._parse_binary_relation,
                               "Equal": self._parse_equal,
                               "Find": self._parse_find}
        self.problem_entity_map = {"Point": self.problem.define_point,
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
                                   "RegularPolygon": self.problem.define_regular_polygon}
        self.problem_binary_relation_map = {"PointOnLine": self.problem.define_point_on_line,
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
                                            "Similar": self.problem.define_similar}
        self.problem_ternary_relation_map = {"Perpendicular": self.problem.define_perpendicular,
                                             "PerpendicularBisector": self.problem.define_perpendicular_bisector,
                                             "TangentLineCircle": self.problem.define_tangent_line_circle,
                                             "TangentCircleCircle": self.problem.define_tangent_circle_circle,
                                             "IntersectLineLine": self.problem.define_intersect_line_line,
                                             "InternallyTangent": self.problem.define_internally_tangent}
        self.problem_quaternion_relation_map = {"IntersectLineCircle": self.problem.define_intersect_line_circle,
                                                "IntersectCircleCircle": self.problem.define_intersect_circle_circle}
        self.problem_five_relation_map = {"InscribedInTriangle": self.problem.define_inscribed_in_triangle}

        # 定理映射
        self.theorem_map = {1: Theorem.theorem_1_xxxx,
                            2: Theorem.theorem_2_xxxx,
                            3: Theorem.theorem_3_xxxx,
                            4: Theorem.theorem_4_xxxx}

        self.parse()  # 解析形式化语句到logic形式

    def parse(self):
        py_expression = Forward()  # 定义解析 formal language 的 表达式
        identifier = Word(alphanums + ' +-*/=.\\{}^_$\'')  # 识别的最小unit 谓词或字母或数字或代数表达式
        arg = Group(py_expression) | identifier  # arg 可能是聚合体，也可能是标志符
        args = arg + ZeroOrMore(Literal(",").suppress() + arg)  # arg组合起来  suppress 的字符不会在结果中出现
        py_expression <<= identifier + Literal("(").suppress() + args + Literal(")").suppress()

        for formal_language in self.problem.formal_languages:
            fl = pre_parse(py_expression.parseString(formal_language).asList())  # 解析formal_language为list形式
            self.parse_func_map[fl[0]](fl)  # 这个警告不用管

    def _parse_entity(self, fl):  # 解析实体
        self.problem_entity_map[fl[0]](fl[1])

    def _parse_binary_relation(self, fl):  # 解析2元关系
        self.problem_binary_relation_map[fl[0]](fl[1][1], fl[2][1])

    def _parse_ternary_relation(self, fl):  # 解析3元关系
        self.problem_ternary_relation_map[fl[0]](fl[1][1], fl[2][1], fl[3][1])

    def _parse_quaternion_relation(self, fl):  # 解析4元关系
        self.problem_quaternion_relation_map[fl[0]](fl[1][1], fl[2][1], fl[3][1], fl[4][1])

    def _parse_five_relation(self, fl):  # 解析5元关系
        self.problem_five_relation_map[fl[0]](fl[1][1], fl[2][1], fl[3][1], fl[4][1], fl[5][1])

    def _parse_equal(self, fl):  # 解析equal
        expr = self._generate_expr(fl[1]) - self._generate_expr(fl[2])
        self.problem.define_equation(expr)

    def _generate_expr(self, fl):  # 将FL解析成代数表达式
        if fl[0] == "Length":  # 生成属性的符号表示
            if fl[1][0] == "Line":
                return self.problem.get_sym_of_attr((AttributionType.LL, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((AttributionType.LA, fl[1][1]))
        elif fl[0] == "Degree":
            if fl[1][0] == "Angle":
                return self.problem.get_sym_of_attr((AttributionType.DA, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((AttributionType.DS, fl[1][1]))
        elif fl[0] == "Radius":
            if fl[1][0] == "Arc":
                return self.problem.get_sym_of_attr((AttributionType.RA, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((AttributionType.RC, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((AttributionType.RS, fl[1][1]))
        elif fl[0] == "Diameter":
            return self.problem.get_sym_of_attr((AttributionType.DC, fl[1][1]))
        elif fl[0] == "Perimeter":
            if fl[1][0] == "Triangle":
                return self.problem.get_sym_of_attr((AttributionType.PT, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((AttributionType.PC, fl[1][1]))
            elif fl[1][0] == "Sector":
                return self.problem.get_sym_of_attr((AttributionType.PS, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                return self.problem.get_sym_of_attr((AttributionType.PQ, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((AttributionType.PP, fl[1][1]))
        elif fl[0] == "Area":
            if fl[1][0] == "Triangle":
                return self.problem.get_sym_of_attr((AttributionType.AT, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((AttributionType.AC, fl[1][1]))
            elif fl[1][0] == "Sector":
                return self.problem.get_sym_of_attr((AttributionType.AS, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                return self.problem.get_sym_of_attr((AttributionType.AQ, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((AttributionType.AP, fl[1][1]))
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
            return sin(self._generate_expr(fl[1]))
        elif fl[0] == "Cos":
            return cos(self._generate_expr(fl[1]))
        elif fl[0] == "Tan":
            return tan(self._generate_expr(fl[1]))
        elif fl[0].isalpha():  # 如果是字母，生成字母的符号表示
            return self.problem.get_sym_of_attr((AttributionType.F, fl[0]))
        else:  # 数字
            return float(fl)

    def _parse_find(self, fl):  # 解析find
        fl[1] = pre_parse(fl[1])  # 解析目标
        self.problem.target_count += 1  # 新目标
        if fl[1][0] in self.problem.relations.keys():
            self.problem.target_type.append(TargetType.relation)  # 位置关系
            target = []
            for i in range(1, len(fl[1])):
                target.append(fl[1][i][1])
            self.problem.target.append([fl[1][0], tuple(target)])
        else:
            self.problem.target_type.append(TargetType.value)  # 代数关系
            self.problem.target.append(self._generate_expr(fl[1]))

    def solve(self):
        for theorem in self.problem.theorem_seqs:
            self.theorem_map[theorem](self.problem)

    """------------auxiliary function------------"""
