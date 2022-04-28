from problem import Problem
from theorem import Theorem
from facts import AttributionType as aType
from facts import TargetType as tType
from sympy import solve, Float, sin, cos, tan
from utility import pre_parse
from utility import RegularExpression as reE

"""后面改进
1.方程式求解问题：①每次求解后要不要简化方程（分为已解决的和未解决的）②解方程的前提条件（最小可求解方程组），如何找出来
2.float、integer、symbols，sympy求解的结果形式不一样（详见test），看看如何统一(不如就统一为实数)"""


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
        self.theorem_map = {1: Theorem.theorem_1_pythagorean,
                            2: Theorem.theorem_2_pythagorean_inverse,
                            3: Theorem.theorem_3_transitivity_of_parallel,
                            4: Theorem.theorem_4_similar_triangle,
                            5: Theorem.theorem_5_similar_triangle_inverse,
                            6: Theorem.theorem_6_congruent_triangle,
                            7: Theorem.theorem_7_congruent_triangle_inverse}

        self.parse()  # 解析形式化语句到logic形式

    def parse(self):
        for formal_language in self.problem.formal_languages:
            fl = reE.get_fl(formal_language)    # 解析形式化语言成树状结构，list形式
            print(fl)
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
        self.problem.define_equation(self._generate_expr(fl))

    def _generate_expr(self, fl):  # 将FL解析成代数表达式
        if fl[0] == "Length":  # 生成属性的符号表示
            if fl[1][0] == "Line":
                return self.problem.get_sym_of_attr((aType.LL.name, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((aType.LA.name, fl[1][1]))
        elif fl[0] == "Degree":
            if fl[1][0] == "Angle":
                return self.problem.get_sym_of_attr((aType.DA.name, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((aType.DS.name, fl[1][1]))
        elif fl[0] == "Radius":
            if fl[1][0] == "Arc":
                return self.problem.get_sym_of_attr((aType.RA.name, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((aType.RC.name, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((aType.RS.name, fl[1][1]))
        elif fl[0] == "Diameter":
            return self.problem.get_sym_of_attr((aType.DC.name, fl[1][1]))
        elif fl[0] == "Perimeter":
            if fl[1][0] == "Triangle":
                return self.problem.get_sym_of_attr((aType.PT.name, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((aType.PC.name, fl[1][1]))
            elif fl[1][0] == "Sector":
                return self.problem.get_sym_of_attr((aType.PS.name, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                return self.problem.get_sym_of_attr((aType.PQ.name, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((aType.PP.name, fl[1][1]))
        elif fl[0] == "Area":
            if fl[1][0] == "Triangle":
                return self.problem.get_sym_of_attr((aType.AT.name, fl[1][1]))
            elif fl[1][0] == "Circle":
                return self.problem.get_sym_of_attr((aType.AC.name, fl[1][1]))
            elif fl[1][0] == "Sector":
                return self.problem.get_sym_of_attr((aType.AS.name, fl[1][1]))
            elif fl[1][0] == "Quadrilateral":
                return self.problem.get_sym_of_attr((aType.AQ.name, fl[1][1]))
            else:
                return self.problem.get_sym_of_attr((aType.AP.name, fl[1][1]))
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
        elif fl[0] == "Equal":
            return self._generate_expr(fl[1]) - self._generate_expr(fl[2])
        else:
            return self._parse_expr(fl)

    def _parse_expr(self, expr):    # 解析表达式字符串为list，然后在组装成sym体系下的表示
        expr_list = reE.get_expr(expr + "~")
        expr_stack = []
        operator_stack = ["~"]  # 栈底元素

        i = 0
        while i < len(expr_list):
            unit = expr_list[i]
            if unit in reE.operator:  # 运算符
                if reE.stack_priority[operator_stack[-1]] < reE.outside_priority[unit]:
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
        fl[1] = pre_parse(fl[1])  # 解析目标
        self.problem.target_count += 1  # 新目标
        self.problem.target_solved.append("unsolved")  # 初始化题目求解情况
        if fl[1][0] in self.problem.relations.keys():  # 位置关系，联系
            self.problem.target_type.append(tType.relation)
            target = []
            for i in range(1, len(fl[1])):
                target.append(fl[1][i][1])
            self.problem.target.append([fl[1][0], tuple(target)])  # ["Parallel", ("AB", "CD")]
        elif fl[1][0] in self.problem.entities.keys():  # 位置关系，实体
            self.problem.target_type.append(tType.entity)
            self.problem.target.append([fl[1][0], fl[1][1]])
        else:
            if fl[1][0] == "Equal":
                self.problem.target_type.append(tType.equal)  # 代数关系，equal
            else:
                self.problem.target_type.append(tType.value)  # 代数关系,求值
            target = self.problem.get_sym_of_attr((aType.T.name, str(format(self.problem.target_count))))
            self.problem.target.append([target, target - self._generate_expr(fl[1]), None, None])  # 求解目标 辅助方程 目标值 前提

    def solve(self):
        for theorem in self.problem.theorem_seqs:  # 应用定理序列
            self.theorem_map[theorem](self.problem)

        self._solve_equations()  # 求解问题的方程组

        for i in range(self.problem.target_count):
            if self.problem.target_type[i] is tType.relation:  # 关系
                if self.problem.target[i][1] in self.problem.relations[self.problem.target[i][0]].items:
                    self.problem.target_solved[i] = "solved"
            elif self.problem.target_type[i] is tType.entity:  # 实体
                if self.problem.target[i][1] in self.problem.entities[self.problem.target[i][0]].items:
                    self.problem.target_solved[i] = "solved"
            else:  # 数量型
                self.problem.target[i][2], self.problem.target[i][3] = self._solve_targets(self.problem.target[i][0],
                                                                                           self.problem.target[i][1])
                if self.problem.target[i][2] is not None:
                    if self.problem.target_type[i] is tType.value:  # 数值型，有解
                        self.problem.target_solved[i] = "solved"
                    elif self.problem.target[i][2] == 0:  # 验证型，且解为0
                        self.problem.target_solved[i] = "solved"

    def _solve_equations(self):  # 求解方程并保存结果
        result = solve(self.problem.equations.items)  # 求解equation
        if len(result) == 0:  # 没有解，返回
            return
        if isinstance(result, list):  # 解不唯一，选择第一个
            result = result[0]
        for attr_var in result.keys():  # 遍历所有的解
            if isinstance(result[attr_var], Float):  # 如果解是实数，保存
                self.problem.value_of_sym[attr_var] = abs(float(result[attr_var]))

    def _solve_targets(self, target, target_equation):  # 求解目标方程，返回目标值
        self.problem.equations.items.append(target_equation)  # 将目标方程添加到方程组
        result = solve(self.problem.equations.items)  # 求解equation
        self.problem.equations.items.remove(target_equation)  # 求解后，移除目标方程

        if len(result) == 0:  # 没有解，返回None
            return None, None

        if isinstance(result, list):  # 解不唯一，选择第一个
            result = result[0]

        if target in result.keys() and isinstance(result[target], Float):
            return abs(float(result[target])), -2  # 有实数解，返回解

        return None, None  # 无实数解，返回None

    """------------auxiliary function------------"""
