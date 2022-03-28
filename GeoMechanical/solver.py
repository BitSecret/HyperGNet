from pyparsing import alphanums, Forward, Group, Word, Literal, ZeroOrMore
from problem import Problem
from theorem import Theorem


class Solver:

    def __init__(self, problem_index, formal_languages, theorem_seqs=None):
        self.problem = Problem(problem_index, formal_languages, theorem_seqs)     # 题目
        self.parse()    # 解析形式化语句到logic形式

    def parse(self):
        py_expression = Forward()    # 定义解析 formal language 的 表达式
        identifier = Word(alphanums + ' +-*/=.\\{}^_$\'')  # 识别的最小unit 谓词或字母或数字或代数表达式
        arg = Group(py_expression) | identifier  # arg 可能是聚合体，也可能是标志符
        args = arg + ZeroOrMore(Literal(",").suppress() + arg)  # arg组合起来  suppress 的字符不会在结果中出现
        py_expression <<= identifier + Literal("(").suppress() + args + Literal(")").suppress()

        py_results = []    # 储存解析结果
        for formal_language in self.problem.formal_languages:    # 解析 formal language
            py_results.append(py_expression.parseString(formal_language).asList())

        for py_result in py_results:    # 将 formal language 转化为 logic
            if py_result[0] == "Point":
                self.problem.define_point(py_result[1])
                continue
            elif py_result[0] == "Line":
                self.problem.define_line(py_result[1])
                continue
            elif py_result[0] == "Angle":
                self.problem.define_angle(py_result[1])
                continue
            elif py_result[0] == "Arc":
                self.problem.define_arc(py_result[1])
                continue
            elif py_result[0] == "Shape":
                self.problem.define_shape(py_result[1])
                continue
            elif py_result[0] == "Circle":
                self.problem.define_circle(py_result[1])
                continue
            elif py_result[0] == "Sector":
                self.problem.define_sector(py_result[1])
                continue
            elif py_result[0] == "Triangle":
                self.problem.define_triangle(py_result[1])
                continue
            elif py_result[0] == "RightTriangle":
                self.problem.define_right_triangle(py_result[1])
                continue
            elif py_result[0] == "IsoscelesTriangle":
                self.problem.define_isosceles_trapezoid(py_result[1])
                continue
            elif py_result[0] == "RegularTriangle":
                self.problem.define_regular_triangle(py_result[1])
                continue
            elif py_result[0] == "Quadrilateral":
                self.problem.define_quadrilateral(py_result[1])
                continue
            elif py_result[0] == "Trapezoid":
                self.problem.define_trapezoid(py_result[1])
                continue
            elif py_result[0] == "IsoscelesTrapezoid":
                self.problem.define_isosceles_trapezoid(py_result[1])
                continue
            elif py_result[0] == "Parallelogram":
                self.problem.define_parallelogram(py_result[1])
                continue
            elif py_result[0] == "Rectangle":
                self.problem.define_rectangle(py_result[1])
                continue
            elif py_result[0] == "Rhombus":
                self.problem.define_rhombus(py_result[1])
                continue
            elif py_result[0] == "Kite":
                self.problem.define_kite(py_result[1])
                continue
            elif py_result[0] == "Square":
                self.problem.define_square(py_result[1])
                continue
            elif py_result[0] == "Polygon":
                self.problem.define_polygon(py_result[1])
                continue
            elif py_result[0] == "RegularPolygon":
                self.problem.define_regular_polygon(py_result[1])
                continue

    def solve(self):
        Theorem.theorem_1_xxxx(self.problem)  # 应用定理
        self.problem.show_problem()    # 展示结果

    """------------auxiliary function------------"""
    def show_result(self):    # 输出求解结果
        pass
