from pyparsing import oneOf, Combine, alphanums, Forward, Group, Word, Literal, ZeroOrMore, nums, alphas, OneOrMore
from facts import ConditionType as cType


class Representation:
    """多种表示的数量"""
    # count_shape
    count_collinear = 2

    count_point = 1  # 点、线、角、弧
    count_line = 2
    count_angle = 1
    count_triangle = 3  # 三角形
    count_right_triangle = 1
    count_isosceles_triangle = 1
    count_equilateral_triangle = 3
    # count_polygon

    count_midpoint = count_line
    count_intersect = 4
    count_parallel = 2
    count_disorder_parallel = 2
    count_perpendicular = count_intersect
    count_perpendicular_bisector = 2
    count_bisector = 1
    count_median = 1
    count_is_altitude = 1
    count_neutrality = 1
    count_circumcenter = 3
    count_incenter = 3
    count_centroid = 3
    count_orthocenter = 3
    count_congruent = 6
    count_similar = 6
    count_mirror_congruent = 6
    count_mirror_similar = 6

    # count_equation

    """one to n"""
    @staticmethod
    def shape(entity):
        results = []
        length = len(entity)
        for i in range(length):
            result = ""
            for j in range(length):
                result += entity[(i + j) % length]
            results.append(result)
        return results

    @staticmethod
    def parallel(relation):
        line1, line2 = relation
        results = [(line1, line2),
                   (line2[::-1], line1[::-1])]
        return results

    @staticmethod
    def disorder_parallel(relation):
        line1, line2 = relation
        results = [(line1, line2),
                   (line2, line1),
                   (line2[::-1], line1[::-1]),
                   (line1[::-1], line2[::-1])]
        return results

    @staticmethod
    def intersect(relation):
        point, line1, line2 = relation
        results = [(point, line1, line2),
                   (point, line2, line1[::-1]),
                   (point, line1[::-1], line2[::-1]),
                   (point, line2[::-1], line1)]
        return results

    @staticmethod
    def perpendicular_bisector(relation):
        point, line1, line2 = relation
        results = [(point, line1, line2),
                   (point, line1[::-1], line2[::-1])]
        return results

    @staticmethod
    def mirror_tri(relation):
        tri1, tri2 = relation
        results = []
        for i in range(3):
            s1 = ""
            s2 = ""
            for j in range(3):
                s1 += tri1[(i + j) % 3]
                s2 += tri2[(j - i + 3) % 3]
            results.append((s1, s2))
            results.append((s2, s1))
        return results


class PreParse:
    # 解析formal language的idt: idt_fl
    idt_fl = Forward()
    expr = Word(alphanums + '.+-*/^{}@#$~')  # 识别的最小unit 谓词或字母或数字或代数表达式
    arg = Group(idt_fl) | expr  # arg 可能是聚合体，也可能是标志符
    args = arg + ZeroOrMore(Literal(",").suppress() + arg)  # arg组合起来  suppress 的字符不会在结果中出现
    idt_fl <<= expr + Literal("(").suppress() + args + Literal(")").suppress()

    # 处理表达式的idt: idt_expr
    float_idt = Combine(OneOrMore(Word(nums)) + "." + OneOrMore(Word(nums)))  # 浮点数
    int_idt = OneOrMore(Word(nums))  # 整数
    alpha_idt = Word(alphas)  # 符号
    operator_idt = oneOf("+ - * / ^ { } @ # $ ~")  # 运算符
    idt_expr = OneOrMore(float_idt | int_idt | alpha_idt | operator_idt)

    operator = ["+", "-", "*", "/", "^", "{", "}", "@", "#", "$", "~"]
    stack_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                      "{": 0, "}": None,
                      "@": 4, "#": 4, "$": 4, "~": 0}
    outside_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                        "{": 5, "}": 0,
                        "@": 4, "#": 4, "$": 4, "~": 0}

    @staticmethod
    def parse_expr(expr):    # trans expr to sym_list
        return PreParse.idt_expr.parseString(expr).asList()

    @staticmethod
    def pre_parse_fls(fls):    # trans FL to parse_tree
        fls_parse_tree = []
        for fl in fls:
            fls_parse_tree.append(PreParse.idt_fl.parseString(fl).asList())
        return fls_parse_tree


class Utility:
    @staticmethod
    def equal(a, b):    # 判断ab是否相等，考虑到浮点数的精度问题
        return abs(a - b) < 0.01

    @staticmethod
    def is_compact_collinear(point1, point2, point3, problem):    # 紧凑共线，3点之间没有别的点
        for collinear in problem.conditions.items[cType.collinear]:
            if point1 in collinear and\
                    point2 in collinear and\
                    point3 in collinear and \
                    collinear.index(point1) < collinear.index(point2) < collinear.index(point3) and \
                    (collinear.index(point1) + collinear.index(point3)) == 2 * collinear.index(point2):
                return True
        return False

    @staticmethod
    def is_collinear(point1, point2, point3, problem):  # 简单共线，3点之间可有别的点
        for collinear in problem.conditions.items[cType.collinear]:
            if point1 in collinear and \
                    point2 in collinear and \
                    point3 in collinear and \
                    collinear.index(point1) < collinear.index(point2) < collinear.index(point3):
                return True
        return False

    @staticmethod
    def is_inside_triangle(line, triangle, problem):    # 判断line是不是triangle的内线
        if line[0] == triangle[0] and \
                Utility.is_collinear(triangle[1], line[1], triangle[2], problem):
            return True
        return False

    @staticmethod
    def coll_points(point1, point2, problem):  # 与 point1 point2 共线 的所有点
        for coll in problem.conditions.items[cType.collinear]:
            if point1 in coll and point2 in coll and coll.index(point1) < coll.index(point2):
                return list(coll)
        return [point1, point2]

    @staticmethod
    def coll_points_one_side(vertex, point, problem):  # 与 vertex、point共线且与 point 同侧的点
        points = []
        for coll in problem.conditions.items[cType.collinear]:
            if vertex in coll and point in coll:
                if coll.index(vertex) < coll.index(point):    # .....V...P..
                    i = coll.index(vertex) + 1
                    while i < len(coll):
                        points.append(coll[i])
                        i += 1
                else:    # ...P.....V...
                    i = 0
                    while i < coll.index(vertex):
                        points.append(coll[i])
                        i += 1
                return points
        return [point]

    @staticmethod
    def same_angles(angle, problem):
        angles = []

        a_points = Utility.coll_points_one_side(angle[1], angle[0], problem)  # 与AO共线、在O的A侧的点(包括A)
        b_points = Utility.coll_points_one_side(angle[1], angle[2], problem)  # 与BO共线、在O的B侧的点(包括B)

        if len(a_points) == 1 and len(b_points) == 1:    # 角只有一种表示
            return [angle]

        for a_point in a_points:
            for b_point in b_points:
                angles.append(a_point + angle[1] + b_point)  # 相同的角设置一样的符号
        return angles

    @staticmethod
    def same_intersects(intersect, problem):
        point, line1, line2 = intersect
        intersects = []

        left_points = Utility.coll_points_one_side(point, line1[0], problem)
        up_points = Utility.coll_points_one_side(point, line2[0], problem)
        right_points = Utility.coll_points(point, line1[0], problem)
        down_points = Utility.coll_points(point, line1[0], problem)
        for point in left_points + [point]:
            if point in right_points:
                right_points.remove(point)
        for point in up_points + [point]:
            if point in down_points:
                down_points.remove(point)
        for left_point in left_points:
            for right_point in right_points:
                for up_point in up_points:
                    for down_point in down_points:
                        intersects.append((point, left_point + right_point, up_point + down_point))

        if intersect not in intersects:    # 确保第1个是原关系
            return [intersect]
        else:
            intersects.remove(intersect)
            return [intersect] + intersects

    @staticmethod
    def same_perpendiculars(perpendicular, problem):
        point, line1, line2 = perpendicular
        perpendiculars = []

        left_points = Utility.coll_points_one_side(point, line1[0], problem)
        up_points = Utility.coll_points_one_side(point, line2[0], problem)
        right_points = Utility.coll_points(point, line1[0], problem)
        down_points = Utility.coll_points(point, line2[0], problem)
        for point in left_points + [point]:
            if point in right_points:
                right_points.remove(point)
        for point in up_points + [point]:
            if point in down_points:
                down_points.remove(point)
        left_points.append(point)
        up_points.append(point)
        right_points.append(point)
        down_points.append(point)

        for left_point in left_points:
            for right_point in right_points:
                if left_point != right_point:
                    for up_point in up_points:
                        for down_point in down_points:
                            if up_point != down_point:
                                perpendiculars.append((point, left_point + right_point, up_point + down_point))

        if perpendicular not in perpendiculars:    # 确保第1个是原关系
            return [perpendicular]
        else:
            perpendiculars.remove(perpendicular)
            return [perpendicular] + perpendiculars

    @staticmethod
    def all_lines_in_coll(coll, problem):
        lines = []
        for point1 in coll:
            for point2 in coll:
                lines.append(point1 + point2)
        return lines
