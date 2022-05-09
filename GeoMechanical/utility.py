from pyparsing import oneOf, Combine, alphanums, Forward, Group, Word, Literal, ZeroOrMore, nums, alphas, OneOrMore


class Representation:

    count_line = 2
    count_angle = 2
    count_tri = 6    # 三角形
    count_rt_tri = 2    # 直角三角形
    count_iso_tri = 2    # 等腰三角形
    count_qua = 8
    count_parallel = 4
    count_perpendicular = 4
    count_congruent = count_tri    # 全等
    count_similar = count_tri    # 相似

    @staticmethod
    def line(entity):
        return [entity, entity[::-1]]

    @staticmethod
    def angle(entity):
        return [entity, entity[::-1]]

    @staticmethod
    def shape(entity):
        entity_inverse = entity[::-1]
        results = []
        length = len(entity)
        for i in range(length):
            result = ""
            result_inverse = ""
            for j in range(length):
                result += entity[(i + j) % length]
                result_inverse += entity_inverse[(i + j) % length]
            results.append(result)
            results.append(result_inverse)
        return results

    @staticmethod
    def parallel(relation):
        line1, line2 = relation
        results = [(line1, line2),
                   (line1[::-1], line2[::-1]),
                   (line2, line1),
                   (line2[::-1], line1[::-1])]
        return results

    @staticmethod
    def perpendicular(relation):
        line1, line2 = relation
        results = [(line1, line2),
                   (line2, line1[::-1]),
                   (line1[::-1], line2[::-1]),
                   (line2[::-1], line1)]
        return results


class PreParse:
    # 处理形式化语言
    entity = ['Point', 'Line', 'Angle', 'Arc', 'Shape', 'Circle', 'Sector', 'Triangle', 'RightTriangle',
              'IsoscelesTriangle', 'RegularTriangle', 'Quadrilateral', 'Trapezoid', 'IsoscelesTrapezoid',
              'Parallelogram', 'Rectangle', 'Rhombus', 'Kite', 'Square', 'Polygon', 'RegularPolygon']
    binary_relation = ['PointOnLine', 'PointOnArc', 'PointOnCircle', 'Midpoint', 'Circumcenter', 'Incenter', 'Centroid',
                       'Orthocenter', 'Parallel', 'BisectsAngle', 'DisjointLineCircle', 'DisjointCircleCircle',
                       'Median', 'HeightOfTriangle', 'HeightOfTrapezoid', 'Contain', 'CircumscribedToTriangle',
                       'Congruent', 'Similar', 'Chord']
    ternary_relation = ['Perpendicular', 'PerpendicularBisector', 'TangentLineCircle', 'TangentCircleCircle',
                        'IntersectLineLine', 'InternallyTangent']
    quaternion_relation = ["IntersectLineCircle", "IntersectCircleCircle"]
    five_relation = ["InscribedInTriangle"]

    idt_fl = Forward()  # 定义解析 formal language 的 表达式
    expr = Word(alphanums + '.+-*/^{}@#$~')  # 识别的最小unit 谓词或字母或数字或代数表达式
    arg = Group(idt_fl) | expr  # arg 可能是聚合体，也可能是标志符
    args = arg + ZeroOrMore(Literal(",").suppress() + arg)  # arg组合起来  suppress 的字符不会在结果中出现
    idt_fl <<= expr + Literal("(").suppress() + args + Literal(")").suppress()

    # 处理表达式的
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
    def parse_expr(expr):
        return PreParse.idt_expr.parseString(expr).asList()

    @staticmethod
    def pre_parse_fl(fl):
        result = [fl[0]]
        if fl[0] == "Equal":    # equal，不用处理
            result = fl
        elif fl[0] == "Find":    # find，需要处理其子内容
            result.append(PreParse.pre_parse_fl(fl[1]))
        elif fl[0] in PreParse.entity:    # 一元关系，不需要处理
            result = fl
        elif fl[0] in PreParse.binary_relation:    # 二元关系
            result.append(tuple([fl[1][1], fl[2][1]]))
        elif fl[0] in PreParse.ternary_relation:    # 三元关系
            result.append(tuple([fl[1][1], fl[2][1], fl[3][1]]))
        elif fl[0] in PreParse.quaternion_relation:    # 四元关系
            result.append(tuple([fl[1][1], fl[2][1], fl[3][1], fl[4][1]]))
        elif fl[0] in PreParse.five_relation:    # 五元关系
            result.append(tuple([fl[1][1], fl[2][1], fl[3][1], fl[4][1], fl[5][1]]))
        elif fl[0] == "PointOn":    # 一些需要extend的语句
            if fl[2][0] == "Line":
                result[0] = "PointOnLine"
            elif fl[2][0] == "Arc":
                result[0] = "PointOnArc"
            else:
                result[0] = "PointOnCircle"
            result.append(tuple([fl[1][1], fl[2][1]]))
        elif fl[0] == "Disjoint":
            if fl[1][0] == "Line":
                result[0] = "DisjointLineCircle"
            else:
                result[0] = "DisjointCircleCircle"
            result.append(tuple([fl[1][1], fl[2][1]]))
        elif fl[0] == "Tangent":
            if fl[2][0] == "Line":
                result[0] = "TangentLineCircle"
            else:
                result[0] = "TangentCircleCircle"
            result.append(tuple([fl[1][1], fl[2][1], fl[3][1]]))
        elif fl[0] == "Intersect":
            if fl[2][0] == "Line":
                result[0] = "IntersectLineLine"
                result.append(tuple([fl[1][1], fl[2][1], fl[3][1], fl[4][1]]))
            elif fl[3][0] == "Line":
                result[0] = "IntersectLineCircle"
                result.append(tuple([fl[1][1], fl[2][1], fl[3][1], fl[4][1], fl[5][1]]))
            else:
                result[0] = "IntersectCircleCircle"
                result.append(tuple([fl[1][1], fl[2][1], fl[3][1], fl[4][1], fl[5][1]]))
        elif fl[0] == "Height":
            if fl[2][0] == "Triangle":
                result[0] = "HeightOfTriangle"
            else:
                result[0] = "HeightOfTrapezoid"
            result.append(tuple([fl[1][1], fl[2][1]]))
        else:    # 属性类语句，只可能出现在find的子语句了
            return ["Value", fl]

        return result

    @staticmethod
    def pre_parse_fls(fls):
        for i in range(len(fls)):
            fls[i] = PreParse.idt_fl.parseString(fls[i]).asList()
            fls[i] = PreParse.pre_parse_fl(fls[i])
        return fls
