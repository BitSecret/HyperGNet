from pyparsing import oneOf, Combine, alphanums, Forward, Group, Word, Literal, ZeroOrMore, nums, alphas, OneOrMore


def get_all_representation_of_shape(shape):
    results = []
    length = len(shape)
    for i in range(length):
        result = ""
        for j in range(length):
            result += shape[(i + j) % length]
        results.append(result)
    return results


def pre_parse(fl):
    if fl[0] == "PointOn":
        if fl[2][0] == "Line":
            fl[0] = "PointOnLine"
        elif fl[2][0] == "Arc":
            fl[0] = "PointOnArc"
        else:
            fl[0] = "PointOnCircle"
    elif fl[0] == "Disjoint":
        if fl[1][0] == "Line":
            fl[0] = "DisjointLineCircle"
        else:
            fl[0] = "DisjointCircleCircle"
    elif fl[0] == "Tangent":
        if fl[2][0] == "Line":
            fl[0] = "TangentLineCircle"
        else:
            fl[0] = "TangentCircleCircle"
    elif fl[0] == "Intersect":
        if fl[2][0] == "Line":
            fl[0] = "IntersectLineLine"
        elif fl[3][0] == "Line":
            fl[0] = "IntersectLineCircle"
        else:
            fl[0] = "IntersectCircleCircle"
    elif fl[0] == "Height":
        if fl[2][0] == "Triangle":
            fl[0] = "HeightOfTriangle"
        else:
            fl[0] = "HeightOfTrapezoid"

    return fl


class RegularExpression:
    # 处理形式化语言
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
    stack_priority = {}
    outside_priority = {}

    @staticmethod
    def get_expr(expr):
        return RegularExpression.idt_expr.parseString(expr).asList()

    @staticmethod
    def get_fl(fl):
        return RegularExpression.idt_fl.parseString(fl).asList()
