from sympy import *
from problem import Problem
from main import load_data
from utility import RegularExpression as reE
from facts import AttributionType
from sympy import solve, Float, sin, cos, tan


def solve_test():
    ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
    f1 = ll_ad - 3  # 如果是浮点数，结果就是根号的形式
    f2 = ll_bd - 14
    f3 = ll_ab - ll_ad - ll_bd
    f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
    f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
    f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
    equation = [f1, f2, f3, f4, f5, f6]
    para = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]
    result = solve(equation, para)
    print(result)

    ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
    f1 = ll_ad - 3.0  # 如果是浮点数，结果就是实数的形式
    f2 = ll_bd - 14.0
    f3 = ll_ab - ll_ad - ll_bd
    f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
    f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
    f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
    equation = [f1, f2, f3, f4, f5, f6]
    para = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]
    result = solve(equation, para)
    print(result)

    ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
    value_of_sym = {ll_ad: 3.0, ll_bd: 14.0}  # 已知值的符号
    f1 = ll_ad - 3.0  # 如果是浮点数，结果就是根号的形式
    f2 = ll_bd - 14.0
    f3 = ll_ab - ll_ad - ll_bd
    f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
    f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
    f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
    equation = [f1, f2, f3, f4, f5, f6]
    paras = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]

    new_para = []
    for para in paras:
        if para in value_of_sym.keys():  # 替换成值
            for i in range(0, len(equation)):
                equation[i] = equation[i].subs(para, value_of_sym[para])
        else:  # 带求解的符号
            new_para.append(para)

    new_equation = []
    for i in range(0, len(equation)):
        if equation[i] != 0:
            new_equation.append(equation[i])

    result = solve(new_equation, new_para)
    print(result)


def test1():
    a, b, m = symbols("a b m")
    f1 = a**2 + b**2 - 6
    f2 = a**2 + b**2 - m
    result = solve([f1, f2], [a, b, m])
    print(result)


def test2():
    a, b, m = symbols("a b m")
    f1 = a + b - 6.0
    f2 = a + b - m
    for i in f2:
        print(i)


def test_symbol_result():
    a, b, t = symbols("a b t")
    f1 = a - b
    f2 = b - t
    result = solve([f1, f2])
    print(result)


def test_symbol_result2():
    a, b, c, t = symbols("a b c t")
    f1 = a - b
    f2 = b - c
    f3 = t - a + c
    result = solve([f1, f2, f3])
    print(result)


def test_symbol_result3():
    a, b, c, t = symbols("a b c t")
    f1 = a - 5
    f2 = c - 5
    f3 = t - a + c
    result = solve([f1, f2, f3])
    print(result)


def test_symbol_result4():
    a, b, c, t = symbols("a b c t")
    f1 = a - b
    f2 = t - a + b
    result = solve([f1, f2])
    print(result)


def test_symbol_result5():
    a, b, c = symbols("a b c")
    f1 = a + b
    f2 = f1 * c
    print(f2)    # c*(a + b)


def _parse_expr(problem, expr):    # 解析表达式为list形式，方便转化为sym体系下的表达式
    expr_list = reE.get_expr(expr + "~")
    expr_stack = []
    operator_stack = ["~"]    # 栈底元素

    i = 0
    while i < len(expr_list):
        unit = expr_list[i]
        if unit in reE.operator:    # 运算符
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
                elif operator_unit == "{":    # 只有unit为"}"，才能到达这个判断
                    i = i + 1
                elif operator_unit == "@":    # sin
                    expr_1 = expr_stack.pop()
                    expr_stack.append(sin(expr_1))
                elif operator_unit == "#":    # cos
                    expr_1 = expr_stack.pop()
                    expr_stack.append(cos(expr_1))
                elif operator_unit == "$":    # tan
                    expr_1 = expr_stack.pop()
                    expr_stack.append(tan(expr_1))
                elif operator_unit == "~":    # 只有unit为"~"，才能到达这个判断，表示表达式处理完成
                    break
        else:    # 实数或符号
            unit = problem.get_sym_of_attr((AttributionType.F.name, unit)) if unit.isalpha() else float(unit)
            expr_stack.append(unit)
            i = i + 1

    return expr_stack.pop()


problems_path = "./test_data/problem.json"
data = load_data(problems_path)[0]
pro = Problem(data["problem_index"], data["formal_languages"], data["theorem_seqs"])
expr_test = "36.1+b-61*5/2^{a+@{4.2}}"  # (用{代替
_parse_expr(pro, expr_test)
