from sympy import *


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


test2()
