from sympy import *
import math
import json
test_define_file = "./test_data/test_define.json"
test_problem_file = "./test_data/problem.json"


# def solve_test():
#     ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
#     f1 = ll_ad - 3  # 如果是浮点数，结果就是根号的形式
#     f2 = ll_bd - 14
#     f3 = ll_ab - ll_ad - ll_bd
#     f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
#     f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
#     f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
#     equation = [f1, f2, f3, f4, f5, f6]
#     para = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]
#     result = solve(equation, para)
#     print(result)
#
#     ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
#     f1 = ll_ad - 3.0  # 如果是浮点数，结果就是实数的形式
#     f2 = ll_bd - 14.0
#     f3 = ll_ab - ll_ad - ll_bd
#     f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
#     f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
#     f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
#     equation = [f1, f2, f3, f4, f5, f6]
#     para = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]
#     result = solve(equation, para)
#     print(result)
#
#     ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd = symbols("ll_ad ll_bd ll_ab ll_bc ll_ca ll_cd")
#     value_of_sym = {ll_ad: 3.0, ll_bd: 14.0}  # 已知值的符号
#     f1 = ll_ad - 3.0  # 如果是浮点数，结果就是根号的形式
#     f2 = ll_bd - 14.0
#     f3 = ll_ab - ll_ad - ll_bd
#     f4 = -ll_ab ** 2 + ll_bc ** 2 + ll_ca ** 2
#     f5 = ll_ad ** 2 - ll_ca ** 2 + ll_cd ** 2
#     f6 = -ll_bc ** 2 + ll_bd ** 2 + ll_cd ** 2
#     equation = [f1, f2, f3, f4, f5, f6]
#     paras = [ll_ad, ll_bd, ll_ab, ll_bc, ll_ca, ll_cd]
#
#     new_para = []
#     for para in paras:
#         if para in value_of_sym.keys():  # 替换成值
#             for i in range(0, len(equation)):
#                 equation[i] = equation[i].subs(para, value_of_sym[para])
#         else:  # 带求解的符号
#             new_para.append(para)
#
#     new_equation = []
#     for i in range(0, len(equation)):
#         if equation[i] != 0:
#             new_equation.append(equation[i])
#
#     result = solve(new_equation, new_para)
#     print(result)
#
#
# def test1():
#     a, b, m = symbols("a b m")
#     f1 = a**2 + b**2 - 6
#     f2 = a**2 + b**2 - m
#     result = solve([f1, f2], [a, b, m])
#     print(result)
#
#
# def test2():
#     a, b, m = symbols("a b m")
#     f1 = a + b - 6.0
#     f2 = a + b - m
#     for i in f2:
#         print(i)
#
#
# def test_symbol_result():
#     a, b, t = symbols("a b t")
#     f1 = a - b
#     f2 = b - t
#     result = solve([f1, f2])
#     print(result)
#
#
# def test_symbol_result2():
#     a, b, c, t = symbols("a b c t")
#     f1 = a - b
#     f2 = b - c
#     f3 = t - a + c
#     result = solve([f1, f2, f3])
#     print(result)
#
#
# def test_symbol_result3():
#     a, b, c, t = symbols("a b c t")
#     f1 = a - 5
#     f2 = c - 5
#     f3 = t - a + c
#     result = solve([f1, f2, f3])
#     print(result)
#
#
# def test_symbol_result4():
#     a, b, c, t = symbols("a b c t")
#     f1 = a - b
#     f2 = t - a + b
#     result = solve([f1, f2])
#     print(result)
#
#
# def test_symbol_result5():
#     a, b, c = symbols("a b c")
#     f1 = a + b
#     f2 = f1 * c
#     print(f2)    # c*(a + b)
#
#
# def test_list(x):    # 解析表达式为list形式，方便转化为sym体系下的表达式
#     a, b, c = x
#     print(a)
#     print(b)
#     print(c)
#
#
# def load_data(problems_path):  # 读取json数据并解析成列表
#     problem_data = json.load(open(problems_path, "r", encoding="utf-8"))  # 文本 Formal Language
#     return list(problem_data.values())
#
#
# def test_define():
#     data = load_data(test_define_file)
#     for i in range(0, len(data)):
#         try:
#             solver = Solver(data[i]["problem_index"], data[i]["formal_languages"], data[i]["theorem_seqs"])
#             solver.problem.show_problem()
#         except Exception as e:
#             print(e)
#         print()
#         if (i + 1) % 20 == 0:
#             a = input("输入1继续执行：")
#
#
# def test_problem():
#     data = load_data(test_problem_file)[1]
#     solver = Solver(data["problem_index"], data["formal_languages"], data["theorem_seqs"])
#     solver.solve()
#     solver.problem.show_problem()

# a, b = symbols("a b")
#
# f1 = a - 1.0
# f2 = a / sin(b) - 2.0
#
# result = solve([f1, f2])
# print(result)


def test1():
    print(180 * -1.12788528272126 / math.pi)
    print(180 * 1.12788528272126 / math.pi)
    print(180 * 2.01370737086854 / math.pi)
    print(180 * 4.26947793631105 / math.pi)

    print(4.26947793631105 - math.pi)

    a = symbols("a", positive=True)
    b = symbols("b", positive=True)
    A = symbols("A", positive=True)
    c = symbols("c", positive=True)
    B = symbols("B", positive=True)
    C = symbols("C", positive=True)
    t_1 = symbols("t_1")

    f1 = a - 9.89949493661167
    f2 = b - 4.24264068711929
    f3 = A - pi / 2
    f4 = c ** 2 - a ** 2 + b ** 2
    f5 = a / sin(A) - b / sin(B)
    f6 = -c / sin(C) + b / sin(B)
    f7 = -180 * C / pi + t_1

    for result in solve([f7, f6, f5, f4, f3, f2, f1]):
        print(result)

    print()

    for result in solve([f1, f2, f3, f4, f5, f6, f7]):
        print(result)


def test2():
    ll_rt = symbols("ll_rt", positive=True)
    ll_ts = symbols("ll_ts", positive=True)
    da_rst = symbols("da_rst", positive=True)
    ll_rs = symbols("ll_rs", positive=True)
    da_srt = symbols("da_srt", positive=True)
    da_rts = symbols("da_rts", positive=True)
    t_1 = symbols("t_1")

    f1 = ll_rt - 9.89949493661167
    f2 = ll_ts - 4.24264068711929
    f3 = da_rst - pi / 2
    f4 = ll_rs ** 2 - ll_rt ** 2 + ll_ts ** 2
    f5 = ll_rt / sin(da_rst) - ll_ts / sin(da_srt)
    f6 = -ll_rs / sin(da_rts) + ll_ts / sin(da_srt)
    f7 = -180 * da_rts / pi + t_1

    for result in solve([f7, f6, f5, f4, f3, f2, f1]):
        print(result)

    print()

    for result in solve([f1, f2, f3, f4, f5, f6, f7]):
        print(result)


def test3():
    # 怎么起名字，还得斟酌一下，tm的...
    ll_rt = symbols("ll_rt", positive=True)    # 换成a就可以求解，为什么？？
    ll_ts = symbols("ll_ts", positive=True)
    da_rst = symbols("da_rst", positive=True)
    ll_rs = symbols("ll_rs", positive=True)
    da_srt = symbols("da_srt", positive=True)
    da_rts = symbols("da_rts", positive=True)
    t_1 = symbols("t_1")

    f1 = ll_rt - 9.89949493661167
    f2 = ll_ts - 4.24264068711929
    f3 = da_rst - pi / 2
    f4 = ll_rs ** 2 - ll_rt ** 2 + ll_ts ** 2
    f5 = ll_rt / sin(da_rst) - ll_ts / sin(da_srt)
    f6 = -ll_rs / sin(da_rts) + ll_ts / sin(da_srt)
    f7 = -180 * da_rts / pi + t_1

    for result in solve([f7, f6, f5, f4, f3, f2, f1]):
        print(result)

    for i in [f7, f6, f5, f4, f3, f2, f1]:
        print(i)


# test3()

# a = symbols("x")
# b = symbols("x")
# print(a - b)
