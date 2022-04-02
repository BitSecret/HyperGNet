from sympy import *

a, b, c, d, e = symbols("a b c d e")
f1 = a + b - c
f11 = a**2 + b**2 - c**2
f2 = a - 3
f3 = b - 4
f4 = d + e
equation = [f11, f2, f3, f4]
para = [a, b, c, d, e]
r3 = solve(equation, para)    # 线性

print(r3)
print()

if isinstance(r3, dict):
    for i in [a, b, c, d]:
        if i in r3.keys():
            print(float(r3[i]))
else:
    for i in range(len(para)):
        if isinstance(r3[0][i], Integer):
            print(para[i], end=": ")
            print(float(r3[0][i]))


