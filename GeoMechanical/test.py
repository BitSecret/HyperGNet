from sympy import *


a, b, c = symbols("a b c")
f1 = a**2 + b**2 - c**2
f2 = a - 3
f3 = b - 4
r3 = solve([f1, f2, f3], [a, b, c])    # 线性
print("r3:", r3)
