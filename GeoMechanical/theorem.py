from facts import AttributionType
from sympy import solve, Float


class Theorem:
    """------------theorem------------"""

    @staticmethod
    def theorem_1_pythagorean(problem):
        """
        勾股定理
        RT三角形  ==>  a**2 + b**2 - c**2 = 0
        """
        update = False    # 存储应用定理是否更新了条件
        for rt in problem.right_triangle.items:
            a = problem.get_sym_of_attr((AttributionType.LL.name, rt[0:2]))
            b = problem.get_sym_of_attr((AttributionType.LL.name, rt[1:3]))
            c = problem.get_sym_of_attr((AttributionType.LL.name, rt[0] + rt[2]))
            update = problem.define_equation(a**2 + b**2 - c**2, problem.right_triangle.indexes[rt], 1) or update
        return update

    @staticmethod
    def theorem_2_transitivity_of_parallel(problem):
        """
        平行的传递性
        AB // CD, CD // EF  ==>  AB // EF
        """
        update = False  # 存储应用定理是否更新了条件
        for item1 in problem.parallel.items:
            for item2 in problem.parallel.items:
                if item1[1] == item2[0]:
                    premise = [problem.parallel.indexes[item1], problem.parallel.indexes[item2]]
                    update = problem.define_parallel(item1[0], item2[1], premise, 2) or update
        return update

    @staticmethod
    def theorem_3_xxxx(problem):
        print("problem {} applied with theorem_3_xxxx".format(problem.problem_index))

    @staticmethod
    def theorem_4_xxxx(problem):
        print("problem {} applied with theorem_4_xxxx".format(problem.problem_index))

    @staticmethod
    def _solve_equations(problem):
        result = solve(problem.equations_unsolved)  # 求解equation
        if len(result) == 0:  # 没有解，返回
            return
        if isinstance(result, list):  # 解不唯一，选择第一个
            result = result[0]
        for attr_var in result.keys():  # 遍历所有的解
            if isinstance(result[attr_var], Float):  # 如果解是实数，保存
                problem.value_of_sym[attr_var] = abs(float(result[attr_var]))
