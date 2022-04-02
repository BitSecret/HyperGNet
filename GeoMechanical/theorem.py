from facts import AttributionType


class Theorem:
    """------------theorem------------"""

    @staticmethod
    def theorem_1_pythagorean(problem):    # 勾股定理
        update = False    # 存储应用定理是否更新了条件
        for rt_triangle in problem.right_triangle.items:
            a = problem.get_sym_of_attr((AttributionType.LL, rt_triangle[0:2]))
            b = problem.get_sym_of_attr((AttributionType.LL, rt_triangle[1:3]))
            c = problem.get_sym_of_attr((AttributionType.LL, rt_triangle[0] + rt_triangle[2]))
            update = problem.define_equation(a**2 + b**2 - c**2, -3, 1) or update
        return update

    @staticmethod
    def theorem_2_xxxx(problem):
        print("problem {} applied with theorem_2_xxxx".format(problem.problem_index))

    @staticmethod
    def theorem_3_xxxx(problem):
        print("problem {} applied with theorem_3_xxxx".format(problem.problem_index))

    @staticmethod
    def theorem_4_xxxx(problem):
        print("problem {} applied with theorem_4_xxxx".format(problem.problem_index))
