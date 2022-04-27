from facts import AttributionType
from utility import get_all_representation_of_shape
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
    def theorem_2_pythagorean_inverse(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT三角形
        """
        update = False    # 存储应用定理是否更新了条件
        for tri in problem.triangle.items:
            a = problem.get_sym_of_attr((AttributionType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((AttributionType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((AttributionType.LL.name, tri[0] + tri[2]))
            t_1 = problem.get_sym_of_attr((AttributionType.T.name, "1"))

            result, premise = Theorem._solve_targets(problem, t_1, t_1 - a**2 - b**2 + c**2)
            if result is not None and result == 0:
                update = problem.define_right_triangle(tri, premise, 2) or update
        return update

    @staticmethod
    def theorem_3_transitivity_of_parallel(problem):
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
    def theorem_4_similar_triangle(problem):
        """
        相似三角形 性质
        两个三角形相似  ==>  对应角相等、对应边成比例
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_5_similar_triangle_inverse(problem):
        """
        相似三角形 定理
        对应角相等、对应边成比例  ==>  两个三角形相似
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_6_congruent_triangle(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应角相等、对应边相等
        """
        update = False  # 存储应用定理是否更新了条件

        for tri_congruent in problem.congruent.items:
            index = problem.congruent.indexes[tri_congruent]    # 前提
            tri1 = tri_congruent[0]    # 三角形
            tri2 = tri_congruent[1]
            tri1_angle = get_all_representation_of_shape(tri1)    # 三角形的角
            tri2_angle = get_all_representation_of_shape(tri2)
            for i in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr((AttributionType.LL.name, tri1[i] + tri1[(i + 1) % 3]))
                l_2 = problem.get_sym_of_attr((AttributionType.LL.name, tri2[i] + tri2[(i + 1) % 3]))
                update = problem.define_equation(l_1 - l_2, index, 6) or update
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((AttributionType.DA.name, tri1_angle[i]))
                angle_2 = problem.get_sym_of_attr((AttributionType.DA.name, tri2_angle[i]))
                update = problem.define_equation(angle_1 - angle_2, index, 6) or update
        return update

    @staticmethod
    def theorem_7_congruent_triangle_inverse(problem):
        """
        全等三角形 定理
        SSS、SAS、ASA、AL、HL  ==>  两个三角形全等
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def _solve_targets(problem, target, target_equation):   # 求解目标值，并返回使用的最小方程组集合(前提)
        problem.equations.items.append(target_equation)  # 将目标方程添加到方程组
        result = solve(problem.equations.items)  # 求解equation
        problem.equations.items.remove(target_equation)  # 求解后，移除目标方程

        if len(result) == 0:  # 没有解，返回None
            return None, None

        if isinstance(result, list):  # 解不唯一，选择第一个
            result = result[0]

        if target in result.keys() and isinstance(result[target], Float):
            return abs(float(result[target])), -2  # 有实数解，返回解

        return None, None
