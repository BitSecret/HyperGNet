import time
from facts import AttributionType as aType
from sympy import solve, Float, pi


class Theorem:
    """------------theorem------------"""
    # 凡是带有 inverse 和 determine 的，在定理应用前都需要扩充条件的操作
    # hardcore版本，解题的可能性大一点，但是时间黑洞，轻易不要使用

    @staticmethod
    def theorem_1_pythagorean(problem):
        """
        勾股定理
        RT三角形  ==>  a**2 + b**2 - c**2 = 0
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.right_triangle.items):
            rt = problem.right_triangle.items[i]
            a = problem.get_sym_of_attr((aType.LL.name, rt[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, rt[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, rt[0] + rt[2]))
            update = problem.define_equation(a ** 2 + b ** 2 - c ** 2,
                                             [problem.right_triangle.indexes[rt]], 1) or update
            i = i + 2    # 一个RT△有2种表示

        return update

    @staticmethod
    def theorem_2_pythagorean_inverse(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT三角形
        """
        problem.find_all_triangle()    # 条件扩充 找到所有的隐藏三角形
        update = False  # 存储应用定理是否更新了条件
        problem.solve_equations()  # 求解方程
        i = 0
        while i < len(problem.triangle.items):
            tri = problem.triangle.items[i]
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))
            a = problem.value_of_sym[a]
            b = problem.value_of_sym[b]
            c = problem.value_of_sym[c]
            if a is not None and b is not None and c is not None:
                if (a ** 2 + b ** 2 - c ** 2) == 0:
                    update = problem.define_right_triangle(tri, [-3], 2) or update
                elif (a ** 2 - b ** 2 + c ** 2) == 0:
                    update = problem.define_right_triangle(tri[1] + tri[0] + tri[2], [-3], 2) or update
                elif (- a ** 2 + b ** 2 + c ** 2) == 0:
                    update = problem.define_right_triangle(tri[1] + tri[2] + tri[0], [-3], 2) or update
            i = i + 6  # 一个三角形有6种表示

        return update

    @staticmethod
    def theorem_2_pythagorean_inverse_hardcore(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT三角形
        """
        problem.find_all_triangle()    # 条件扩充 找到所有的隐藏三角形
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.triangle.items):
            tri = problem.triangle.items[i]
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))
            t_1 = problem.get_sym_of_attr((aType.T.name, "1"))

            result, premise = Theorem._solve_targets(problem, t_1, t_1 - a ** 2 - b ** 2 + c ** 2)
            if result is not None and result == 0:
                update = problem.define_right_triangle(tri, premise, 2) or update

            result, premise = Theorem._solve_targets(problem, t_1, t_1 - a ** 2 + b ** 2 - c ** 2)
            if result is not None and result == 0:
                update = problem.define_right_triangle(tri[1] + tri[0] + tri[2], premise, 2) or update

            result, premise = Theorem._solve_targets(problem, t_1, t_1 + a ** 2 - b ** 2 - c ** 2)
            if result is not None and result == 0:
                update = problem.define_right_triangle(tri[1] + tri[2] + tri[0], premise, 2) or update

            i = i + 6  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_3_right_triangle_determine(problem):
        """
        直角三角形 判定
        ①满足勾股定理  ==>  RT三角形
        ②三角形两边垂直  ==>  RT三角形
        ③三角形有一个角是直角  ==>  RT三角形
        """
        update = Theorem.theorem_2_pythagorean_inverse(problem)    # 勾股定理

        for pp in problem.perpendicular.items:    # 垂直关系
            if pp[1][1] == pp[2][0] and (pp[2][1] + pp[1][0]) in problem.line.items:    # 有 AB ⊥ BC
                premise = [problem.perpendicular.indexes[pp], problem.line.indexes[pp[2][1] + pp[1][0]]]
                update = problem.define_right_triangle(pp[1] + pp[2][1], premise, 3) or update

        for key in problem.sym_of_attr.keys():    # 有90°的角
            if key[0] == "DD" and problem.sym_of_attr[key][1] is not None and \
                    problem.sym_of_attr[key][1] == 90 and (key[1][2] + key[1][0]) in problem.line.items:
                update = problem.define_right_triangle(key[1], [-3, problem.line.indexes[key[1][2] + key[1][0]]], 3) \
                         or update

        return update

    @staticmethod
    def theorem_4_transitivity_of_parallel(problem):
        """
        平行的传递性
        01 AB // CD, CD // EF  ==>  AB // EF
        02 CD // AB, EF // CD  ==>  EF // AB (冗余)
        """
        update = False  # 存储应用定理是否更新了条件
        for item1 in problem.parallel.items:
            for item2 in problem.parallel.items:
                if item1[1] == item2[0] and item1[0] != item2[1]:
                    premise = [problem.parallel.indexes[item1], problem.parallel.indexes[item2]]
                    update = problem.define_parallel((item1[0], item2[1]), premise, 4) or update
        return update

    @staticmethod
    def theorem_5_transitivity_of_perpendicular(problem):
        """
        垂直的传递性
        01 AB ⊥ CD, CD // EF  ==>  AB ⊥ EF
        02 CD ⊥ BA, EF // CD  ==>  EF ⊥ BA (冗余)
        """
        update = False
        i = 0
        while i < len(problem.perpendicular.items):
            pp = problem.perpendicular.items[i]
            for pl in problem.parallel.items:
                if pp[2] == pl[0]:
                    premise = [problem.perpendicular.indexes[pp], problem.parallel.indexes[pl]]
                    update = problem.define_perpendicular(("$", pp[1], pl[1]), premise, 5) or update
            i = i + 4    # 跳过冗余表示
        return update

    @staticmethod
    def theorem_6_similar_triangle(problem):
        """
        相似三角形 性质
        两个三角形相似  ==>  对应角相等、对应边成比例
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_7_similar_triangle_inverse(problem):
        """
        相似三角形 定理
        对应角相等、对应边成比例  ==>  两个三角形相似
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_8_congruent_triangle(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应角相等、对应边相等
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.congruent.items):
            index = [problem.congruent.indexes[problem.congruent.items[i]]]  # 前提
            tri1 = problem.congruent.items[i][0]  # 三角形
            tri2 = problem.congruent.items[i][1]
            for i in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr((aType.LL.name, tri1[i] + tri1[(i + 1) % 3]))
                l_2 = problem.get_sym_of_attr((aType.LL.name, tri2[i] + tri2[(i + 1) % 3]))
                update = problem.define_equation(l_1 - l_2, index, 6) or update
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((aType.DA.name, tri1[i] + tri1[(i + 1) % 3] + tri1[(i + 2) % 3]))
                angle_2 = problem.get_sym_of_attr((aType.DA.name, tri2[i] + tri2[(i + 1) % 3] + tri2[(i + 2) % 3]))
                update = problem.define_equation(angle_1 - angle_2, index, 7) or update

            i = i + 6  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_9_congruent_triangle_inverse(problem):
        """
        全等三角形 定理
        SSS、SAS、ASA、AL、HL  ==>  两个三角形全等
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_10_perimeter_of_shape(problem):
        """
        三角形、圆、扇形、四边形周长公式
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.triangle.items):    # 三角形
            tri = problem.triangle.items[i]
            p = problem.get_sym_of_attr((aType.PT.name, tri))
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[2] + tri[0]))
            update = problem.define_equation(p - a - b - c, [problem.triangle.indexes[tri]], 10) or update
            i = i + 6    # 一个三角形6种表示

        i = 0
        while i < len(problem.quadrilateral.items):  # 四边形
            qua = problem.quadrilateral.items[i]
            p = problem.get_sym_of_attr((aType.PQ.name, qua))
            a = problem.get_sym_of_attr((aType.LL.name, qua[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, qua[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, qua[2:4]))
            d = problem.get_sym_of_attr((aType.LL.name, qua[3] + qua[0]))
            update = problem.define_equation(p - a - b - c - d, [problem.triangle.indexes[qua]], 10) or update
            i = i + 8  # 一个四边形6种表示

        for cir in problem.circle.items:    # 圆
            p = problem.get_sym_of_attr((aType.PC.name, cir))
            r = problem.get_sym_of_attr((aType.RC.name, cir))
            update = problem.define_equation(p - 2 * pi * r, [problem.circle.indexes[cir]], 10) or update

        for sec in problem.sector.items:    # 扇形
            p = problem.get_sym_of_attr((aType.PS.name, sec))
            r = problem.get_sym_of_attr((aType.RS.name, sec))
            d = problem.get_sym_of_attr((aType.DS.name, sec))
            update = problem.define_equation(p - pi * r * d / 180 - 2 * r, [problem.sector.indexes[sec]], 10) or update

        return update

    @staticmethod
    def _solve_targets(problem, target, target_equation):  # 求解目标值，并返回使用的最小方程组集合(前提)
        problem.equations.items.append(target_equation)  # 将目标方程添加到方程组
        result = solve(problem.equations.items)  # 求解equation
        problem.equations.items.remove(target_equation)  # 求解后，移除目标方程

        if len(result) == 0:  # 没有解，返回None
            return None, None

        if isinstance(result, list):  # 解不唯一，选择第一个
            result = result[0]

        if target in result.keys() and isinstance(result[target], Float):
            return abs(float(result[target])), [-3]  # 有实数解，返回解

        return None, None
