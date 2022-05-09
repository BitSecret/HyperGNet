import time
from facts import AttributionType as aType
from sympy import solve, Float, pi, sin, cos
from utility import Representation as rep


class Theorem:
    """------------theorem------------"""

    # 凡是带有 inverse 和 determine 的，在定理应用前都需要扩充条件的操作
    # hardcore版本，解题的可能性大一点，但是时间黑洞，轻易不要使用

    @staticmethod
    def theorem_1_pythagorean(problem):
        """
        勾股定理
        RT△  ==>  a**2 + b**2 - c**2 = 0
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
            i = i + rep.count_rt_tri  # 一个RT△有2种表示

        return update

    @staticmethod
    def theorem_2_pythagorean_inverse(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT△
        """
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
            i = i + rep.count_tri  # 一个三角形有6种表示

        return update

    @staticmethod
    def theorem_2_pythagorean_inverse_hardcore(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT△
        """
        problem.find_all_triangle()  # 条件扩充 找到所有的隐藏三角形
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

            i = i + rep.count_tri  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_3_right_triangle_determine(problem):
        """
        直角三角形 判定
        ①满足勾股定理  ==>  RT△
        ②三角形两边垂直  ==>  RT△
        ③三角形有一个角是直角  ==>  RT△
        """
        update = Theorem.theorem_2_pythagorean_inverse(problem)  # 勾股定理

        for pp in problem.perpendicular.items:  # 垂直关系
            if pp[1][1] == pp[2][0] and (pp[2][1] + pp[1][0]) in problem.line.items:  # 有 AB ⊥ BC
                premise = [problem.perpendicular.indexes[pp], problem.line.indexes[pp[2][1] + pp[1][0]]]
                update = problem.define_right_triangle(pp[1] + pp[2][1], premise, 3) or update

        for key in problem.sym_of_attr.keys():  # 有90°的角
            if key[0] == aType.DA.name and problem.value_of_sym[problem.sym_of_attr[key]] is not None and \
                    problem.value_of_sym[problem.sym_of_attr[key]] == pi / 2 and (
                    key[1][2] + key[1][0]) in problem.line.items:
                update = problem.define_right_triangle(key[1], [-3, problem.line.indexes[key[1][2] + key[1][0]]], 3) \
                         or update

        return update

    @staticmethod
    def theorem_4_transitivity_of_parallel(problem):
        """
        平行的传递性
        AB // CD, CD // EF  ==>  AB // EF
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
        AB ⊥ CD, CD // EF  ==>  AB ⊥ EF
        """
        update = False
        i = 0
        while i < len(problem.perpendicular.items):
            pp = problem.perpendicular.items[i]
            for pl in problem.parallel.items:
                if pp[2] == pl[0]:
                    premise = [problem.perpendicular.indexes[pp], problem.parallel.indexes[pl]]
                    update = problem.define_perpendicular(("$", pp[1], pl[1]), premise, 5) or update
            i = i + rep.count_perpendicular  # 跳过冗余表示
        return update

    @staticmethod
    def theorem_6_similar_triangle(problem):
        """
        相似三角形 性质
        相似△  ==>  对应角相等、对应边成比例
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.similar.items):
            index = [problem.similar.indexes[problem.similar.items[i]]]  # 前提
            tri1 = problem.similar.items[i][0]  # 三角形
            tri2 = problem.similar.items[i][1]
            ratios = []  # 对应边的比
            for j in range(3):
                # 对应边的比
                l_1 = problem.get_sym_of_attr((aType.LL.name, tri1[j] + tri1[(j + 1) % 3]))
                l_2 = problem.get_sym_of_attr((aType.LL.name, tri2[j] + tri2[(j + 1) % 3]))
                ratios.append(l_1 / l_2)
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((aType.DA.name, tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3]))
                angle_2 = problem.get_sym_of_attr((aType.DA.name, tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3]))
                update = problem.define_equation(angle_1 - angle_2, index, 6) or update
            update = problem.define_equation(ratios[0] - ratios[1], index, 6) or update  # 对应边的比值相等
            update = problem.define_equation(ratios[1] - ratios[2], index, 6) or update

            i = i + rep.count_similar  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_7_similar_triangle_determine(problem):
        """
        相似三角形 判定
        xxx  ==>  相似△
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
            for j in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr((aType.LL.name, tri1[j] + tri1[(j + 1) % 3]))
                l_2 = problem.get_sym_of_attr((aType.LL.name, tri2[j] + tri2[(j + 1) % 3]))
                update = problem.define_equation(l_1 - l_2, index, 8) or update
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((aType.DA.name, tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3]))
                angle_2 = problem.get_sym_of_attr((aType.DA.name, tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3]))
                update = problem.define_equation(angle_1 - angle_2, index, 8) or update

            i = i + rep.count_congruent  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_9_congruent_triangle_determine(problem):
        """
        全等三角形 判定
        SSS、SAS、ASA、AL、HL  ==>  两个三角形全等
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_10_triangle(problem):
        """
        三角形 性质
        △ABC  ==>  内角和为180°
        """
        update = False
        i = 0
        while i < len(problem.triangle.items):
            tri = problem.triangle.items[i]
            a = problem.get_sym_of_attr((aType.DA.name, tri))
            b = problem.get_sym_of_attr((aType.DA.name, tri[1:3] + tri[0]))
            c = problem.get_sym_of_attr((aType.DA.name, tri[2] + tri[0:2]))
            update = problem.define_equation(a + b + c - pi, [problem.triangle.indexes[tri]], 10) or update
            i = i + rep.count_tri  # 跳过冗余表示
        return update

    @staticmethod
    def theorem_11_isosceles_triangle(problem):
        """
        等腰三角形 性质
        等腰△  ==>  腰相等、底角相等
        """
        update = False
        i = 0
        while i < len(problem.isosceles_triangle.items):
            index = [problem.isosceles_triangle.indexes[problem.isosceles_triangle.items[i]]]  # 前提
            tri = problem.isosceles_triangle.items[i]
            # 两腰相等
            l_1 = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            l_2 = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))
            update = problem.define_equation(l_1 - l_2, index, 11) or update
            # 底角相等
            angle_1 = problem.get_sym_of_attr((aType.DA.name, tri))
            angle_2 = problem.get_sym_of_attr((aType.DA.name, tri[1:3] + tri[0]))
            update = problem.define_equation(angle_1 - angle_2, index, 11) or update

            i = i + rep.count_iso_tri  # 等腰三角形两种表示

        return update

    @staticmethod
    def theorem_12_isosceles_triangle_determine(problem):
        """
        等腰三角形 判定
        xxxx  ==>  等腰△
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_13_tangent_radius(problem):
        """
        圆的直径 性质
        直径所对的圆周角是直角
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_14_center_and_circumference_angle(problem):
        """
        圆的弦 性质
        同弦所对的圆周角是圆心角的一半
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_15_parallel(problem):
        """
        平行线 性质
        两直线平行  ==>  内错角相等，同旁内角互补
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_16_parallel_inverse(problem):
        """
        平行线 判定
        xxx  ==>  两直线平行
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_17_flat_angle(problem):
        """
        平角 性质
        pointOn(O, AB)  ==>  AOC + COB = 180°
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_18_intersecting_chord(problem):
        """
        相交弦 性质
        若圆内任意弦AB、弦CD交于点P  则PA·PB=PC·PD
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_19_polygon(problem):
        """
        多边形 性质
        多边形  ==>  内角和 = (n - 2 ) * 180°
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_20_angle_bisector(problem):
        """
        角平分线线 性质
        △ABC, AD是角平分线  ==>  AB/AC=BD/DC
        """
        update = False  # 存储应用定理是否更新了条件
        pass

    @staticmethod
    def theorem_21_sine(problem):
        """
        正弦定理
        """
        update = False
        problem.solve_equations()    # 应用定理前先解方程
        i = 0
        while i < len(problem.triangle.items):
            tri = problem.triangle.items[i]
            index = [problem.triangle.indexes[tri]]
            ratios = []
            known_count = [0, 0, 0]    # 记录方程中已知变量的个数, =3方程才有意义
            for j in range(3):
                line = problem.get_sym_of_attr((aType.LL.name, tri[j] + tri[(j + 1) % 3]))
                angle = problem.get_sym_of_attr((aType.DA.name, tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3]))
                if problem.value_of_sym[line] is not None:
                    known_count[j] += 1
                if problem.value_of_sym[angle] is not None:
                    known_count[j] += 1
                ratios.append(line / sin(angle))

            for j in range(3):
                if known_count[j] + known_count[(j + 1) % 3] == 3:
                    update = problem.define_equation(ratios[j] - ratios[(j + 1) % 3], index, 21) or update

            i = i + rep.count_tri    # 一个三角形多种表示

        return update

    @staticmethod
    def theorem_22_cosine(problem):
        """
        余弦定理
        """
        update = False
        i = 0
        while i < len(problem.triangle.items):
            tri = problem.triangle.items[i]
            index = [problem.triangle.indexes[tri]]
            for j in range(3):
                a = problem.get_sym_of_attr((aType.LL.name, tri[j] + tri[(j + 1) % 3]))
                b = problem.get_sym_of_attr((aType.LL.name, tri[(j + 1) % 3] + tri[(j + 2) % 3]))
                c = problem.get_sym_of_attr((aType.LL.name, tri[(j + 2) % 3] + tri[j]))
                angle = problem.get_sym_of_attr((aType.DA.name, tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3]))
                update = problem.define_equation(a ** 2 - b ** 2 - c ** 2 + 2 * b * c * cos(angle), index, 22) or update

            i = i + rep.count_tri  # 一个三角形多种表示

        return update

    @staticmethod
    def theorem_50_perimeter_of_tri(problem):
        """
        三角形、圆、扇形、四边形周长公式
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.triangle.items):  # 三角形
            tri = problem.triangle.items[i]
            p = problem.get_sym_of_attr((aType.PT.name, tri))
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[2] + tri[0]))
            update = problem.define_equation(p - a - b - c, [problem.triangle.indexes[tri]], 50) or update
            i = i + rep.count_tri  # 一个三角形6种表示

        return update

    @staticmethod
    def theorem_51_perimeter_of_shape(problem):
        """
        三角形、圆、扇形、四边形周长公式
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.triangle.items):  # 三角形
            tri = problem.triangle.items[i]
            p = problem.get_sym_of_attr((aType.PT.name, tri))
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[2] + tri[0]))
            update = problem.define_equation(p - a - b - c, [problem.triangle.indexes[tri]], 10) or update
            i = i + 6  # 一个三角形6种表示

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

        for cir in problem.circle.items:  # 圆
            p = problem.get_sym_of_attr((aType.PC.name, cir))
            r = problem.get_sym_of_attr((aType.RC.name, cir))
            update = problem.define_equation(p - 2 * pi * r, [problem.circle.indexes[cir]], 10) or update

        for sec in problem.sector.items:  # 扇形
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

        if isinstance(result, list):  # 解不唯一
            result = result[0]

        if target in result.keys() and isinstance(result[target], Float):
            return abs(float(result[target])), [-3]  # 有实数解，返回解

        return None, None
