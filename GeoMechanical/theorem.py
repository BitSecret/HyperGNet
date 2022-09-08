import time
from facts import AttributionType as aType
from facts import EquationType as eType
from facts import ConditionType as cType
from sympy import solve, Float, pi, sin, cos
from utility import Representation as rep


class Theorem:
    """------------常识------------"""

    @staticmethod
    def nous_1_extend_shape(problem):
        """
        拼图法构造新shape
        Shape(ABC), Shape(CBD) ==> Shape(ABD)
        """
        shape_update = False  # 是否更新了条件
        update = True
        traversed = []  # 记录已经计算过的，避免重复计算
        while update:
            update = False
            for shape1 in problem.conditions.items[cType.shape]:
                for shape2 in problem.conditions.items[cType.shape]:
                    if shape1[len(shape1) - 1] == shape2[0] and shape1[len(shape1) - 2] == shape2[1] and \
                            (shape1, shape2) not in traversed:
                        traversed.append((shape1, shape2))
                        same_length = 2
                        while same_length < len(shape1) and same_length < len(shape2):  # 共同点的数量
                            if shape1[len(shape1) - same_length - 1] == shape2[same_length]:
                                same_length += 1
                            else:
                                break
                        new_shape = shape1[0:len(shape1) - same_length + 1]  # shape1不同的部分、第一个共点
                        new_shape += shape2[same_length:len(shape2)]  # shape2不同的部分
                        new_shape += shape1[len(shape1) - 1]  # 第2个共点

                        i = 0  # 当前长度为3窗口起始的位置
                        while len(new_shape) > 2 and i < len(new_shape):
                            point1 = new_shape[i]
                            point2 = new_shape[(i + 1) % len(new_shape)]
                            point3 = new_shape[(i + 2) % len(new_shape)]

                            is_coll = False  # 判断是否共线
                            for coll in problem.conditions.items[cType.collinear]:
                                if point1 in coll and point2 in coll and point3 in coll:
                                    is_coll = True
                                    new_shape = new_shape.replace(point2, "")  # 三点共线，去掉中间的点
                                    break
                            if not is_coll:  # 不共线，窗口后移
                                i += 1

                        if 2 < len(new_shape) == len(set(new_shape)):  # 是图形且没有环
                            premise = [problem.conditions.get_index(shape1, cType.shape),
                                       problem.conditions.get_index(shape2, cType.shape)]
                            update = problem.define_shape(new_shape, premise, 1) or update
                            shape_update = update or shape_update

        return shape_update

    @staticmethod
    def nous_2_extend_shape_and_area_addition(problem):
        """
        拼图法构造新shape，并记录shape之间的面积相加关系
        Shape(ABC), Shape(CBD) ==> Shape(ABD), a_abc + a_cbd - a_abd = 0
        """
        shape_update = False  # 是否更新了条件
        update = True
        traversed = []  # 记录已经计算过的，避免重复计算
        while update:
            update = False
            for shape1 in problem.conditions.items[cType.shape]:
                for shape2 in problem.conditions.items[cType.shape]:
                    if shape1[len(shape1) - 1] == shape2[0] and shape1[len(shape1) - 2] == shape2[1] and \
                            (shape1, shape2) not in traversed:
                        traversed.append((shape1, shape2))
                        same_length = 2
                        while same_length < len(shape1) and same_length < len(shape2):  # 共同点的数量
                            if shape1[len(shape1) - same_length - 1] == shape2[same_length]:
                                same_length += 1
                            else:
                                break
                        new_shape = shape1[0:len(shape1) - same_length + 1]  # shape1不同的部分、第一个共点
                        new_shape += shape2[same_length:len(shape2)]  # shape2不同的部分
                        new_shape += shape1[len(shape1) - 1]  # 第2个共点

                        i = 0  # 当前长度为3窗口起始的位置
                        while len(new_shape) > 2 and i < len(new_shape):
                            point1 = new_shape[i]
                            point2 = new_shape[(i + 1) % len(new_shape)]
                            point3 = new_shape[(i + 2) % len(new_shape)]

                            is_coll = False  # 判断是否共线
                            for coll in problem.conditions.items[cType.collinear]:
                                if point1 in coll and point2 in coll and point3 in coll:
                                    is_coll = True
                                    new_shape = new_shape.replace(point2, "")  # 三点共线，去掉中间的点
                                    break
                            if not is_coll:  # 不共线，窗口后移
                                i += 1

                        if 2 < len(new_shape) == len(set(new_shape)):  # 是图形且没有环
                            premise = [problem.conditions.get_index(shape1, cType.shape),
                                       problem.conditions.get_index(shape2, cType.shape)]
                            update = problem.define_shape(new_shape, premise, 2) or update
                            shape_update = update or shape_update
                            a1 = problem.get_sym_of_attr((aType.A.name, shape1))
                            a2 = problem.get_sym_of_attr((aType.A.name, shape2))
                            a3 = problem.get_sym_of_attr((aType.A.name, new_shape))
                            problem.define_equation(a1 + a2 - a3, eType.basic, premise, 2)

        return shape_update

    @staticmethod
    def nous_3_extend_line_addition(problem):
        """
        拼图法构造线段之间的相加关系
        Line(AB), Line(BC) ==> Line(AC), ll_ab + ll_bc - ll_ac = 0
        """
        update = False
        for coll in problem.conditions.items[cType.collinear]:
            premise = [problem.conditions.get_index(coll, cType.collinear)]
            for i in range(0, len(coll) - 2):
                for j in range(i + 1, len(coll) - 1):
                    for k in range(j + 1, len(coll)):
                        sym1 = problem.get_sym_of_attr((aType.LL.name, coll[i] + coll[j]))
                        sym2 = problem.get_sym_of_attr((aType.LL.name, coll[j] + coll[k]))
                        sym3 = problem.get_sym_of_attr((aType.LL.name, coll[i] + coll[k]))
                        update = problem.define_equation(sym1 + sym2 - sym3, eType.basic, premise, 3) or update  # 相加关系

        return update

    @staticmethod
    def nous_4_extend_angle_addition(problem):
        """
        拼图法构造角之间的相加关系
        Angle(ABC), Angle(CBD) ==> Angle(ABD), da_abc + da_cbd - da_abd = 0
        """
        update = False
        for angle1 in problem.conditions.items[cType.angle]:
            for angle2 in problem.conditions.items[cType.angle]:
                if angle1[0] == angle2[2] and angle1[1] == angle2[1]:
                    angle3 = angle2[0:2] + angle1[2]
                    sym1 = problem.get_sym_of_attr((aType.DA.name, angle1))
                    sym2 = problem.get_sym_of_attr((aType.DA.name, angle2))
                    sym3 = problem.get_sym_of_attr((aType.DA.name, angle3))
                    premise = [problem.conditions.get_index(angle1, cType.angle),
                               problem.conditions.get_index(angle2, cType.angle)]
                    update = problem.define_equation(sym1 + sym2 - sym3, eType.basic, premise, 4) or update

        return update

    @staticmethod
    def nous_5_extend_flat_angle(problem):
        """
        拼图法赋予平角180°
        Line(AB), Line(BC) ==> da_abc = 180°
        """
        update = False
        for coll in problem.conditions.items[cType.collinear]:
            premise = [problem.conditions.get_index(coll, cType.collinear)]
            for i in range(0, len(coll) - 2):
                for j in range(i + 1, len(coll) - 1):
                    for k in range(j + 1, len(coll)):
                        sym_of_angle = problem.get_sym_of_attr((aType.DA.name, coll[i] + coll[j] + coll[k]))
                        update = problem.define_equation(sym_of_angle - 180, eType.basic, premise, 5) or update
                        sym_of_angle = problem.get_sym_of_attr((aType.DA.name, coll[k] + coll[j] + coll[i]))
                        update = problem.define_equation(sym_of_angle - 180, eType.basic, premise, 5) or update

        return update

    @staticmethod
    def nous_6_(problem):
        pass

    @staticmethod
    def nous_7_(problem):
        pass

    @staticmethod
    def nous_8_(problem):
        pass

    @staticmethod
    def nous_9_(problem):
        pass

    @staticmethod
    def nous_10_(problem):
        pass

    """------------辅助线------------"""

    @staticmethod
    def auxiliary_11_(problem):
        pass

    @staticmethod
    def auxiliary_12_(problem):
        pass

    @staticmethod
    def auxiliary_13_(problem):
        pass

    @staticmethod
    def auxiliary_14_(problem):
        pass

    @staticmethod
    def auxiliary_15_(problem):
        pass

    @staticmethod
    def auxiliary_16_(problem):
        pass

    @staticmethod
    def auxiliary_17_(problem):
        pass

    @staticmethod
    def auxiliary_18_(problem):
        pass

    @staticmethod
    def auxiliary_19_(problem):
        pass

    @staticmethod
    def auxiliary_20_(problem):
        pass

    """------------定理------------"""

    @staticmethod
    def theorem_21_triangle_property_angle_sum(problem):
        """
        三角形 性质
        △ABC  ==>  内角和为180°
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            a = problem.get_sym_of_attr((aType.DA.name, tri))
            b = problem.get_sym_of_attr((aType.DA.name, tri[1:3] + tri[0]))
            c = problem.get_sym_of_attr((aType.DA.name, tri[2] + tri[0:2]))
            equation = a + b + c - 180
            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(equation, eType.theorem, premise, 10) or update
            i = i + rep.count_tri  # 跳过冗余表示
        return update

    @staticmethod
    def theorem_22_right_triangle_pythagorean(problem):
        """
        勾股定理
        RT△  ==>  a**2 + b**2 - c**2 = 0
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.right_triangle]):
            rt = problem.conditions.items[cType.right_triangle][i]
            a = problem.get_sym_of_attr((aType.LL.name, rt[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, rt[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, rt[0] + rt[2]))
            update = problem.define_equation(a ** 2 + b ** 2 - c ** 2,
                                             eType.theorem,
                                             [problem.conditions.get_index(rt, cType.right_triangle)],
                                             1) or update
            i = i + rep.count_rt_tri

        return update

    @staticmethod
    def theorem_23_right_triangle_property(problem):
        pass

    @staticmethod
    def theorem_24_right_triangle_pythagorean_inverse(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT△
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))

            if problem.value_of_sym[a] is not None and \
                    problem.value_of_sym[b] is not None and \
                    problem.value_of_sym[c] is not None:
                premise = [problem.conditions.get_index(a - problem.value_of_sym[a], cType.equation),
                           problem.conditions.get_index(b - problem.value_of_sym[b], cType.equation),
                           problem.conditions.get_index(c - problem.value_of_sym[c], cType.equation)]
                a = problem.value_of_sym[a]
                b = problem.value_of_sym[b]
                c = problem.value_of_sym[c]
                if abs(a ** 2 + b ** 2 - c ** 2) < 0.001:
                    update = problem.define_right_triangle(tri, premise, 2) or update
                elif abs(a ** 2 - b ** 2 + c ** 2) < 0.001:
                    update = problem.define_right_triangle(tri[1] + tri[0] + tri[2], premise, 2) or update
                elif abs(- a ** 2 + b ** 2 + c ** 2) < 0.001:
                    update = problem.define_right_triangle(tri[1] + tri[2] + tri[0], premise, 2) or update
            i = i + rep.count_tri

        return update

    @staticmethod
    def theorem_25_right_triangle_judgment(problem):
        """
        直角三角形 判定
        ②三角形有一个角是直角  ==>  RT△
        """
        update = False

        for key in problem.sym_of_attr.keys():
            if key[0] == aType.DA.name:  # 如果是角
                sym = problem.sym_of_attr[key]
                value = problem.value_of_sym[problem.sym_of_attr[key]]
                if value is not None and abs(value - 90) < 0.01 and \
                        key[1] in problem.conditions.items[cType.triangle]:  # 如果是90°的角

                    premise = [problem.conditions.get_index(sym - value, cType.equation),
                               problem.conditions.get_index(key[1], cType.triangle)]
                    update = problem.define_right_triangle(key[1], premise, 3) or update

        return update

    @staticmethod
    def theorem_26_isosceles_triangle_property_angle_equal(problem):
        """
        等腰三角形 性质
        等腰△  ==>  腰相等、底角相等
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.isosceles_triangle]):
            tri = problem.conditions.items[cType.isosceles_triangle][i]
            premise = [problem.conditions.get_index(tri, cType.isosceles_triangle)]  # 前提
            # 两腰相等
            l_1 = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            l_2 = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))
            update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 11) or update
            # 底角相等
            angle_1 = problem.get_sym_of_attr((aType.DA.name, tri))
            angle_2 = problem.get_sym_of_attr((aType.DA.name, tri[1:3] + tri[0]))
            update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 11) or update
            i = i + rep.count_iso_tri

        return update

    @staticmethod
    def theorem_27_isosceles_triangle_property_side_equal(problem):
        """
        等腰三角形 性质
        等腰△  ==>  腰相等、底角相等
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.isosceles_triangle]):
            tri = problem.conditions.items[cType.isosceles_triangle][i]
            premise = [problem.conditions.get_index(tri, cType.isosceles_triangle)]  # 前提
            # 两腰相等
            l_1 = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            l_2 = problem.get_sym_of_attr((aType.LL.name, tri[0] + tri[2]))
            update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 11) or update
            # 底角相等
            angle_1 = problem.get_sym_of_attr((aType.DA.name, tri))
            angle_2 = problem.get_sym_of_attr((aType.DA.name, tri[1:3] + tri[0]))
            update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 11) or update
            i = i + rep.count_iso_tri

        return update

    @staticmethod
    def theorem_28_isosceles_triangle_property_line_coincidence(problem):
        pass

    @staticmethod
    def theorem_29_isosceles_triangle_judgment_angle_equal(problem):
        pass

    @staticmethod
    def theorem_30_isosceles_triangle_judgment_side_equal(problem):
        pass

    @staticmethod
    def theorem_31_equilateral_triangle_property_angle_equal(problem):
        pass

    @staticmethod
    def theorem_32_equilateral_triangle_property_side_equal(problem):
        pass

    @staticmethod
    def theorem_33_equilateral_triangle_judgment_angle_equal(problem):
        pass

    @staticmethod
    def theorem_34_equilateral_triangle_judgment_side_equal(problem):
        pass

    @staticmethod
    def theorem_35_equilateral_triangle_judgment_isos_and_angle60(problem):
        pass

    @staticmethod
    def theorem_36_intersect_property(problem):
        pass

    @staticmethod
    def theorem_37_parallel_property(problem):
        pass

    @staticmethod
    def theorem_38_parallel_judgment(problem):
        pass

    @staticmethod
    def theorem_39_perpendicular_property(problem):
        pass

    @staticmethod
    def theorem_40_perpendicular_judgment(problem):
        pass

    @staticmethod
    def theorem_41_parallel_perpendicular_combination(problem):
        """
        垂直的传递性
        AB ⊥ CD, CD // EF  ==>  AB ⊥ EF
        """
        update = False
        for item1 in problem.parallel.items:
            for item2 in problem.parallel.items:
                if item1[1] == item2[0] and item1[0] != item2[1]:
                    premise = [problem.parallel.indexes[item1], problem.parallel.indexes[item2]]
                    update = problem.define_parallel((item1[0], item2[1]), premise, 4) or update

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
    def theorem_42_perpendicular_bisector_property_perpendicular(problem):
        pass

    @staticmethod
    def theorem_43_perpendicular_bisector_property_bisector(problem):
        pass

    @staticmethod
    def theorem_44_perpendicular_bisector_property_distance_equal(problem):
        pass

    @staticmethod
    def theorem_45_perpendicular_bisector_judgment(problem):
        pass

    @staticmethod
    def theorem_46_bisector_property_line_ratio(problem):
        pass

    @staticmethod
    def theorem_47_bisector_property_angle_equal(problem):
        pass

    @staticmethod
    def theorem_48_bisector_property_distance_equal(problem):
        pass

    @staticmethod
    def theorem_49_bisector_judgment_line_ratio(problem):
        pass

    @staticmethod
    def theorem_50_bisector_judgment_angle_equal(problem):
        pass

    @staticmethod
    def theorem_51_altitude_property(problem):
        pass

    @staticmethod
    def theorem_52_altitude_judgment(problem):
        pass

    @staticmethod
    def theorem_53_neutrality_property_similar(problem):
        pass

    @staticmethod
    def theorem_54_neutrality_property_angle_equal(problem):
        pass

    @staticmethod
    def theorem_55_neutrality_property_line_ratio(problem):
        pass

    @staticmethod
    def theorem_56_neutrality_judgment(problem):
        pass

    @staticmethod
    def theorem_57_circumcenter_property(problem):
        pass

    @staticmethod
    def theorem_58_incenter_property(problem):
        pass

    @staticmethod
    def theorem_59_centroid_property(problem):
        pass

    @staticmethod
    def theorem_60_orthocenter_property(problem):
        pass

    @staticmethod
    def theorem_61_congruent_property_line_euqal(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应角相等、对应边相等
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.conditions.items[cType.congruent]):
            congruent = problem.conditions.items[cType.congruent][i]
            tri1 = congruent[0]  # 三角形
            tri2 = congruent[1]
            premise = [problem.conditions.get_index(congruent, cType.congruent)]  # 前提

            for j in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr((aType.LL.name, tri1[j] + tri1[(j + 1) % 3]))
                l_2 = problem.get_sym_of_attr((aType.LL.name, tri2[j] + tri2[(j + 1) % 3]))
                update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 8) or update
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((aType.DA.name, tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3]))
                angle_2 = problem.get_sym_of_attr((aType.DA.name, tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3]))
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 8) or update

            i = i + rep.count_congruent  # 一个全等关系有3种表示

        return update

    @staticmethod
    def theorem_62_congruent_property_angle_equal(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应角相等、对应边相等
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.conditions.items[cType.congruent]):
            congruent = problem.conditions.items[cType.congruent][i]
            tri1 = congruent[0]  # 三角形
            tri2 = congruent[1]
            premise = [problem.conditions.get_index(congruent, cType.congruent)]  # 前提

            for j in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr((aType.LL.name, tri1[j] + tri1[(j + 1) % 3]))
                l_2 = problem.get_sym_of_attr((aType.LL.name, tri2[j] + tri2[(j + 1) % 3]))
                update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 8) or update
                # 对应角相等
                angle_1 = problem.get_sym_of_attr((aType.DA.name, tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3]))
                angle_2 = problem.get_sym_of_attr((aType.DA.name, tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3]))
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 8) or update

            i = i + rep.count_congruent  # 一个全等关系有3种表示

        return update

    @staticmethod
    def theorem_63_congruent_property_area_equal(problem):
        pass

    @staticmethod
    def theorem_64_congruent_judgment_sss(problem):
        pass

    @staticmethod
    def theorem_65_congruent_judgment_sas(problem):
        pass

    @staticmethod
    def theorem_66_congruent_judgment_asa(problem):
        pass

    @staticmethod
    def theorem_67_congruent_judgment_aas(problem):
        pass

    @staticmethod
    def theorem_68_congruent_judgment_hl(problem):
        pass

    @staticmethod
    def theorem_69_similar_property_angle_euqal(problem):
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
    def theorem_70_similar_property_line_ratio(problem):
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
    def theorem_71_similar_property_perimeter_ratio(problem):
        pass

    @staticmethod
    def theorem_72_similar_property_area_square_ratio(problem):
        pass

    @staticmethod
    def theorem_73_similar_judgment_sss(problem):
        pass

    @staticmethod
    def theorem_74_similar_judgment_sas(problem):
        pass

    @staticmethod
    def theorem_75_similar_judgment_aa(problem):
        pass

    @staticmethod
    def theorem_76_triangle_perimeter_formula(problem):
        """
        三角形周长公式
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):  # 三角形
            tri = problem.conditions.items[cType.triangle][i]
            p = problem.get_sym_of_attr((aType.P.name, tri))
            a = problem.get_sym_of_attr((aType.LL.name, tri[0:2]))
            b = problem.get_sym_of_attr((aType.LL.name, tri[1:3]))
            c = problem.get_sym_of_attr((aType.LL.name, tri[2] + tri[0]))
            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(p - a - b - c, eType.theorem, premise, 10) or update
            i = i + rep.count_tri

        return update

    @staticmethod
    def theorem_77_triangle_area_formula_common(problem):
        pass

    @staticmethod
    def theorem_78_triangle_area_formula_heron(problem):
        pass

    @staticmethod
    def theorem_79_triangle_area_formula_sine(problem):
        pass

    @staticmethod
    def theorem_80_sine(problem):
        """
        正弦定理
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            ratios_unit = []
            known_count = [0, 0, 0]  # 记录方程中已知变量的个数, =3方程才有意义
            for j in range(3):
                line = problem.get_sym_of_attr((aType.LL.name, tri[j] + tri[(j + 1) % 3]))
                angle = problem.get_sym_of_attr((aType.DA.name, tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3]))
                if problem.value_of_sym[line] is not None:
                    known_count[j] += 1
                if problem.value_of_sym[angle] is not None:
                    known_count[j] += 1
                ratios_unit.append(line / sin(angle * pi / 180))

            for j in range(3):
                if known_count[j] + known_count[(j + 1) % 3] == 3:
                    equation = ratios_unit[j] - ratios_unit[(j + 1) % 3]
                    premise = [problem.conditions.get_index(tri, cType.triangle)]
                    update = problem.define_equation(equation, eType.theorem, premise, 21) or update

            i = i + rep.count_tri  # 一个三角形多种表示

        return update

    @staticmethod
    def theorem_81_cosine(problem):
        """
        余弦定理
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            for j in range(3):
                a = problem.get_sym_of_attr((aType.LL.name, tri[j] + tri[(j + 1) % 3]))
                b = problem.get_sym_of_attr((aType.LL.name, tri[(j + 1) % 3] + tri[(j + 2) % 3]))
                c = problem.get_sym_of_attr((aType.LL.name, tri[(j + 2) % 3] + tri[j]))
                angle = problem.get_sym_of_attr((aType.DA.name, tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3]))
                equation = a ** 2 - b ** 2 - c ** 2 + 2 * b * c * cos(angle * pi / 180)
                premise = [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(equation, eType.theorem, premise, 22) or update

            i = i + rep.count_tri  # 一个三角形多种表示

        return update
