from facts import AttributionType as aType
from facts import EquationType as eType
from facts import ConditionType as cType
from sympy import pi, sin, cos, sqrt
from utility import Representation as rep
from utility import Utility as util


class Theorem:
    """------------常识------------"""

    @staticmethod
    def nous_1_area_addition(problem):
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
                            a1 = problem.get_sym_of_attr(shape1, aType.AS)
                            a2 = problem.get_sym_of_attr(shape2, aType.AS)
                            a3 = problem.get_sym_of_attr(new_shape, aType.AS)
                            problem.define_equation(a1 + a2 - a3, eType.basic, premise, 2)

        return shape_update

    @staticmethod
    def nous_2_line_addition(problem):
        """
        拼图法构造线段之间的相加关系
        Collinear(ABC) ==> ll_ab + ll_bc - ll_ac = 0
        """
        update = False
        for coll in problem.conditions.items[cType.collinear]:
            premise = [problem.conditions.get_index(coll, cType.collinear)]
            for i in range(0, len(coll) - 2):
                for j in range(i + 1, len(coll) - 1):
                    for k in range(j + 1, len(coll)):
                        sym1 = problem.get_sym_of_attr(coll[i] + coll[j], aType.LL)
                        sym2 = problem.get_sym_of_attr(coll[j] + coll[k], aType.LL)
                        sym3 = problem.get_sym_of_attr(coll[i] + coll[k], aType.LL)
                        update = problem.define_equation(sym1 + sym2 - sym3, eType.basic, premise, 3) or update  # 相加关系

        return update

    @staticmethod
    def nous_3_angle_addition(problem):
        """
        拼图法构造角之间的相加关系
        Angle(ABC), Angle(CBD) ==> Angle(ABD), da_abc + da_cbd - da_abd = 0
        注：之所以搞这么麻烦是因为会发生组合爆炸
        """
        update = False
        init_shape = []
        for shape in problem.conditions.items[cType.shape]:  # 得到构图图形
            if problem.conditions.get_premise(shape, cType.shape)[0] == -1:
                init_shape.append(shape)

        init_angle = []
        for shape in init_shape:
            count = len(shape)
            i = 0
            while i < count:
                init_angle.append(shape[i] + shape[(i + 1) % count] + shape[(i + 2) % count])
                i += 1

        for angle1 in init_angle:
            for angle2 in init_angle:
                if angle1[0] == angle2[2] and angle1[1] == angle2[1]:
                    angle3 = angle2[0:2] + angle1[2]
                    sym1 = problem.get_sym_of_attr(angle1, aType.MA)
                    sym2 = problem.get_sym_of_attr(angle2, aType.MA)
                    sym3 = problem.get_sym_of_attr(angle3, aType.MA)
                    premise = [problem.conditions.get_index(angle1, cType.angle),
                               problem.conditions.get_index(angle2, cType.angle)]
                    update = problem.define_equation(sym1 + sym2 - sym3, eType.basic, premise, 3) or update
        return update

    @staticmethod
    def nous_4_intersect_extend(problem):
        pass

    @staticmethod
    def nous_5_perpendicular_extend(problem):
        pass

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
            a = problem.get_sym_of_attr(tri, aType.MA)
            b = problem.get_sym_of_attr(tri[1:3] + tri[0], aType.MA)
            c = problem.get_sym_of_attr(tri[2] + tri[0:2], aType.MA)
            equation = a + b + c - 180
            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(equation, eType.theorem, premise, 21) or update
            i = i + rep.count_triangle  # 跳过冗余表示
        return update

    @staticmethod
    def theorem_22_triangle_property_equal_line_angle(problem):
        """
        三角形 性质 等边对等角
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            line1 = problem.get_sym_of_attr(tri[0] + tri[1], aType.LL)
            line2 = problem.get_sym_of_attr(tri[2] + tri[0], aType.LL)
            angle1 = problem.get_sym_of_attr(tri[0] + tri[1] + tri[2], aType.MA)
            angle2 = problem.get_sym_of_attr(tri[1] + tri[2] + tri[0], aType.MA)
            m = problem.get_sym_of_attr("1", aType.M)

            result, premise = problem.solve_equations(m, m + line1 - line2)  # 等边对等角
            if result is not None and abs(result) < 0.01:
                premise += [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(angle1 - angle2, eType.basic, premise, 22) or update
            result, premise = problem.solve_equations(m, m + angle1 - angle2)  # 等角对等边
            if result is not None and abs(result) < 0.01:
                premise += [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(line1 - line2, eType.basic, premise, 22) or update

        return update

    @staticmethod
    def theorem_23_pythagorean(problem):
        """
        勾股定理
        RT△  ==>  a**2 + b**2 - c**2 = 0
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.right_triangle]):
            rt = problem.conditions.items[cType.right_triangle][i]
            a = problem.get_sym_of_attr(rt[0:2], aType.LL)
            b = problem.get_sym_of_attr(rt[1:3], aType.LL)
            c = problem.get_sym_of_attr(rt[0] + rt[2], aType.LL)
            update = problem.define_equation(a ** 2 + b ** 2 - c ** 2,
                                             eType.theorem,
                                             [problem.conditions.get_index(rt, cType.right_triangle)],
                                             22) or update
            i = i + rep.count_right_triangle

        return update

    @staticmethod
    def theorem_24_right_triangle_property_rt(problem):
        """
        直角三角形性质
        RT△  ==>  角90°、边垂直
        """
        update = False
        for rt in problem.conditions.items[cType.right_triangle]:
            premise = [problem.conditions.get_index(rt, cType.right_triangle)]
            sym = problem.get_sym_of_attr(rt, aType.MA)
            update = problem.define_equation(sym - 90, eType.theorem, premise, 23) or update  # 角90°
            update = problem.define_perpendicular((rt[0], rt[0:2], rt[2] + rt[1]), premise, 23) or update  # 边垂直

        return update

    @staticmethod
    def theorem_25_right_triangle_property_opposite_side(problem):
        """
        直角三角形性质
        RT△  ==>  30°所对直角边等于斜边一半
        """
        update = False
        for rt in problem.conditions.items[cType.right_triangle]:
            premise = [problem.conditions.get_index(rt, cType.right_triangle)]

            angle = problem.get_sym_of_attr(rt[1] + rt[2] + rt[0], aType.MA)
            m = problem.get_sym_of_attr("1", aType.M)
            result, eq_premise = problem.solve_equations(m, m - angle + 30)
            if result is not None and abs(result) < 0.01:
                a = problem.get_sym_of_attr(rt[0:2], aType.LL)
                b = problem.get_sym_of_attr(rt[1:3], aType.LL)
                c = problem.get_sym_of_attr(rt[2] + rt[0], aType.LL)  # 斜边
                update = problem.define_equation(a - 0.5 * c, eType.theorem, premise + eq_premise, 24) or update
                update = problem.define_equation(b - 0.866 * c, eType.theorem, premise + eq_premise, 24) or update

            angle = problem.get_sym_of_attr(rt[2] + rt[0] + rt[1], aType.MA)
            m = problem.get_sym_of_attr("1", aType.M)
            result, eq_premise = problem.solve_equations(m, m - angle + 30)
            if result is not None and abs(result) < 0.01:
                a = problem.get_sym_of_attr(rt[0:2], aType.LL)
                b = problem.get_sym_of_attr(rt[1:3], aType.LL)
                c = problem.get_sym_of_attr(rt[2] + rt[0], aType.LL)  # 斜边
                update = problem.define_equation(b - 0.5 * c, eType.theorem, premise + eq_premise, 24) or update
                update = problem.define_equation(a - 0.866 * c, eType.theorem, premise + eq_premise, 24) or update

        return update

    @staticmethod
    def theorem_26_pythagorean_inverse(problem):
        """
        勾股定理 逆定理
        a**2 + b**2 - c**2 = 0  ==>  RT△
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            if tri not in problem.conditions.items[cType.right_triangle]:
                a = problem.get_sym_of_attr(tri[0:2], aType.LL)
                b = problem.get_sym_of_attr(tri[1:3], aType.LL)
                c = problem.get_sym_of_attr(tri[0] + tri[2], aType.LL)

                if problem.value_of_sym[a] is not None and \
                        problem.value_of_sym[b] is not None and \
                        problem.value_of_sym[c] is not None:
                    m = problem.get_sym_of_attr("1", aType.M)
                    result, premise = problem.solve_equations(m, m - a ** 2 - b ** 2 + c ** 2)
                    if result is not None and abs(result) < 0.01:
                        update = problem.define_right_triangle(tri, premise, 25) or update
                    result, premise = problem.solve_equations(m, m - a ** 2 + b ** 2 - c ** 2)
                    if result is not None and abs(result) < 0.01:
                        update = problem.define_right_triangle(tri[1] + tri[0] + tri[2], premise, 25) or update
                    result, premise = problem.solve_equations(m, m + a ** 2 - b ** 2 - c ** 2)
                    if result is not None and abs(result) < 0.01:
                        update = problem.define_right_triangle(tri[1] + tri[2] + tri[0], premise, 25) or update

            i = i + rep.count_triangle

        return update

    @staticmethod
    def theorem_27_right_triangle_judgment(problem):
        """
        直角三角形 判定
        角是直角  ==>  RT△
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            angle = problem.get_sym_of_attr(tri, aType.MA)
            m = problem.get_sym_of_attr("1", aType.M)
            result, premise = problem.solve_equations(m, m - angle + 90)
            if result is not None and abs(result) < 0.01:
                premise += [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_right_triangle(tri, premise, 27) or update
        return update

    @staticmethod
    def theorem_28_isosceles_triangle_property_angle_equal(problem):
        """
        等腰三角形 性质
        等腰△  ==>  底角相等
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.isosceles_triangle]):
            tri = problem.conditions.items[cType.isosceles_triangle][i]
            premise = [problem.conditions.get_index(tri, cType.isosceles_triangle)]  # 前提
            # 底角相等
            angle_1 = problem.get_sym_of_attr(tri, aType.MA)
            angle_2 = problem.get_sym_of_attr(tri[1:3] + tri[0], aType.MA)
            update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 27) or update

            i = i + rep.count_isosceles_triangle

        return update

    @staticmethod
    def theorem_29_isosceles_triangle_property_side_equal(problem):
        """
        等腰三角形 性质
        等腰△  ==>  腰相等
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.isosceles_triangle]):
            tri = problem.conditions.items[cType.isosceles_triangle][i]
            premise = [problem.conditions.get_index(tri, cType.isosceles_triangle)]  # 前提
            # 两腰相等
            l_1 = problem.get_sym_of_attr(tri[0:2], aType.LL)
            l_2 = problem.get_sym_of_attr(tri[0] + tri[2], aType.LL)
            update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 28) or update

            i = i + rep.count_isosceles_triangle

        return update

    @staticmethod
    def theorem_30_isosceles_triangle_property_line_coincidence(problem):
        """
        等腰三角形 性质
        等腰三角形  ==>  高、中线、角平分线、垂直平分线重合
        """
        update = False
        lines = []
        triangles = []
        premises = []
        for tri in problem.conditions.items[cType.isosceles_triangle]:
            premise_tri = [problem.conditions.get_index(tri, cType.isosceles_triangle)]
            for is_altitude in problem.conditions.items[cType.is_altitude]:
                if is_altitude[1] == tri:
                    lines.append(is_altitude[0])
                    triangles.append(tri)
                    premise = premise_tri + [problem.conditions.get_index(is_altitude, cType.is_altitude)]
                    premises.append(premise)
            for median in problem.conditions.items[cType.median]:
                if median[1] == tri:
                    lines.append(median[0])
                    triangles.append(tri)
                    premise = premise_tri + [problem.conditions.get_index(median, cType.median)]
                    premises.append(premise)
            for bisector in problem.conditions.items[cType.bisector]:
                if bisector[1] == tri:
                    lines.append(bisector[0])
                    triangles.append(tri)
                    premise = premise_tri + [problem.conditions.get_index(bisector, cType.bisector)]
                    premises.append(premise)
            for perpendicular_bisector in problem.conditions.items[cType.perpendicular_bisector]:
                if perpendicular_bisector[1] == tri[1:3] and perpendicular_bisector[2][0] == tri[0]:
                    lines.append(tri[0] + perpendicular_bisector[0])
                    triangles.append(tri)
                    premise = premise_tri + [
                        problem.conditions.get_index(perpendicular_bisector, cType.perpendicular_bisector)]
                    premises.append(premise)

        for i in range(len(lines)):
            update = problem.define_is_altitude((lines[i], triangles[i]), premises[i], 29) or update
            update = problem.define_median((lines[i], triangles[i]), premises[i], 29) or update
            update = problem.define_bisector((lines[i], triangles[i]), premises[i], 29) or update
            update = problem.define_perpendicular_bisector((lines[i][1], triangles[i][1:3], lines[i]),
                                                           premises[i], 29) or update

        return update

    @staticmethod
    def theorem_31_isosceles_triangle_judgment_angle_equal(problem):
        """
        等腰三角形 判定
        两底角相等  ==>  等腰三角形
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            if tri not in problem.conditions.items[cType.isosceles_triangle]:
                a1 = problem.get_sym_of_attr(tri, aType.MA)
                a2 = problem.get_sym_of_attr(tri[1] + tri[2] + tri[0], aType.MA)
                m = problem.get_sym_of_attr("1", aType.M)
                result, eq_premise = problem.solve_equations(m, m - a1 + a2)
                if result is not None and abs(result) < 0.01:
                    premise = [problem.conditions.get_index(tri, cType.triangle)] + eq_premise
                    update = problem.define_isosceles_triangle(tri, premise, 30) or update
        return update

    @staticmethod
    def theorem_32_isosceles_triangle_judgment_side_equal(problem):
        """
        等腰三角形 判定
        两腰相等  ==>  等腰三角形
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            if tri not in problem.conditions.items[cType.isosceles_triangle]:
                s1 = problem.get_sym_of_attr(tri[0:2], aType.LL)
                s2 = problem.get_sym_of_attr(tri[2] + tri[0], aType.LL)
                m = problem.get_sym_of_attr("1", aType.M)
                result, eq_premise = problem.solve_equations(m, m - s1 + s2)
                if result is not None and abs(result) < 0.01:
                    premise = [problem.conditions.get_index(tri, cType.triangle)] + eq_premise
                    update = problem.define_isosceles_triangle(tri, premise, 31) or update
        return update

    @staticmethod
    def theorem_33_equilateral_triangle_property_angle_equal(problem):
        """
        等边三角形 性质
        等边△  ==>  角相等为60°
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.equilateral_triangle]):
            tri = problem.conditions.items[cType.equilateral_triangle][i]
            angle1 = problem.get_sym_of_attr(tri, aType.MA)
            angle2 = problem.get_sym_of_attr(tri[1] + tri[2] + tri[0], aType.MA)
            angle3 = problem.get_sym_of_attr(tri[2] + tri[0] + tri[1], aType.MA)
            premise = [problem.conditions.get_index(tri, cType.equilateral_triangle)]
            update = problem.define_equation(angle1 - 60, eType.theorem, premise, 32) or update
            update = problem.define_equation(angle2 - 60, eType.theorem, premise, 32) or update
            update = problem.define_equation(angle3 - 60, eType.theorem, premise, 32) or update

            i += rep.count_equilateral_triangle  # 6种表示

        return update

    @staticmethod
    def theorem_34_equilateral_triangle_property_side_equal(problem):
        pass

    @staticmethod
    def theorem_35_equilateral_triangle_judgment_angle_equal(problem):
        pass

    @staticmethod
    def theorem_36_equilateral_triangle_judgment_side_equal(problem):
        pass

    @staticmethod
    def theorem_37_equilateral_triangle_judgment_isos_and_angle60(problem):
        pass

    @staticmethod
    def theorem_38_intersect_property(problem):
        """
        相交 性质
        相交  ==>  对顶角相等、邻补角互补
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.intersect]):
            intersect = problem.conditions.items[cType.intersect][i]
            point, line1, line2 = intersect
            equations = []  # 对顶角相等、邻补角互补
            if len(set(point + line1 + line2)) == 4:
                if line2[1] == point:
                    angle1 = problem.get_sym_of_attr(line1[0] + line2[::-1], aType.MA)
                    angle2 = problem.get_sym_of_attr(line2 + line1[1], aType.MA)
                    equations.append(angle1 + angle2 - 180)
                elif line1[1] == point:
                    angle1 = problem.get_sym_of_attr(line2[1] + line1[::-1], aType.MA)
                    angle2 = problem.get_sym_of_attr(line1 + line2[0], aType.MA)
                    equations.append(angle1 + angle2 - 180)
                elif line2[0] == point:
                    angle1 = problem.get_sym_of_attr(line1[1] + line2, aType.MA)
                    angle2 = problem.get_sym_of_attr(line2[::-1] + line1[0], aType.MA)
                    equations.append(angle1 + angle2 - 180)
                elif line1[0] == point:
                    angle1 = problem.get_sym_of_attr(line2[0] + line1, aType.MA)
                    angle2 = problem.get_sym_of_attr(line1[::-1] + line2[1], aType.MA)
                    equations.append(angle1 + angle2 - 180)
            elif len(set(point + line1 + line2)) == 5:
                angle1 = problem.get_sym_of_attr(line1[0] + point + line2[0], aType.MA)
                angle2 = problem.get_sym_of_attr(line2[0] + point + line1[1], aType.MA)
                angle3 = problem.get_sym_of_attr(line1[1] + point + line2[1], aType.MA)
                angle4 = problem.get_sym_of_attr(line2[1] + point + line1[0], aType.MA)
                equations.append(angle1 + angle2 - 180)
                equations.append(angle3 + angle4 - 180)
                equations.append(angle1 - angle3)
                equations.append(angle2 - angle4)
            premise = [problem.conditions.get_index(intersect, cType.intersect)]
            for equation in equations:
                update = problem.define_equation(equation, eType.theorem, premise, 37) or update

            i += rep.count_intersect

        return update

    @staticmethod
    def theorem_39_parallel_property(problem):
        """
        平行 性质
        平行  ==>  内错角相等、同旁内角互补
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.parallel]):
            para = problem.conditions.items[cType.parallel][i]
            premise = [problem.conditions.get_index(para, cType.parallel)]
            points_ab = util.coll_points(para[0][0], para[0][1], problem)
            points_cd = util.coll_points(para[1][0], para[1][1], problem)

            ipsilateral = []  # 同旁内角
            alternate = []  # 内错角
            for p_ab in points_ab:
                for p_cd in points_cd:
                    if p_ab + p_cd in problem.conditions.items[cType.line]:
                        ipsilateral.append([points_cd[0] + p_cd + p_ab, p_cd + p_ab + points_ab[0]])
                        ipsilateral.append([points_ab[-1] + p_ab + p_cd, p_ab + p_cd + points_cd[-1]])
                        alternate.append([p_cd + p_ab + points_ab[0], p_ab + p_cd + points_cd[-1]])
                        alternate.append([points_cd[0] + p_cd + p_ab, points_ab[-1] + p_ab + p_cd])

            for ips in ipsilateral:  # 同旁内角互补
                if len(set(ips[0])) == 3 and len(set(ips[1])) == 3:
                    angle1 = problem.get_sym_of_attr(ips[0], aType.MA)
                    angle2 = problem.get_sym_of_attr(ips[1], aType.MA)
                    update = problem.define_equation(angle1 + angle2 - 180, eType.basic, premise, 38) or update
            for alt in alternate:  # 内错角相等
                if len(set(alt[0])) == 3 and len(set(alt[1])) == 3:
                    angle1 = problem.get_sym_of_attr(alt[0], aType.MA)
                    angle2 = problem.get_sym_of_attr(alt[1], aType.MA)
                    update = problem.define_equation(angle1 - angle2, eType.basic, premise, 38) or update
            i += rep.count_parallel

        return update

    @staticmethod
    def theorem_40_parallel_judgment(problem):
        pass

    @staticmethod
    def theorem_41_perpendicular_judgment(problem):
        """
        垂直 判定
        角为90°  ==>  垂直
        """
        pass

    @staticmethod
    def theorem_42_parallel_perpendicular_combination(problem):
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
    def theorem_43_midpoint_judgment(problem):
        """
        中点 判定
        """
        pass

    @staticmethod
    def theorem_44_perpendicular_bisector_property_distance_equal(problem):
        pass

    @staticmethod
    def theorem_45_perpendicular_bisector_judgment(problem):
        pass

    @staticmethod
    def theorem_46_bisector_property_line_ratio(problem):
        """
        角平分线 性质
        角平分线AD  ==>  BD/DC=AB/AC
        """
        update = False
        for bisector in problem.conditions.items[cType.bisector]:
            line, tri = bisector
            line11 = problem.get_sym_of_attr(tri[1] + line[1], aType.LL)
            line12 = problem.get_sym_of_attr(line[1] + tri[2], aType.LL)
            line21 = problem.get_sym_of_attr(tri[0] + tri[1], aType.LL)
            line22 = problem.get_sym_of_attr(tri[2] + tri[0], aType.LL)
            update = problem.define_equation(line11 / line12 - line21 / line22, eType.theorem,
                                             [problem.conditions.get_index(bisector, cType.bisector)], 45) or update
        return update

    @staticmethod
    def theorem_47_bisector_property_angle_equal(problem):
        """
        角平分线 性质
        角平分线AD  ==>  ∠DAB=∠CAD
        """
        update = False
        for bisector in problem.conditions.items[cType.bisector]:
            line, tri = bisector
            premise = [problem.conditions.get_index(bisector, cType.bisector)]
            angle1 = problem.get_sym_of_attr(line[1] + tri[0:2], aType.MA)
            angle2 = problem.get_sym_of_attr(tri[2] + line, aType.MA)
            update = problem.define_equation(angle1 - angle2, eType.theorem, premise, 46) or update

        return update

    @staticmethod
    def theorem_48_bisector_judgment_angle_equal(problem):
        """
        角平分线 判定
        平分的两角相等  ==>  DA为△ABC的角平分线
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            for line in problem.conditions.items[cType.line]:
                if (line, tri) not in problem.conditions.items[cType.bisector] and \
                        util.is_inside_triangle(line, tri, problem):  # 是三角形的内线
                    angle1 = problem.get_sym_of_attr(line[1] + tri[0:2], aType.MA)
                    angle2 = problem.get_sym_of_attr(tri[2] + tri[0] + line[1], aType.MA)
                    m = problem.get_sym_of_attr("1", aType.M)
                    result, premise = problem.solve_equations(m, m - angle2 + angle1)
                    if result is not None and abs(result) < 0.01:  # 且平分的两个角相等
                        update = problem.define_bisector((line, tri), premise, 47) or update
        return update

    @staticmethod
    def theorem_49_median_property(problem):
        pass

    @staticmethod
    def theorem_50_median_judgment(problem):
        pass

    @staticmethod
    def theorem_51_altitude_property(problem):
        pass

    @staticmethod
    def theorem_52_altitude_judgment(problem):
        """
        高 判定
        垂直于底边  ==>  高
        """
        update = False
        for perpendicular in problem.conditions.items[cType.perpendicular]:
            point, line1, line2 = perpendicular
            premise = [problem.conditions.get_index(perpendicular, cType.perpendicular)]
            is_altitude = []
            if len(set(point + line1 + line2)) == 3 and line1[0] == line2[1]:
                is_altitude.append((line2, line2 + line1[1]))
            elif len(set(point + line1 + line2)) == 4 and line2[1] == point:
                is_altitude.append((line2, line2[0] + line1[0] + line2[1]))
                is_altitude.append((line2, line2 + line1[1]))
                is_altitude.append((line2, line2[0] + line1))
            elif len(set(point + line1 + line2)) == 5:
                is_altitude.append((line2[0] + point, line2[0] + line1[0] + point))
                is_altitude.append((line2[0] + point, line2[0] + point + line1[1]))
                is_altitude.append((line2[0] + point, line2[0] + line1))
                is_altitude.append((line2[1] + point, line2[1] + point + line1[0]))
                is_altitude.append((line2[1] + point, line2[1] + line1[1] + point))
                is_altitude.append((line2[1] + point, line2[1] + line1[::-1]))

            for is_al in is_altitude:
                if is_al[1] in problem.conditions.items[cType.triangle]:
                    result_premise = premise + [problem.conditions.get_index(is_al[1], cType.triangle)]
                    update = problem.define_is_altitude(is_al, result_premise, 51) or update
        return update

    @staticmethod
    def theorem_53_neutrality_property_similar(problem):
        pass

    @staticmethod
    def theorem_54_neutrality_property_angle_equal(problem):
        pass

    @staticmethod
    def theorem_55_neutrality_property_line_ratio(problem):
        """
        中位线 性质
        中位线  ==>  两侧对边成比例
        """
        update = False
        for neutrality in problem.conditions.items[cType.neutrality]:
            line, tri = neutrality
            line11 = problem.get_sym_of_attr(tri[0] + line[0], aType.LL)
            line12 = problem.get_sym_of_attr(line[0] + tri[1], aType.LL)
            line21 = problem.get_sym_of_attr(tri[0] + line[1], aType.LL)
            line22 = problem.get_sym_of_attr(line[1] + tri[2], aType.LL)
            update = problem.define_equation(line11 / line12 - line21 / line22, eType.theorem,
                                             [problem.conditions.get_index(neutrality, cType.neutrality)], 55) or update
        return update

    @staticmethod
    def theorem_56_neutrality_judgment(problem):
        """
        中位线 判定
        △ABC、DE//BC  ==>  DE为△ABC的中位线
        """
        update = False
        for tri in problem.conditions.items[cType.triangle]:
            for parallel in problem.conditions.items[cType.parallel]:
                if tri[1:3] == parallel[1] and\
                        util.is_collinear(tri[0], parallel[0][0], tri[1], problem) and \
                        util.is_collinear(tri[0], parallel[0][1], tri[2], problem):
                    premise = [problem.conditions.get_index(tri, cType.triangle),
                               problem.conditions.get_index(parallel, cType.parallel)]
                    update = problem.define_neutrality((parallel[0], tri), premise, 56) or update
        return update

    @staticmethod
    def theorem_57_circumcenter_property_line_equal(problem):
        pass

    @staticmethod
    def theorem_58_circumcenter_property_intersect(problem):
        pass

    @staticmethod
    def theorem_59_circumcenter_judgment(problem):
        pass

    @staticmethod
    def theorem_60_incenter_property_line_equal(problem):
        pass

    @staticmethod
    def theorem_61_incenter_property_intersect(problem):
        """
        内心 性质
        内心  ==>  过内心的三角形的内线是三角形的 角平分线
        """
        update = False
        for incenter in problem.conditions.items[cType.incenter]:
            point, tri = incenter
            premise = [problem.conditions.get_index(incenter, cType.incenter)]

            for line in problem.conditions.items[cType.line]:  # 推导关系
                if util.is_inside_triangle(line, tri, problem) and \
                        util.is_collinear(line[0], point, line[1], problem):
                    update = problem.define_bisector((line, tri), premise, 59) or update

            for i in range(3):  # 角相等 数量关系
                if point + tri[i] in problem.conditions.items[cType.line]:
                    angle1 = problem.get_sym_of_attr(point + tri[0] + tri[1], aType.MA)
                    angle2 = problem.get_sym_of_attr(tri[2] + tri[0] + point, aType.MA)
                    update = problem.define_equation(angle1 - angle2, eType.basic, premise, 59) or update

        return update

    @staticmethod
    def theorem_62_incenter_property_judgment(problem):
        pass

    @staticmethod
    def theorem_63_centroid_property_line_equal(problem):
        """
        内心 性质
        内心  ==>  过内心的三角形的内线是三角形的 角平分线
        """
        update = False
        for centroid in problem.conditions.items[cType.centroid]:
            point, tri = centroid
            for line in problem.conditions.items[cType.line]:
                if util.is_inside_triangle(line, tri, problem) and \
                        util.is_collinear(line[0], point, line[1], problem):
                    premise = [problem.conditions.get_index(centroid, cType.centroid)]
                    a = problem.get_sym_of_attr(line[0] + point, aType.LL)
                    b = problem.get_sym_of_attr(point + line[1], aType.LL)
                    update = problem.define_equation(a - 2 * b, eType.theorem, premise, 60) or update
        return update

    @staticmethod
    def theorem_64_centroid_property_intersect(problem):
        """
        内心 性质
        内心  ==>  过内心的三角形的内线是三角形的 角平分线
        """
        update = False
        for centroid in problem.conditions.items[cType.centroid]:
            point, tri = centroid
            for line in problem.conditions.items[cType.line]:
                if util.is_inside_triangle(line, tri, problem) and \
                        util.is_collinear(line[0], point, line[1], problem):
                    premise = [problem.conditions.get_index(centroid, cType.centroid)]
                    update = problem.define_median((line, tri), premise, 61) or update
        return update

    @staticmethod
    def theorem_65_centroid_property_judgment(problem):
        pass

    @staticmethod
    def theorem_66_orthocenter_property_line_equal(problem):
        pass

    @staticmethod
    def theorem_67_orthocenter_property_intersect(problem):
        pass
    @staticmethod
    def theorem_68_orthocenter_property_judgment(problem):
        pass

    @staticmethod
    def theorem_69_congruent_property_line_equal(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应边相等
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
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3], aType.LL)
                update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 64) or update

            i = i + rep.count_congruent  # 一个全等关系有6种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_congruent]):
            mirror_congruent = problem.conditions.items[cType.mirror_congruent][i]
            tri1 = mirror_congruent[0]  # 三角形
            tri2 = mirror_congruent[1]
            premise = [problem.conditions.get_index(mirror_congruent, cType.mirror_congruent)]  # 前提

            for j in range(3):
                # 对应边相等
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                update = problem.define_equation(l_1 - l_2, eType.theorem, premise, 64) or update

            i = i + rep.count_mirror_congruent  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_70_congruent_property_angle_equal(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  对应角相等
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.conditions.items[cType.congruent]):
            congruent = problem.conditions.items[cType.congruent][i]
            tri1 = congruent[0]  # 三角形
            tri2 = congruent[1]
            premise = [problem.conditions.get_index(congruent, cType.congruent)]  # 前提

            for j in range(3):
                # 对应角相等
                angle_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3], aType.MA)
                angle_2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3], aType.MA)
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 65) or update

            i = i + rep.count_congruent  # 一个全等关系有6种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_congruent]):
            mirror_congruent = problem.conditions.items[cType.mirror_congruent][i]
            tri1 = mirror_congruent[0]  # 三角形
            tri2 = mirror_congruent[1]
            premise = [problem.conditions.get_index(mirror_congruent, cType.mirror_congruent)]  # 前提

            for j in range(3):
                # 对应角相等
                angle_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3], aType.MA)
                angle_2 = problem.get_sym_of_attr(tri2[(4 - j) % 3] + tri2[(2 - j) % 3] + tri2[(3 - j) % 3], aType.MA)
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 65) or update

            i = i + rep.count_mirror_congruent  # 一个全等关系有3种表示

        return update

    @staticmethod
    def theorem_71_congruent_property_area_equal(problem):
        """
        全等三角形 性质
        两个三角形全等  ==>  面积相等
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.congruent]):
            congruent = problem.conditions.items[cType.congruent][i]
            a1 = problem.get_sym_of_attr(congruent[0], aType.AS)
            a2 = problem.get_sym_of_attr(congruent[1], aType.AS)
            premise = [problem.conditions.get_index(congruent, cType.congruent)]
            update = problem.define_equation(a1 - a2, eType.theorem, premise, 66) or update
            i = i + rep.count_congruent

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_congruent]):
            mirror_congruent = problem.conditions.items[cType.mirror_congruent][i]
            a1 = problem.get_sym_of_attr(mirror_congruent[0], aType.AS)
            a2 = problem.get_sym_of_attr(mirror_congruent[1], aType.AS)
            premise = [problem.conditions.get_index(mirror_congruent, cType.mirror_congruent)]
            update = problem.define_equation(a1 - a2, eType.theorem, premise, 66) or update
            i = i + rep.count_mirror_congruent

        return update

    @staticmethod
    def theorem_72_congruent_judgment_sss(problem):
        pass

    @staticmethod
    def theorem_73_congruent_judgment_sas(problem):
        """
        全等三角形 判定：SAS
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri1 = problem.conditions.items[cType.triangle][i]
            j = i + rep.count_triangle
            while j < len(problem.conditions.items[cType.triangle]):
                tri2 = problem.conditions.items[cType.triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.congruent]:  # 判断全等
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]
                        m = problem.get_sym_of_attr("1", aType.M)

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        s12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a2 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        s22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.LL)

                        result, eq_premise = problem.solve_equations(m, m - s11 + s12)
                        if result is not None and abs(result) < 0.01:  # s1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - a1 + a2)
                        if result is not None and abs(result) < 0.01:  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - s21 + s22)
                        if result is not None and abs(result) < 0.01:  # s2相等
                            premise += eq_premise
                            update = problem.define_congruent((tri1, tri2), premise, 68) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_congruent]:  # 判断镜像全等
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]
                        m = problem.get_sym_of_attr("1", aType.M)

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        m_s12 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        m_s22 = problem.get_sym_of_attr(tri2[2 - k] + tri2[(4 - k) % 3], aType.LL)

                        result, eq_premise = problem.solve_equations(m, m - s11 + m_s12)

                        if result is not None and abs(result) < 0.01:  # s1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - a1 + m_a2)
                        if result is not None and abs(result) < 0.01:  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - s21 + m_s22)
                        if result is not None and abs(result) < 0.01:  # s2相等
                            premise += eq_premise
                            update = problem.define_mirror_congruent((tri1, tri2), premise, 68) or update
                            break

                j = j + 1
            i = i + rep.count_triangle

        return update

    @staticmethod
    def theorem_74_congruent_judgment_asa(problem):
        pass

    @staticmethod
    def theorem_75_congruent_judgment_aas(problem):
        pass

    @staticmethod
    def theorem_76_congruent_judgment_hl(problem):
        """
        全等三角形 判定：HL
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.right_triangle]):
            tri1 = problem.conditions.items[cType.right_triangle][i]
            j = i + rep.count_right_triangle
            while j < len(problem.conditions.items[cType.right_triangle]):
                tri2 = problem.conditions.items[cType.right_triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.congruent]:  # 判断全等
                    premise = [problem.conditions.get_index(tri1, cType.right_triangle),
                               problem.conditions.get_index(tri2, cType.right_triangle)]
                    m = problem.get_sym_of_attr("1", aType.M)

                    h11 = problem.get_sym_of_attr(tri1[0:2], aType.LL)
                    h12 = problem.get_sym_of_attr(tri2[0:2], aType.LL)
                    l1 = problem.get_sym_of_attr(tri1[0] + tri1[2], aType.LL)
                    l2 = problem.get_sym_of_attr(tri2[0] + tri2[2], aType.LL)
                    h21 = problem.get_sym_of_attr(tri1[1:3], aType.LL)
                    h22 = problem.get_sym_of_attr(tri2[1:3], aType.LL)

                    result, eq_premise = problem.solve_equations(m, m - l1 + l2)
                    if result is not None and abs(result) < 0.01:  # 斜边相等
                        premise += eq_premise
                    else:
                        continue

                    result, eq_premise = problem.solve_equations(m, m - h11 + h12)
                    if result is not None and abs(result) < 0.01:  # 直角边相等
                        update = problem.define_congruent((tri1, tri2), premise + eq_premise, 72) or update

                    result, eq_premise = problem.solve_equations(m, m - h21 + h22)
                    if result is not None and abs(result) < 0.01:  # 另一组直角边相等
                        update = problem.define_congruent((tri1, tri2), premise + eq_premise, 72) or update

                if (tri1, tri2) not in problem.conditions.items[cType.congruent]:  # 判断镜像全等
                    premise = [problem.conditions.get_index(tri1, cType.right_triangle),
                               problem.conditions.get_index(tri2, cType.right_triangle)]
                    m = problem.get_sym_of_attr("1", aType.M)

                    h11 = problem.get_sym_of_attr(tri1[0:2], aType.LL)
                    h12 = problem.get_sym_of_attr(tri2[1:3], aType.LL)
                    l1 = problem.get_sym_of_attr(tri1[0] + tri1[2], aType.LL)
                    l2 = problem.get_sym_of_attr(tri2[0] + tri2[2], aType.LL)
                    h21 = problem.get_sym_of_attr(tri1[1:3], aType.LL)
                    h22 = problem.get_sym_of_attr(tri2[0:2], aType.LL)

                    result, eq_premise = problem.solve_equations(m, m - l1 + l2)

                    if result is not None and abs(result) < 0.01:  # 斜边相等
                        premise += eq_premise
                    else:
                        continue

                    result, eq_premise = problem.solve_equations(m, m - h11 + h12)
                    if result is not None and abs(result) < 0.01:  # 直角边相等
                        update = problem.define_mirror_congruent((tri1, tri2), premise + eq_premise, 72) or update

                    result, eq_premise = problem.solve_equations(m, m - h21 + h22)
                    if result is not None and abs(result) < 0.01:  # 另一组直角边相等
                        update = problem.define_mirror_congruent((tri1, tri2), premise + eq_premise, 72) or update

                j = j + 1
            i = i + rep.count_right_triangle

    @staticmethod
    def theorem_77_similar_property_angle_equal(problem):
        """
        相似三角形 性质
        相似△  ==>  对应角相等
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.conditions.items[cType.similar]):
            similar = problem.conditions.items[cType.similar][i]
            tri1 = similar[0]  # 三角形
            tri2 = similar[1]
            for j in range(3):
                # 对应角相等
                angle_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3], aType.MA)
                angle_2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3] + tri2[(j + 2) % 3], aType.MA)
                premise = [problem.conditions.get_index(similar, cType.similar)]
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 72) or update

            i = i + rep.count_similar  # 一个全等关系有6种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            for j in range(3):
                # 对应角相等
                angle_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3] + tri1[(j + 2) % 3], aType.MA)
                angle_2 = problem.get_sym_of_attr(tri2[(4 - j) % 3] + tri2[(2 - j) % 3] + tri2[(3 - j) % 3], aType.MA)
                premise = [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
                update = problem.define_equation(angle_1 - angle_2, eType.theorem, premise, 72) or update

            i = i + rep.count_mirror_similar  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_78_similar_property_line_ratio(problem):
        """
        相似三角形 性质
        相似△  ==>  对应边成比例
        """
        update = False  # 存储应用定理是否更新了条件

        i = 0
        while i < len(problem.conditions.items[cType.similar]):
            similar = problem.conditions.items[cType.similar][i]
            tri1 = similar[0]  # 三角形
            tri2 = similar[1]
            ratios = []  # 对应边的比
            for j in range(3):
                # 对应边的比
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3], aType.LL)
                ratios.append(l_1 / l_2)
            premise = [problem.conditions.get_index(similar, cType.similar)]
            update = problem.define_equation(ratios[0] - ratios[1], eType.theorem, premise, 73) or update  # 对应边的比值相等
            update = problem.define_equation(ratios[1] - ratios[2], eType.theorem, premise, 73) or update

            i = i + rep.count_similar  # 一个全等关系有3种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            ratios = []  # 对应边的比
            for j in range(3):
                # 对应边的比
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                ratios.append(l_1 / l_2)
            premise = [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
            update = problem.define_equation(ratios[0] - ratios[1], eType.theorem, premise, 73) or update  # 对应边的比值相等
            update = problem.define_equation(ratios[1] - ratios[2], eType.theorem, premise, 73) or update

            i = i + rep.count_mirror_similar  # 一个全等关系有3种表示

        return update

    @staticmethod
    def theorem_79_similar_property_perimeter_ratio(problem):
        """
        相似三角形 性质
        相似△  ==>  周长成比例
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.similar]):
            similar = problem.conditions.items[cType.similar][i]
            tri1 = similar[0]  # 三角形
            tri2 = similar[1]
            for j in range(3):
                # 对应边的比
                l1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3], aType.LL)
                m = problem.get_sym_of_attr("1", aType.M)
                result, premise = problem.solve_equations(m, m - l1 / l2)
                if result is not None:
                    p1 = problem.get_sym_of_attr(tri1, aType.PT)
                    p2 = problem.get_sym_of_attr(tri2, aType.PT)
                    premise += [problem.conditions.get_index(similar, cType.similar)]
                    update = problem.define_equation(p1 / p2 - result, eType.theorem, premise, 74) or update
                    break
            i = i + rep.count_similar  # 一个全等关系有3种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            for j in range(3):
                # 对应边的比
                l1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                m = problem.get_sym_of_attr("1", aType.M)
                result, premise = problem.solve_equations(m, m - l1 / l2)
                if result is not None:
                    p1 = problem.get_sym_of_attr(tri1, aType.PT)
                    p2 = problem.get_sym_of_attr(tri2, aType.PT)
                    premise += [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
                    update = problem.define_equation(p1 / p2 - result, eType.theorem, premise, 74) or update
                    break

            i = i + rep.count_mirror_similar  # 一个全等关系有3种表示

        return update

    @staticmethod
    def theorem_80_similar_property_area_square_ratio(problem):
        pass

    @staticmethod
    def theorem_81_similar_judgment_sss(problem):
        pass

    @staticmethod
    def theorem_82_similar_judgment_sas(problem):
        """
        相似三角形 判定：SAS
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri1 = problem.conditions.items[cType.triangle][i]
            j = i + rep.count_triangle
            while j < len(problem.conditions.items[cType.triangle]):
                tri2 = problem.conditions.items[cType.triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.similar]:  # 判断相似
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]
                        m = problem.get_sym_of_attr("1", aType.M)

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        s12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a2 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        s22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.LL)

                        result, eq_premise = problem.solve_equations(m, m - a1 + a2)
                        if result is not None and abs(result) < 0.01:  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - s11 / s12 + s21 / s22)
                        if result is not None and abs(result) < 0.01:  # 两边 成比例
                            premise += eq_premise
                            update = problem.define_similar((tri1, tri2), premise, 77) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_similar]:  # 判断镜像相似
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]
                        m = problem.get_sym_of_attr("1", aType.M)

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        m_s12 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        m_s22 = problem.get_sym_of_attr(tri2[2 - k] + tri2[(4 - k) % 3], aType.LL)

                        result, eq_premise = problem.solve_equations(m, m - a1 + m_a2)
                        if result is not None and abs(result) < 0.01:  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_equations(m, m - s11 / m_s12 + s21 / m_s22)
                        if result is not None and abs(result) < 0.01:  # 两边 成比例
                            premise += eq_premise
                            update = problem.define_mirror_similar((tri1, tri2), premise, 77) or update
                            break

                j = j + 1
            i = i + rep.count_triangle

        return update

    @staticmethod
    def theorem_83_similar_judgment_aa(problem):
        """
        相似三角形 判定：AA
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri1 = problem.conditions.items[cType.triangle][i]
            j = i + rep.count_triangle
            while j < len(problem.conditions.items[cType.triangle]):
                tri2 = problem.conditions.items[cType.triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.similar]:  # 相似
                    equal_count = 0
                    premise = [problem.conditions.get_index(tri1, cType.triangle),
                               problem.conditions.get_index(tri2, cType.triangle)]
                    for k in range(3):  # 对应角相等
                        angle_1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        angle_2 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)

                        m = problem.get_sym_of_attr("1", aType.M)
                        result, eq_premise = problem.solve_equations(m, m - angle_1 + angle_2)
                        if result is not None and abs(result) < 0.01:
                            equal_count += 1
                            premise += eq_premise
                        if equal_count >= 2:
                            update = problem.define_similar((tri1, tri2), premise, 78) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_similar]:  # 镜像相似
                    m_equal_count = 0
                    m_premise = [problem.conditions.get_index(tri1, cType.triangle),
                                 problem.conditions.get_index(tri2, cType.triangle)]
                    for k in range(3):  # 对应角相等
                        angle_1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_angle_2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[(5 - k) % 3] + tri2[(3 - k) % 3],
                                                            aType.MA)

                        m = problem.get_sym_of_attr("1", aType.M)
                        result, eq_premise = problem.solve_equations(m, m - angle_1 + m_angle_2)
                        if result is not None and abs(result) < 0.01:
                            m_equal_count += 1
                            m_premise += eq_premise
                        if m_equal_count >= 2:
                            update = problem.define_mirror_similar((tri1, tri2), m_premise, 78) or update
                            break

                j = j + 1
            i = i + rep.count_triangle

        return update

    @staticmethod
    def theorem_84_triangle_perimeter_formula(problem):
        """
        三角形周长公式：三边和
        """
        update = False  # 存储应用定理是否更新了条件
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):  # 三角形
            tri = problem.conditions.items[cType.triangle][i]
            p = problem.get_sym_of_attr(tri, aType.PT)
            a = problem.get_sym_of_attr(tri[0:2], aType.LL)
            b = problem.get_sym_of_attr(tri[1:3], aType.LL)
            c = problem.get_sym_of_attr(tri[2] + tri[0], aType.LL)
            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(p - a - b - c, eType.theorem, premise, 79) or update
            i = i + rep.count_triangle

        return update

    @staticmethod
    def theorem_85_triangle_area_formula_common(problem):
        """
        三角形面积公式：底乘高
        """
        update = False
        for is_altitude in problem.conditions.items[cType.is_altitude]:
            altitude, tri = is_altitude
            len_altitude = problem.get_sym_of_attr(altitude, aType.LL)  # 高
            len_base = problem.get_sym_of_attr(tri[1:3], aType.LL)  # 底
            area_tri = problem.get_sym_of_attr(tri, aType.AS)
            premise = [problem.conditions.get_index(is_altitude, cType.is_altitude)]
            update = problem.define_equation(area_tri - 0.5 * len_base * len_altitude, eType.theorem,
                                             premise, 80) or update
        return update

    @staticmethod
    def theorem_86_triangle_area_formula_heron(problem):
        """
        面积 海伦公式
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            a = problem.get_sym_of_attr(tri, aType.AS)
            s1 = problem.get_sym_of_attr(tri[0:2], aType.LL)
            s2 = problem.get_sym_of_attr(tri[1:3], aType.LL)
            s3 = problem.get_sym_of_attr(tri[2] + tri[0], aType.LL)
            p = (s1 + s2 + s3) / 2
            equation = a - sqrt(p * (p - s1) * (p - s2) * (p - s3))
            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(equation, eType.theorem, premise, 81) or update
            i += rep.count_triangle
        return update

    @staticmethod
    def theorem_87_triangle_area_formula_sine(problem):
        """
        三角形面积 = 1/2 * a * b * sinC
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            for j in range(3):
                s1 = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3], aType.LL)
                angle = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3] + tri[(j + 2) % 3], aType.MA)
                s2 = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3], aType.LL)
                a = problem.get_sym_of_attr(tri, aType.AS)
                premise = [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(a - 0.5 * s1 * s2 * sin(angle), eType.theorem, premise, 82) or update

            i += rep.count_triangle
        return update

    @staticmethod
    def theorem_88_sine(problem):
        """
        正弦定理
        """
        # update = False
        # i = 0
        # while i < len(problem.conditions.items[cType.triangle]):
        #     tri = problem.conditions.items[cType.triangle][i]
        #     ratios_unit = []
        #     known_count = [0, 0, 0]  # 记录方程中已知变量的个数, =3方程才有意义
        #     for j in range(3):
        #         line = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3], aType.LL)
        #         angle = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3], aType.MA)
        #         if problem.value_of_sym[line] is not None:
        #             known_count[j] += 1
        #         if problem.value_of_sym[angle] is not None:
        #             known_count[j] += 1
        #         ratios_unit.append(line / sin(angle * pi / 180))
        #
        #     for j in range(3):
        #         if known_count[j] + known_count[(j + 1) % 3] == 3:
        #             equation = ratios_unit[j] - ratios_unit[(j + 1) % 3]
        #             premise = [problem.conditions.get_index(tri, cType.triangle)]
        #             update = problem.define_equation(equation, eType.theorem, premise, 80) or update
        #
        #     i = i + rep.count_triangle  # 一个三角形多种表示
        #
        # return update
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            ratios_unit = []
            for j in range(3):
                line = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3], aType.LL)
                angle = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3], aType.MA)
                ratios_unit.append(line / sin(angle * pi / 180))

            premise = [problem.conditions.get_index(tri, cType.triangle)]
            update = problem.define_equation(ratios_unit[0] - ratios_unit[1], eType.theorem, premise, 83) or update
            update = problem.define_equation(ratios_unit[1] - ratios_unit[2], eType.theorem, premise, 83) or update

            i = i + rep.count_triangle  # 一个三角形多种表示

        return update

    @staticmethod
    def theorem_89_cosine(problem):
        """
        余弦定理
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            for j in range(3):
                a = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3], aType.LL)
                b = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3], aType.LL)
                c = problem.get_sym_of_attr(tri[(j + 2) % 3] + tri[j], aType.LL)
                angle = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3], aType.MA)
                equation = a ** 2 - b ** 2 - c ** 2 + 2 * b * c * cos(angle * pi / 180)
                premise = [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(equation, eType.theorem, premise, 84) or update

            i = i + rep.count_triangle  # 一个三角形多种表示

        return update
