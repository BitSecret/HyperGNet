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
        update = False  # 是否更新了条件
        shape_update = True
        traversed = []  # 记录已经计算过的，避免重复计算
        while shape_update:
            shape_update = False
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

                        if 2 < len(new_shape) == len(set(new_shape)):  # 是图形且没有环
                            premise = [problem.conditions.get_index(shape1, cType.shape),
                                       problem.conditions.get_index(shape2, cType.shape)]

                            a1 = problem.get_sym_of_attr(shape1, aType.AS)
                            a2 = problem.get_sym_of_attr(shape2, aType.AS)
                            a3 = problem.get_sym_of_attr(new_shape, aType.AS)
                            eq_update = problem.define_equation(a1 + a2 - a3, eType.basic, premise, 1)
                            update = eq_update or update
                            shape_update = eq_update or shape_update

        return update

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
                        update = problem.define_equation(sym1 + sym2 - sym3, eType.basic, premise, 2) or update  # 相加关系

        return update

    @staticmethod
    def nous_3_angle_addition(problem):
        """
        拼图法构造角之间的相加关系
        Angle(ABC), Angle(CBD) ==> Angle(ABD), da_abc + da_cbd - da_abd = 0
        注：之所以搞这么麻烦是因为会发生组合爆炸
        """
        update = False
        init_shape = []    # 所有的初始shape
        for shape in problem.conditions.items[cType.shape]:  # 得到构图图形
            if problem.conditions.get_premise(shape, cType.shape)[0] == -1:
                init_shape.append(shape)

        init_angle = []    # 初始shape中的所有角
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
            update = problem.define_equation(equation, eType.basic, premise, 21) or update

            i += rep.count_triangle  # 跳过冗余表示
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

            result, premise = problem.solve_target(line1 - line2)  # 等边对等角
            if result is not None and util.equal(result, 0):
                premise += [problem.conditions.get_index(tri, cType.triangle)]
                update = problem.define_equation(angle1 - angle2, eType.basic, premise, 22) or update
            result, premise = problem.solve_target(angle1 - angle2)  # 等角对等边
            if result is not None and util.equal(result, 0):
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
                                             eType.complex,
                                             [problem.conditions.get_index(rt, cType.right_triangle)],
                                             23) or update
            i += rep.count_right_triangle

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
            update = problem.define_perpendicular((rt[0], rt[0:2], rt[2] + rt[1]), premise, 24) or update  # 边垂直

        return update

    @staticmethod
    def theorem_25_right_triangle_property_special_rt(problem):
        """
        直角三角形性质
        RT△  ==>  30°、60°、45°特殊直角三角形边的比
        """
        update = False
        for rt in problem.conditions.items[cType.right_triangle]:
            premise = [problem.conditions.get_index(rt, cType.right_triangle)]

            angle = problem.get_sym_of_attr(rt[1] + rt[2] + rt[0], aType.MA)
            result, eq_premise = problem.solve_target(angle)
            if result is not None:
                a = problem.get_sym_of_attr(rt[0:2], aType.LL)
                b = problem.get_sym_of_attr(rt[1:3], aType.LL)
                c = problem.get_sym_of_attr(rt[2] + rt[0], aType.LL)  # 斜边
                equations = []
                if util.equal(result, 30):    # angle是30°角
                    equations.append(c - 2 * a)
                    equations.append(c - 2 / sqrt(3) * b)
                    equations.append(b - sqrt(3) * a)
                elif util.equal(result, 60):    # angle是60°角
                    equations.append(c - 2 * b)
                    equations.append(c - 2 / sqrt(3) * a)
                    equations.append(a - sqrt(3) * b)
                elif util.equal(result, 45):    # angle是45°角
                    equations.append(c - sqrt(2) * a)
                    equations.append(c - sqrt(2) * b)
                    equations.append(a - b)
                for equation in equations:
                    update = problem.define_equation(equation, eType.complex, premise + eq_premise, 25) or update

            angle = problem.get_sym_of_attr(rt[2] + rt[0] + rt[1], aType.MA)    # 看看另一个角能不能求解
            result, eq_premise = problem.solve_target(angle)
            if result is not None:
                a = problem.get_sym_of_attr(rt[0:2], aType.LL)
                b = problem.get_sym_of_attr(rt[1:3], aType.LL)
                c = problem.get_sym_of_attr(rt[2] + rt[0], aType.LL)  # 斜边
                equations = []
                if util.equal(result, 30):  # angle是30°角
                    equations.append(c - 2 * b)
                    equations.append(c - 2 / sqrt(3) * a)
                    equations.append(a - sqrt(3) * b)
                elif util.equal(result, 60):  # angle是60°角
                    equations.append(c - 2 * a)
                    equations.append(c - 2 / sqrt(3) * b)
                    equations.append(b - sqrt(3) * a)
                elif util.equal(result, 45):  # angle是45°角
                    equations.append(c - sqrt(2) * a)
                    equations.append(c - sqrt(2) * b)
                    equations.append(a - b)
                for equation in equations:
                    update = problem.define_equation(equation, eType.complex, premise + eq_premise, 25) or update

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

                result, premise = problem.solve_target(a ** 2 + b ** 2 - c ** 2)
                if result is not None and util.equal(result, 0):
                    update = problem.define_right_triangle(tri, premise, 26) or update
                result, premise = problem.solve_target(a ** 2 - b ** 2 + c ** 2)
                if result is not None and util.equal(result, 0):
                    update = problem.define_right_triangle(tri[1] + tri[0] + tri[2], premise, 26) or update
                result, premise = problem.solve_target(-a ** 2 + b ** 2 + c ** 2)
                if result is not None and util.equal(result, 0):
                    update = problem.define_right_triangle(tri[1] + tri[2] + tri[0], premise, 26) or update

            i += rep.count_triangle

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
            result, premise = problem.solve_target(angle)
            if result is not None and util.equal(result, 90):
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
            update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 28) or update

            i += rep.count_isosceles_triangle

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
            update = problem.define_equation(l_1 - l_2, eType.basic, premise, 29) or update

            i += rep.count_isosceles_triangle

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
            update = problem.define_is_altitude((lines[i], triangles[i]), premises[i], 30) or update
            update = problem.define_median((lines[i], triangles[i]), premises[i], 30) or update
            update = problem.define_bisector((lines[i], triangles[i]), premises[i], 30) or update
            update = problem.define_perpendicular_bisector((lines[i][1], triangles[i][1:3], lines[i]),
                                                           premises[i], 30) or update
            i += 1

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
                result, eq_premise = problem.solve_target(a1 - a2)
                if result is not None and util.equal(result, 0):
                    premise = [problem.conditions.get_index(tri, cType.triangle)] + eq_premise
                    update = problem.define_isosceles_triangle(tri, premise, 31) or update
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
                result, eq_premise = problem.solve_target(s1 - s2)
                if result is not None and util.equal(result, 0):
                    premise = [problem.conditions.get_index(tri, cType.triangle)] + eq_premise
                    update = problem.define_isosceles_triangle(tri, premise, 32) or update
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
            update = problem.set_value_of_sym(angle1, 60, premise, 33) or update
            update = problem.set_value_of_sym(angle2, 60, premise, 33) or update
            update = problem.set_value_of_sym(angle3, 60, premise, 33) or update

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
                update = problem.define_equation(equation, eType.basic, premise, 38) or update

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
                    update = problem.define_equation(angle1 + angle2 - 180, eType.basic, premise, 39) or update
            for alt in alternate:  # 内错角相等
                if len(set(alt[0])) == 3 and len(set(alt[1])) == 3:
                    angle1 = problem.get_sym_of_attr(alt[0], aType.MA)
                    angle2 = problem.get_sym_of_attr(alt[1], aType.MA)
                    update = problem.define_equation(angle1 - angle2, eType.basic, premise, 39) or update
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
        平行、垂直的组合推导
        AB ⊥ CD, CD // EF  ==>  AB ⊥ EF
        """
        pass

    @staticmethod
    def theorem_43_midpoint_judgment(problem):
        """
        中点 判定
        """
        update = False
        for line1 in problem.conditions.items[cType.line]:
            for line2 in problem.conditions.items[cType.line]:
                if line1[1] == line2[0] and line1[0] != line2[1] and\
                        (line1[1], line1[0] + line2[1]) not in problem.conditions.items[cType.midpoint] and\
                        util.is_collinear(line1[0], line1[1], line2[1], problem):
                    l1 = problem.get_sym_of_attr(line1, aType.LL)
                    l2 = problem.get_sym_of_attr(line2, aType.LL)
                    result, premise = problem.solve_target(l1 - l2)
                    if result is not None and util.equal(result, 0):
                        premise.append(problem.conditions.get_index(line1, cType.line))
                        premise.append(problem.conditions.get_index(line2, cType.line))
                        update = problem.define_midpoint((line1[1], line1[0] + line2[1]), premise, 43) or update
        return update

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
            update = problem.define_equation(line11 * line22 - line12 * line21, eType.complex,
                                             [problem.conditions.get_index(bisector, cType.bisector)], 46) or update
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
            update = problem.define_equation(angle1 - angle2, eType.basic, premise, 47) or update

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
                    result, premise = problem.solve_target(angle2 - angle1)
                    if result is not None and util.equal(result, 0):  # 且平分的两个角相等
                        update = problem.define_bisector((line, tri), premise, 48) or update
        return update

    @staticmethod
    def theorem_49_median_property(problem):
        """
        中线 性质
        中线  ==>  与底边的交点是底边的中点
        """
        update = False
        for median in problem.conditions.items[cType.median]:
            line, tri = median
            premise = [problem.conditions.get_index[median, cType.median]]
            update = problem.define_midpoint((line[1], tri[1:3]), premise, 49) or update
        return update

    @staticmethod
    def theorem_50_median_judgment(problem):
        """
        中线 判定
        底边中点  ==>  中线
        """
        Theorem.theorem_43_midpoint_judgment(problem)    # 先寻找中点
        update = False
        for midpoint in problem.conditions.items[cType.midpoint]:
            point, line = midpoint
            for f_point in problem.conditions.items[cType.point]:
                if f_point + point in problem.conditions.items[cType.line] and\
                        point + line in problem.conditions.items[cType.triangle]:
                    premise = [problem.conditions.get_index[midpoint, cType.midpoint],
                               problem.conditions.get_index[f_point + point, cType.line],
                               problem.conditions.get_index[point + line, cType.triangle]]
                    update = problem.define_median((f_point + point, point + line), premise, 50) or update
        return update

    @staticmethod
    def theorem_51_altitude_property(problem):
        pass

    @staticmethod
    def theorem_52_altitude_judgment(problem):
        """
        高 判定
        垂直于底边  ==>  高
        """
        pass

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
            update = problem.define_equation(line11 * line22 - line12 * line21, eType.complex,
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
                    update = problem.define_bisector((line, tri), premise, 61) or update

            for i in range(3):  # 角相等 数量关系
                if point + tri[i] in problem.conditions.items[cType.line]:
                    angle1 = problem.get_sym_of_attr(point + tri[0] + tri[1], aType.MA)
                    angle2 = problem.get_sym_of_attr(tri[2] + tri[0] + point, aType.MA)
                    update = problem.define_equation(angle1 - angle2, eType.basic, premise, 61) or update

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
                    update = problem.define_equation(a - 2 * b, eType.complex, premise, 63) or update
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
                    update = problem.define_median((line, tri), premise, 64) or update
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
                update = problem.define_equation(l_1 - l_2, eType.basic, premise, 69) or update

            i += rep.count_congruent  # 一个全等关系有6种表示

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
                update = problem.define_equation(l_1 - l_2, eType.basic, premise, 69) or update

            i += rep.count_mirror_congruent  # 一个全等关系有6种表示

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
                update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 70) or update

            i += rep.count_congruent  # 一个全等关系有6种表示

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
                update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 70) or update

            i += rep.count_mirror_congruent  # 一个全等关系有3种表示

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
            update = problem.define_equation(a1 - a2, eType.basic, premise, 71) or update
            i += rep.count_congruent

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_congruent]):
            mirror_congruent = problem.conditions.items[cType.mirror_congruent][i]
            a1 = problem.get_sym_of_attr(mirror_congruent[0], aType.AS)
            a2 = problem.get_sym_of_attr(mirror_congruent[1], aType.AS)
            premise = [problem.conditions.get_index(mirror_congruent, cType.mirror_congruent)]
            update = problem.define_equation(a1 - a2, eType.basic, premise, 71) or update
            i += rep.count_mirror_congruent

        return update

    @staticmethod
    def theorem_72_congruent_judgment_sss(problem):
        """
        全等三角形 判定：SSS
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri1 = problem.conditions.items[cType.triangle][i]
            j = i + rep.count_triangle
            while j < len(problem.conditions.items[cType.triangle]):
                tri2 = problem.conditions.items[cType.triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.congruent]:  # 判断全等
                    premise = [problem.conditions.get_index(tri1, cType.triangle),
                               problem.conditions.get_index(tri2, cType.triangle)]

                    s1 = problem.get_sym_of_attr(tri1[0] + tri1[1], aType.LL)  # 第1组边
                    s2 = problem.get_sym_of_attr(tri2[0] + tri2[1], aType.LL)
                    result, eq_premise = problem.solve_target(s1 - s2)
                    if result is not None and util.equal(result, 0):
                        premise += eq_premise
                        s1 = problem.get_sym_of_attr(tri1[1] + tri1[2], aType.LL)  # 第2组边
                        s2 = problem.get_sym_of_attr(tri2[1] + tri2[2], aType.LL)
                        result, eq_premise = problem.solve_target(s1 - s2)
                        if result is not None and util.equal(result, 0):
                            premise += eq_premise
                            s1 = problem.get_sym_of_attr(tri1[2] + tri1[0], aType.LL)  # 第3组边
                            s2 = problem.get_sym_of_attr(tri2[2] + tri2[0], aType.LL)
                            result, eq_premise = problem.solve_target(s1 - s2)
                            if result is not None and util.equal(result, 0):
                                premise += eq_premise
                                update = problem.define_congruent((tri1, tri2), premise, 72) or update

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_congruent]:  # 判断镜像全等
                    premise = [problem.conditions.get_index(tri1, cType.triangle),
                               problem.conditions.get_index(tri2, cType.triangle)]

                    s1 = problem.get_sym_of_attr(tri1[0] + tri1[1], aType.LL)  # 第1组边
                    s2 = problem.get_sym_of_attr(tri2[0] + tri2[2], aType.LL)
                    result, eq_premise = problem.solve_target(s1 - s2)
                    if result is not None and util.equal(result, 0):
                        premise += eq_premise
                        s1 = problem.get_sym_of_attr(tri1[1] + tri1[2], aType.LL)  # 第2组边
                        s2 = problem.get_sym_of_attr(tri2[2] + tri2[1], aType.LL)
                        result, eq_premise = problem.solve_target(s1 - s2)
                        if result is not None and util.equal(result, 0):
                            premise += eq_premise
                            s1 = problem.get_sym_of_attr(tri1[2] + tri1[0], aType.LL)  # 第3组边
                            s2 = problem.get_sym_of_attr(tri2[1] + tri2[0], aType.LL)
                            result, eq_premise = problem.solve_target(s1 - s2)
                            if result is not None and util.equal(result, 0):
                                premise += eq_premise
                                update = problem.define_mirror_congruent((tri1, tri2), premise, 72) or update

                j += 1
            i += rep.count_triangle

        return update

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

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        s12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a2 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        s22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.LL)

                        result, eq_premise = problem.solve_target(s11 - s12)
                        if result is not None and util.equal(result, 0):  # s1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a1 - a2)
                        if result is not None and util.equal(result, 0):  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s21 - s22)
                        if result is not None and util.equal(result, 0):  # s2相等
                            premise += eq_premise
                            update = problem.define_congruent((tri1, tri2), premise, 73) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_congruent]:  # 判断镜像全等
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        m_s12 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        m_s22 = problem.get_sym_of_attr(tri2[2 - k] + tri2[(4 - k) % 3], aType.LL)

                        result, eq_premise = problem.solve_target(s11 - m_s12)

                        if result is not None and util.equal(result, 0):  # s1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a1 - m_a2)
                        if result is not None and util.equal(result, 0):  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s21 - m_s22)
                        if result is not None and util.equal(result, 0):  # s2相等
                            premise += eq_premise
                            update = problem.define_mirror_congruent((tri1, tri2), premise, 73) or update
                            break

                j += 1
            i += rep.count_triangle

        return update

    @staticmethod
    def theorem_74_congruent_judgment_asa(problem):
        """
        全等三角形 判定：ASA
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

                        a11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        s1 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        s2 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.LL)
                        a21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3] + tri1[k], aType.MA)
                        a22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3] + tri2[k], aType.MA)

                        result, eq_premise = problem.solve_target(s1 - s2)
                        if result is not None and util.equal(result, 0):  # s相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a11 - a12)
                        if result is not None and util.equal(result, 0):  # a1 相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a21 - a22)
                        if result is not None and util.equal(result, 0):  # a2 相等
                            premise += eq_premise
                            update = problem.define_congruent((tri1, tri2), premise, 74) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_congruent]:  # 判断镜像全等
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]

                        a11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a12 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        s1 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        m_s2 = problem.get_sym_of_attr(tri2[2 - k] + tri2[(4 - k) % 3], aType.LL)
                        a21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3] + tri1[k], aType.MA)
                        m_a22 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[(4 - k) % 3] + tri2[2 - k], aType.MA)

                        result, eq_premise = problem.solve_target(s1 - m_s2)
                        if result is not None and util.equal(result, 0):  # s相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a11 - m_a12)
                        if result is not None and util.equal(result, 0):  # a1 相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a21 - m_a22)
                        if result is not None and util.equal(result, 0):  # a2 相等
                            premise += eq_premise
                            update = problem.define_mirror_congruent((tri1, tri2), premise, 74) or update
                            break

                j += 1
            i += rep.count_triangle

        return update

    @staticmethod
    def theorem_75_congruent_judgment_aas(problem):
        """
        全等三角形 判定：AAS
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

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        s12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL)
                        a11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        a21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3] + tri1[k], aType.MA)
                        a22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3] + tri2[k], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 2) % 3] + tri1[k], aType.LL)
                        s22 = problem.get_sym_of_attr(tri2[(k + 2) % 3] + tri2[k], aType.LL)

                        result, eq_premise = problem.solve_target(a11 - a12)
                        if result is not None and util.equal(result, 0):  # a1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a21 - a22)
                        if result is not None and util.equal(result, 0):  # a2 相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s11 - s12)
                        if result is not None and util.equal(result, 0):  # 左侧临边
                            premise += eq_premise
                            update = problem.define_congruent((tri1, tri2), premise, 75) or update
                            break
                        result, eq_premise = problem.solve_target(s21 - s22)
                        if result is not None and util.equal(result, 0):  # 右侧临边
                            premise += eq_premise
                            update = problem.define_congruent((tri1, tri2), premise, 75) or update

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_congruent]:  # 判断镜像全等
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        m_s12 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL)
                        a11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a12 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        a21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3] + tri1[k], aType.MA)
                        m_a22 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[(4 - k) % 3] + tri2[2 - k], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 2) % 3] + tri1[k], aType.LL)
                        m_s22 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[(3 - k) % 3], aType.LL)

                        result, eq_premise = problem.solve_target(a11 - m_a12)
                        if result is not None and util.equal(result, 0):  # a1相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(a21 - m_a22)
                        if result is not None and util.equal(result, 0):  # a2 相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s11 - m_s12)
                        if result is not None and util.equal(result, 0):  # 左侧临边
                            premise += eq_premise
                            update = problem.define_mirror_congruent((tri1, tri2), premise, 75) or update
                            break
                        result, eq_premise = problem.solve_target(s21 - m_s22)
                        if result is not None and util.equal(result, 0):  # 右侧临边
                            premise += eq_premise
                            update = problem.define_mirror_congruent((tri1, tri2), premise, 75) or update

                j += 1
            i += rep.count_triangle

        return update

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

                    h11 = problem.get_sym_of_attr(tri1[0:2], aType.LL)
                    h12 = problem.get_sym_of_attr(tri2[0:2], aType.LL)
                    l1 = problem.get_sym_of_attr(tri1[0] + tri1[2], aType.LL)
                    l2 = problem.get_sym_of_attr(tri2[0] + tri2[2], aType.LL)
                    h21 = problem.get_sym_of_attr(tri1[1:3], aType.LL)
                    h22 = problem.get_sym_of_attr(tri2[1:3], aType.LL)

                    result, eq_premise = problem.solve_target(l1 - l2)
                    if result is not None and util.equal(result, 0):  # 斜边相等
                        premise += eq_premise
                    else:
                        continue

                    result, eq_premise = problem.solve_target(h11 - h12)
                    if result is not None and util.equal(result, 0):  # 直角边相等
                        update = problem.define_congruent((tri1, tri2), premise + eq_premise, 72) or update

                    result, eq_premise = problem.solve_target(h21 - h22)
                    if result is not None and util.equal(result, 0):  # 另一组直角边相等
                        update = problem.define_congruent((tri1, tri2), premise + eq_premise, 72) or update

                if (tri1, tri2) not in problem.conditions.items[cType.congruent]:  # 判断镜像全等
                    premise = [problem.conditions.get_index(tri1, cType.right_triangle),
                               problem.conditions.get_index(tri2, cType.right_triangle)]

                    h11 = problem.get_sym_of_attr(tri1[0:2], aType.LL)
                    h12 = problem.get_sym_of_attr(tri2[1:3], aType.LL)
                    l1 = problem.get_sym_of_attr(tri1[0] + tri1[2], aType.LL)
                    l2 = problem.get_sym_of_attr(tri2[0] + tri2[2], aType.LL)
                    h21 = problem.get_sym_of_attr(tri1[1:3], aType.LL)
                    h22 = problem.get_sym_of_attr(tri2[0:2], aType.LL)

                    result, eq_premise = problem.solve_target(l1 - l2)

                    if result is not None and util.equal(result, 0):  # 斜边相等
                        premise += eq_premise
                    else:
                        continue

                    result, eq_premise = problem.solve_target(h11 - h12)
                    if result is not None and util.equal(result, 0):  # 直角边相等
                        update = problem.define_mirror_congruent((tri1, tri2), premise + eq_premise, 72) or update

                    result, eq_premise = problem.solve_target(h21 - h22)
                    if result is not None and util.equal(result, 0):  # 另一组直角边相等
                        update = problem.define_mirror_congruent((tri1, tri2), premise + eq_premise, 72) or update

                j += 1
            i += rep.count_right_triangle

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
                update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 77) or update

            i += rep.count_similar  # 一个全等关系有6种表示

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
                update = problem.define_equation(angle_1 - angle_2, eType.basic, premise, 77) or update

            i += rep.count_mirror_similar  # 一个全等关系有6种表示

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
            ll = [[], []]  # 对应边的比
            for j in range(3):
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[j] + tri2[(j + 1) % 3], aType.LL)
                ll[0].append(l_1)
                ll[1].append(l_2)
            premise = [problem.conditions.get_index(similar, cType.similar)]
            update = problem.define_equation(ll[0][0] * ll[1][1] - ll[0][1] * ll[1][0],
                                             eType.complex, premise, 78) or update  # 对应边的比值相等
            update = problem.define_equation(ll[0][0] * ll[1][2] - ll[0][2] * ll[1][0],
                                             eType.complex, premise, 78) or update
            update = problem.define_equation(ll[0][2] * ll[1][1] - ll[0][1] * ll[1][2],
                                             eType.complex, premise, 78) or update

            i += rep.count_similar  # 一个全等关系有3种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            ll = [[], []]  # 对应边的比
            for j in range(3):
                l_1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l_2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                ll[0].append(l_1)
                ll[1].append(l_2)
            premise = [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
            update = problem.define_equation(ll[0][0] * ll[1][1] - ll[0][1] * ll[1][0],
                                             eType.complex, premise, 78) or update  # 对应边的比值相等
            update = problem.define_equation(ll[0][0] * ll[1][2] - ll[0][2] * ll[1][0],
                                             eType.complex, premise, 78) or update
            update = problem.define_equation(ll[0][2] * ll[1][1] - ll[0][1] * ll[1][2],
                                             eType.complex, premise, 78) or update

            i += rep.count_mirror_similar  # 一个全等关系有3种表示

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
                result, premise = problem.solve_target(l1 / l2)
                if result is not None:
                    p1 = problem.get_sym_of_attr(tri1, aType.PT)
                    p2 = problem.get_sym_of_attr(tri2, aType.PT)
                    premise += [problem.conditions.get_index(similar, cType.similar)]
                    update = problem.define_equation(p1 - p2 * result, eType.complex, premise, 79) or update
                    break
            i += rep.count_similar  # 一个全等关系有6种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            for j in range(3):
                # 对应边的比
                l1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                result, premise = problem.solve_target(l1 / l2)
                if result is not None:
                    p1 = problem.get_sym_of_attr(tri1, aType.PT)
                    p2 = problem.get_sym_of_attr(tri2, aType.PT)
                    premise += [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
                    update = problem.define_equation(p1 - p2 * result, eType.complex, premise, 79) or update
                    break

            i = i + rep.count_mirror_similar  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_80_similar_property_area_square_ratio(problem):
        """
        相似三角形 性质
        相似△  ==>  面积成乘方比
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
                result, premise = problem.solve_target(l1 / l2)
                if result is not None:
                    a1 = problem.get_sym_of_attr(tri1, aType.AT)
                    a2 = problem.get_sym_of_attr(tri2, aType.AT)
                    premise += [problem.conditions.get_index(similar, cType.similar)]
                    update = problem.define_equation(a1 - a2 * result, eType.complex, premise, 80) or update
                    break
            i += rep.count_similar  # 一个全等关系有6种表示

        i = 0  # 镜像
        while i < len(problem.conditions.items[cType.mirror_similar]):
            mirror_similar = problem.conditions.items[cType.mirror_similar][i]
            tri1 = mirror_similar[0]  # 三角形
            tri2 = mirror_similar[1]
            for j in range(3):
                # 对应边的比
                l1 = problem.get_sym_of_attr(tri1[j] + tri1[(j + 1) % 3], aType.LL)
                l2 = problem.get_sym_of_attr(tri2[(3 - j) % 3] + tri2[2 - j], aType.LL)
                result, premise = problem.solve_target(l1 / l2)
                if result is not None:
                    a1 = problem.get_sym_of_attr(tri1, aType.AT)
                    a2 = problem.get_sym_of_attr(tri2, aType.AT)
                    premise += [problem.conditions.get_index(mirror_similar, cType.mirror_similar)]
                    update = problem.define_equation(a1 - a2 * result, eType.complex, premise, 80) or update
                    break

            i += rep.count_mirror_similar  # 一个全等关系有6种表示

        return update

    @staticmethod
    def theorem_81_similar_judgment_sss(problem):
        """
        相似三角形 判定：SSS
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri1 = problem.conditions.items[cType.triangle][i]
            j = i + rep.count_triangle
            while j < len(problem.conditions.items[cType.triangle]):
                tri2 = problem.conditions.items[cType.triangle][j]

                if (tri1, tri2) not in problem.conditions.items[cType.similar]:  # 判断相似
                    premise = [problem.conditions.get_index(tri1, cType.triangle),
                               problem.conditions.get_index(tri2, cType.triangle)]
                    ll = [[], []]
                    for k in range(3):
                        ll[0].append(problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL))
                        ll[1].append(problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL))
                    same_line_count = 0
                    result, eq_premise = problem.solve_target(ll[0][0] * ll[1][1] - ll[0][1] * ll[1][0])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    result, eq_premise = problem.solve_target(ll[0][0] * ll[1][2] - ll[0][2] * ll[1][0])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    result, eq_premise = problem.solve_target(ll[0][1] * ll[1][2] - ll[0][2] * ll[1][1])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    if same_line_count >= 2:
                        update = problem.define_similar((tri1, tri2), premise, 81) or update

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_similar]:  # 判断镜像相似
                    premise = [problem.conditions.get_index(tri1, cType.triangle),
                               problem.conditions.get_index(tri2, cType.triangle)]
                    ll = [[], []]
                    for k in range(3):
                        ll[0].append(problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL))
                        ll[1].append(problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL))
                    same_line_count = 0
                    result, eq_premise = problem.solve_target(ll[0][0] * ll[1][1] - ll[0][1] * ll[1][0])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    result, eq_premise = problem.solve_target(ll[0][0] * ll[1][2] - ll[0][2] * ll[1][0])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    result, eq_premise = problem.solve_target(ll[0][1] * ll[1][2] - ll[0][2] * ll[1][1])
                    if result is not None and util.equal(result, 0):  # 边比值相等
                        premise += eq_premise
                        same_line_count += 1
                    if same_line_count >= 2:
                        update = problem.define_mirror_similar((tri1, tri2), premise, 81) or update

                j += 1
            i += rep.count_triangle

        return update

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

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        s12 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        a2 = problem.get_sym_of_attr(tri2[k] + tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        s22 = problem.get_sym_of_attr(tri2[(k + 1) % 3] + tri2[(k + 2) % 3], aType.LL)

                        result, eq_premise = problem.solve_target(a1 - a2)
                        if result is not None and util.equal(result, 0):  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s11 * s22 - s12 * s21)
                        if result is not None and util.equal(result, 0):  # 两边 成比例
                            premise += eq_premise
                            update = problem.define_similar((tri1, tri2), premise, 82) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_similar]:  # 判断镜像相似
                    for k in range(3):
                        premise = [problem.conditions.get_index(tri1, cType.triangle),
                                   problem.conditions.get_index(tri2, cType.triangle)]

                        s11 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3], aType.LL)
                        m_s12 = problem.get_sym_of_attr(tri2[(3 - k) % 3] + tri2[2 - k], aType.LL)
                        a1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_a2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[2 - k] + tri2[(3 - k) % 3], aType.MA)
                        s21 = problem.get_sym_of_attr(tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.LL)
                        m_s22 = problem.get_sym_of_attr(tri2[2 - k] + tri2[(4 - k) % 3], aType.LL)

                        result, eq_premise = problem.solve_target(a1 - m_a2)
                        if result is not None and util.equal(result, 0):  # a相等
                            premise += eq_premise
                        else:
                            continue
                        result, eq_premise = problem.solve_target(s11 * m_s22 - m_s12 * s21)
                        if result is not None and util.equal(result, 0):  # 两边 成比例
                            premise += eq_premise
                            update = problem.define_mirror_similar((tri1, tri2), premise, 82) or update
                            break

                j += 1
            i += rep.count_triangle

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

                        result, eq_premise = problem.solve_target(angle_1 - angle_2)
                        if result is not None and util.equal(result, 0):
                            equal_count += 1
                            premise += eq_premise
                        if equal_count >= 2:
                            update = problem.define_similar((tri1, tri2), premise, 83) or update
                            break

                if (tri1, tri2) not in problem.conditions.items[cType.mirror_similar]:  # 镜像相似
                    m_equal_count = 0
                    m_premise = [problem.conditions.get_index(tri1, cType.triangle),
                                 problem.conditions.get_index(tri2, cType.triangle)]
                    for k in range(3):  # 对应角相等
                        angle_1 = problem.get_sym_of_attr(tri1[k] + tri1[(k + 1) % 3] + tri1[(k + 2) % 3], aType.MA)
                        m_angle_2 = problem.get_sym_of_attr(tri2[(4 - k) % 3] + tri2[(5 - k) % 3] + tri2[(3 - k) % 3],
                                                            aType.MA)

                        result, eq_premise = problem.solve_target(angle_1 - m_angle_2)
                        if result is not None and util.equal(result, 0):
                            m_equal_count += 1
                            m_premise += eq_premise
                        if m_equal_count >= 2:
                            update = problem.define_mirror_similar((tri1, tri2), m_premise, 83) or update
                            break

                j += 1
            i += rep.count_triangle

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
            update = problem.define_equation(p - a - b - c, eType.basic, premise, 84) or update
            i += rep.count_triangle

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
            update = problem.define_equation(area_tri - 0.5 * len_base * len_altitude, eType.complex,
                                             premise, 85) or update
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
            update = problem.define_equation(equation, eType.complex, premise, 86) or update
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
                update = problem.define_equation(a - 0.5 * s1 * s2 * sin(angle), eType.complex, premise, 87) or update

            i += rep.count_triangle
        return update

    @staticmethod
    def theorem_88_sine(problem):
        """
        正弦定理
        """
        update = False
        i = 0
        while i < len(problem.conditions.items[cType.triangle]):
            tri = problem.conditions.items[cType.triangle][i]
            line_and_angle = [[], []]    # 边和角的符号表示
            for j in range(3):
                line = problem.get_sym_of_attr(tri[j] + tri[(j + 1) % 3], aType.LL)
                angle = problem.get_sym_of_attr(tri[(j + 1) % 3] + tri[(j + 2) % 3] + tri[(j + 3) % 3], aType.MA)
                line_and_angle[0].append(line)
                line_and_angle[1].append(sin(angle * pi / 180))

            premise = [problem.conditions.get_index(tri, cType.triangle)]
            equations = [line_and_angle[0][0] * line_and_angle[1][1] - line_and_angle[0][1] * line_and_angle[1][0],
                         line_and_angle[0][0] * line_and_angle[1][2] - line_and_angle[0][2] * line_and_angle[1][0],
                         line_and_angle[0][1] * line_and_angle[1][2] - line_and_angle[0][2] * line_and_angle[1][1]]
            for equation in equations:
                update = problem.define_equation(equation, eType.complex, premise, 88) or update

            i += rep.count_triangle  # 一个三角形多种表示

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
                update = problem.define_equation(equation, eType.complex, premise, 89) or update

            i += rep.count_triangle  # 一个三角形多种表示

        return update
