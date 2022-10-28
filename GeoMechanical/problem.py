from utility import Representation as rep
from utility import Utility as util
from facts import AttributionType as aType
from facts import EquationType as eType
from facts import ConditionType as cType
from facts import Condition
from facts import FormalLanguage
from sympy import symbols, solve, Float, Integer
from func_timeout import func_set_timeout
import os
from graphviz import Digraph
from theorem import TheoremMap
import pickle


class ProblemLogic:

    def __init__(self):
        """------Entity, Entity Relation, Equation------"""
        self.conditions = Condition()  # 题目条件

        """------------symbols and equation------------"""
        self.sym_of_attr = {}  # 属性的符号表示 (entity, aType): sym
        self.value_of_sym = {}  # 符号的值 sym: value
        self.basic_equations = {}
        self.complex_equations = {}
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        """----------Target----------"""
        self.target_count = 0  # 目标个数
        self.targets = []  # 解题目标

    """------------Construction------------"""

    def define_shape(self, shape, premise, theorem, root=True):
        if self.conditions.add(shape, cType.shape, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.conditions.get_index(shape, cType.shape)]

            for point in shape:  # 添加构成shape的点
                self.define_point(point, premise, -2)

            for shape in rep.shape(shape):  # shape多种表示
                self.conditions.add(shape, cType.shape, premise, -2)

            i = 0  # 当前长度为3窗口起始的位置
            while len(shape) > 2 and i < len(shape):  # 去掉共线点，得到实际图形
                point1 = shape[i]
                point2 = shape[(i + 1) % len(shape)]
                point3 = shape[(i + 2) % len(shape)]

                if util.is_collinear(point1, point2, point3, self):  # 三点共线，去掉中间的点
                    shape = shape.replace(point2, "")
                else:  # 不共线，窗口后移
                    i += 1

            if len(shape) == 3:  # 三角形
                self.define_triangle(shape, premise, -2)
            else:  # 多边形
                self.define_polygon(shape, premise, -2)
            return True
        return False

    def define_collinear(self, points, premise, theorem):  # 共线
        if len(points) > 2 and self.conditions.add(points, cType.collinear, premise, theorem):
            premise = [self.conditions.get_index(points, cType.collinear)]
            self.conditions.add(points[::-1], cType.collinear, premise, -2)

            for i in range(0, len(points) - 2):  # 定义平角
                f_angle = points[i:i + 3]
                self.define_angle(f_angle, premise, -2, False)
                self.define_angle(f_angle[::-1], premise, -2, False)
            return True
        return False

    """------------define Entity------------"""

    def define_point(self, point, premise, theorem):  # 点
        return self.conditions.add(point, cType.point, premise, theorem)

    def define_line(self, line, premise, theorem, root=True):  # 线
        if self.conditions.add(line, cType.line, premise, theorem):
            if root:  # 如果是 definition tree 的根节点，那子节点都使用当前节点的 premise
                premise = [self.conditions.get_index(line, cType.line)]
            self.conditions.add(line[::-1], cType.line, premise, -2)  # 一条线2种表示
            self.define_point(line[0], premise, -2)  # 定义线上的点
            self.define_point(line[1], premise, -2)
            return True
        return False

    def define_angle(self, angle, premise, theorem, root=True):  # 角
        if self.conditions.add(angle, cType.angle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(angle, cType.angle)]
            self.define_line(angle[0:2], premise, -2, False)  # 构成角的两个线
            self.define_line(angle[1:3], premise, -2, False)
            return True
        return False

    def define_triangle(self, triangle, premise, theorem, root=True):  # 三角形
        if self.conditions.add(triangle, cType.triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.triangle)]

            for triangle in rep.shape(triangle):  # 所有表示形式
                self.conditions.add(triangle, cType.triangle, premise, -2)
                self.define_angle(triangle, premise, -2, False)  # 定义3个角
            return True
        return False

    def define_right_triangle(self, triangle, premise, theorem, root=True):  # 直角三角形
        if self.conditions.add(triangle, cType.right_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.right_triangle)]
            self.define_triangle(triangle, premise, -2, False)  # RT三角形也是普通三角形
            return True
        return False

    def define_isosceles_triangle(self, triangle, premise, theorem, root=True):  # 等腰三角形
        if self.conditions.add(triangle, cType.isosceles_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.isosceles_triangle)]
            self.define_triangle(triangle, premise, -2, False)  # 等腰三角形也是普通三角形
            return True
        return False

    def define_equilateral_triangle(self, triangle, premise, theorem, root=True):  # 等边三角形
        if self.conditions.add(triangle, cType.equilateral_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.equilateral_triangle)]
            for triangle in rep.shape(triangle):
                self.conditions.add(triangle, cType.equilateral_triangle, premise, -2)
                self.define_isosceles_triangle(triangle, premise, -2, False)  # 等边也是等腰
            return True
        return False

    def define_polygon(self, shape, premise, theorem, root=True):  # 多边形
        if self.conditions.add(shape, cType.polygon, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.polygon)]

            for polygon in rep.shape(shape):
                self.conditions.add(polygon, cType.polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            return True
        return False

    """------------define Relation------------"""

    def define_midpoint(self, ordered_pair, premise, theorem, root=True):  # 中点
        point, line = ordered_pair
        if self.conditions.add(ordered_pair, cType.midpoint, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.midpoint)]
            self.conditions.add((point, line[::-1]), cType.midpoint, premise, -2)  # 中点有两中表示形式

            line1 = self.get_sym_of_attr(line[0] + point, aType.LL)  # 中点性质扩展
            line2 = self.get_sym_of_attr(point + line[1], aType.LL)
            self.define_equation(line1 - line2, eType.basic, premise, -2)

            self.define_point(point, premise, -2)  # 定义点和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_intersect(self, ordered_pair, premise, theorem, root=True):  # 线相交
        point, line1, line2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.intersect, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.intersect)]

            for intersect in util.same_intersects(ordered_pair, self):  # 根据共线得到多个相交关系
                for all_form in rep.intersect(intersect):  # 一个相交关系有4种表示方式
                    self.conditions.add(all_form, cType.intersect, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义实体
            self.define_line(line2, premise, -2, False)
            self.define_point(point, premise, -2)
            return True
        return False

    def define_parallel(self, ordered_pair, premise, theorem, root=True):  # 线平行
        line1, line2, = ordered_pair
        if self.conditions.add(ordered_pair, cType.parallel, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.parallel)]

            for parallel in rep.parallel(ordered_pair):  # 平行的4种表示
                self.conditions.add(parallel, cType.parallel, premise, -2)

            self.define_disorder_parallel(ordered_pair, premise, -2, False)    # 无序平行
            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            return True
        return False

    def define_disorder_parallel(self, ordered_pair, premise, theorem, root=True):  # 无序平行
        line1, line2, = ordered_pair
        if self.conditions.add(ordered_pair, cType.disorder_parallel, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.disorder_parallel)]

            for disorder_parallel in rep.disorder_parallel(ordered_pair):  # 平行的4种表示
                self.conditions.add(disorder_parallel, cType.disorder_parallel, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            return True
        return False

    def define_perpendicular(self, ordered_pair, premise, theorem, root=True):
        if self.conditions.add(ordered_pair, cType.perpendicular, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.perpendicular)]

            for pp in util.same_perpendiculars(ordered_pair, self):    # 根据共线得到多个垂直关系
                for all_form in rep.intersect(pp):  # 一个垂直关系有4种表示方式
                    self.conditions.add(all_form, cType.perpendicular, premise, -2)

                premise = [self.conditions.get_index(pp, cType.perpendicular)]
                point, line1, line2 = pp
                sym = []  # 垂直的扩展：90°角
                if len(set(point + line1 + line2)) == 3:
                    if line1[0] == line2[1]:
                        sym.append(self.get_sym_of_attr(line2 + line1[1], aType.MA))
                    elif line1[1] == line2[1]:
                        sym.append(self.get_sym_of_attr(line1 + line2[0], aType.MA))
                    elif line1[1] == line2[0]:
                        sym.append(self.get_sym_of_attr(line2[::-1] + line1[0], aType.MA))
                    elif line1[0] == line2[0]:
                        sym.append(self.get_sym_of_attr(line1[::-1] + line2[1], aType.MA))
                elif len(set(point + line1 + line2)) == 4:
                    if line2[1] == point:
                        sym.append(self.get_sym_of_attr(line1[0] + line2[::-1], aType.MA))
                        sym.append(self.get_sym_of_attr(line2 + line1[1], aType.MA))
                    elif line1[1] == point:
                        sym.append(self.get_sym_of_attr(line2[1] + line1[::-1], aType.MA))
                        sym.append(self.get_sym_of_attr(line1 + line2[0], aType.MA))
                    elif line2[0] == point:
                        sym.append(self.get_sym_of_attr(line1[1] + line2, aType.MA))
                        sym.append(self.get_sym_of_attr(line2[::-1] + line1[0], aType.MA))
                    elif line1[0] == point:
                        sym.append(self.get_sym_of_attr(line2[0] + line1, aType.MA))
                        sym.append(self.get_sym_of_attr(line1[::-1] + line2[1], aType.MA))
                elif len(set(point + line1 + line2)) == 5:
                    sym.append(self.get_sym_of_attr(line1[0] + point + line2[0], aType.MA))
                    sym.append(self.get_sym_of_attr(line2[0] + point + line1[1], aType.MA))
                    sym.append(self.get_sym_of_attr(line1[1] + point + line2[1], aType.MA))
                    sym.append(self.get_sym_of_attr(line2[1] + point + line1[0], aType.MA))
                    self.define_intersect(pp, premise, -2, False)  # 垂直也是相交

                for s in sym:  # 设置直角为90°
                    self.set_value_of_sym(s, 90, premise, -2)

            return True
        return False

    def define_perpendicular_bisector(self, ordered_pair, premise, theorem, root=True):  # 垂直平分
        point, line1, line2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.perpendicular_bisector, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.perpendicular_bisector)]
            for all_shape in rep.perpendicular_bisector(ordered_pair):  # 2种表示
                self.conditions.add(all_shape, cType.perpendicular_bisector, premise, theorem)

            self.define_perpendicular(ordered_pair, premise, -2, False)  # 垂直平分也是垂直
            self.define_midpoint((point, line1), premise, -2, False)  # 垂直平分也是平分
            self.define_line(line1, premise, -2, False)
            self.define_line(line2, premise, -2, False)
            self.define_point(point, premise, -2)
            return True
        return False

    def define_bisector(self, ordered_pair, premise, theorem, root=True):  # 角平分线
        line, angle = ordered_pair
        if self.conditions.add(ordered_pair, cType.bisector, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.bisector)]

            same_angles = util.same_angles(angle, self)    # 共线的存在，使得一个角平分线有多种表示
            for point in util.coll_points_one_side(line[0], line[1], self):
                for same_angle in same_angles:
                    self.conditions.add((line[0] + point, same_angle), cType.bisector, premise, -2)

            self.define_angle(angle, premise, -2, False)  # 定义角和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_median(self, ordered_pair, premise, theorem, root=True):  # 中线
        line, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.median, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.median)]
            self.define_line(line, premise, -2)  # 定义实体
            self.define_triangle(triangle, premise, -2)
            self.define_midpoint((line[1], triangle[1:3]), premise, -2, False)  # 底边中点
            return True
        return False

    def define_is_altitude(self, ordered_pair, premise, theorem, root=True):  # 高
        height, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.is_altitude, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.is_altitude)]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_neutrality(self, ordered_pair, premise, theorem, root=True):  # 中位线
        line, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.neutrality, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.neutrality)]
            self.define_line(line, premise, -2, False)  # 定义实体
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_circumcenter(self, ordered_pair, premise, theorem, root=True):  # 外心
        point, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.circumcenter, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.circumcenter)]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.conditions.add((point, tri), cType.circumcenter, premise, -2)
            self.define_point(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_incenter(self, ordered_pair, premise, theorem, root=True):  # 内心
        point, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.incenter, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.incenter)]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.conditions.add((point, tri), cType.incenter, premise, -2)
            self.define_point(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_centroid(self, ordered_pair, premise, theorem, root=True):  # 重心
        point, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.centroid, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.centroid)]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.conditions.add((point, tri), cType.centroid, premise, -2)
            self.define_point(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_orthocenter(self, ordered_pair, premise, theorem, root=True):  # 垂心
        point, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.orthocenter, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.orthocenter)]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.conditions.add((point, tri), cType.orthocenter, premise, -2)
            self.define_point(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_congruent(self, ordered_pair, premise, theorem, root=True):  # 全等
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.congruent, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.congruent)]
            triangle1_all = rep.shape(triangle1)
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):  # 6种
                self.conditions.add((triangle1_all[i], triangle2_all[i]), cType.congruent, premise, -2)
                self.conditions.add((triangle2_all[i], triangle1_all[i]), cType.congruent, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_similar(self, ordered_pair, premise, theorem, root=True):  # 相似
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.similar, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.similar)]
            triangle1_all = rep.shape(triangle1)
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):  # 6种表示方式
                self.conditions.add((triangle1_all[i], triangle2_all[i]), cType.similar, premise, -2)
                self.conditions.add((triangle2_all[i], triangle1_all[i]), cType.similar, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_mirror_congruent(self, ordered_pair, premise, theorem, root=True):  # 镜像全等
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.mirror_congruent, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.mirror_congruent)]
            for all_shape in rep.mirror_tri(ordered_pair):  # 6种表示
                self.conditions.add(all_shape, cType.mirror_congruent, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_mirror_similar(self, ordered_pair, premise, theorem, root=True):  # 镜像相似
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.mirror_similar, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.mirror_similar)]
            for all_shape in rep.mirror_tri(ordered_pair):  # 6种表示
                self.conditions.add(all_shape, cType.mirror_similar, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    """------------Equation and Symbol------------"""

    def define_equation(self, equation, equation_type, premise, theorem):  # 定义方程
        if equation not in self.conditions.items[cType.equation]:    # 新方程，加入条件
            self.conditions.add(equation, cType.equation, premise, theorem)
            if equation_type is eType.basic and equation not in self.basic_equations.keys():
                self.basic_equations[equation] = equation  # 简单方程
                self.equation_solved = False
            elif equation_type is eType.complex and equation not in self.complex_equations.keys():
                self.complex_equations[equation] = equation  # 复杂方程
                self.equation_solved = False
            return True
        return False

    def get_sym_of_attr(self, entity, attr_type):  # 得到属性的符号表示
        if attr_type is aType.M:  # 表示目标/中间值类型的符号，不用存储在符号库
            return symbols(attr_type.name.lower() + "_" + entity)

        if (entity, attr_type) not in self.sym_of_attr.keys():  # 若无符号，新建符号
            if attr_type in [aType.MA, aType.F]:
                sym = symbols(attr_type.name.lower() + "_" + entity.lower())  # 角、自由变量可以有负的
            else:
                sym = symbols(attr_type.name.lower() + "_" + entity.lower(), positive=True)  # 属性值没有负数

            self.value_of_sym[sym] = None  # 符号值

            if attr_type in [aType.LL, aType.AS, aType.PT]:  # entity 有多种表示形式
                for all_form in rep.shape(entity):
                    self.sym_of_attr[(all_form, attr_type)] = sym
            else:  # entity 只有一种形式
                self.sym_of_attr[(entity, attr_type)] = sym  # 符号

        else:  # 有符号就返回符号
            sym = self.sym_of_attr[(entity, attr_type)]

        return sym

    def set_value_of_sym(self, sym, value, premise, theorem, solved_by_equation=False):  # 设置符号的值
        if self.value_of_sym[sym] is None:  # 如果当前符号未赋值
            self.value_of_sym[sym] = value
            if sym - value not in self.conditions.items[cType.equation]:  # 如果符号值没有保存在equation中
                self.define_equation(sym - value, eType.value, premise, theorem)
                if not solved_by_equation:  # 如果不是通过解方程得到的
                    self.equation_solved = False
            return True
        return False


class Problem(ProblemLogic):

    def __init__(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):
        super().__init__()
        """------------题目输入------------"""
        self.problem_index = problem_index
        self.fl = FormalLanguage(construction_fls, text_fls, image_fls, target_fls, problem_index)
        self.theorem_seqs = theorem_seqs
        self.answer = answer

        """------------辅助功能------------"""
        self.solve_time_list = []    # 求解时间

        self.dot = None    # 以下是求解树相关数据结构
        self.nodes = []  # 点集 (node_name, node_type, c_index) 三元组
        self.node_count = 0  # 结点计数
        self.edges = {}  # 边集 key:start_node value:end_node

        self.used = []  # 解题过程中用到的 problem 条件的 index

    """------------构造图形------------"""

    def construct_all_shape(self):
        """
        拼图法构造新shape
        Shape(ABC), Shape(CBD) ==> Shape(ABD)
        """
        update = True
        traversed = []  # 记录已经计算过的，避免重复计算
        while update:
            update = False
            for shape1 in self.conditions.items[cType.shape]:
                for shape2 in self.conditions.items[cType.shape]:
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
                            premise = [self.conditions.get_index(shape1, cType.shape),
                                       self.conditions.get_index(shape2, cType.shape)]
                            update = self.define_shape(new_shape, premise, -2) or update

    def construct_all_line(self):
        """
        拼图法构造新line
        Collinear(ABC) ==> Line(AB), Line(BC), Line(AC)
        """
        i = 0
        while i < len(self.conditions.items[cType.collinear]):
            coll = self.conditions.items[cType.collinear][i]
            for line in util.all_lines_in_coll(coll, self):
                self.define_line(line, [self.conditions.get_index(coll, cType.collinear)], -2)
            i += rep.count_collinear

    def angle_representation_alignment(self):  # 使角的角度表示符号一致
        for angle in self.conditions.items[cType.angle]:
            if (aType.MA, angle) in self.sym_of_attr.keys():  # 有符号了就不用再赋予了
                continue

            sym = self.get_sym_of_attr(angle, aType.MA)    # 角的符号表示

            same_angles = util.same_angles(angle, self)
            for same_angle in same_angles:
                self.sym_of_attr[(same_angle, aType.MA)] = sym    # 本质相同的角赋予相同的符号表示

    def flat_angle(self):  # 平角赋予180°
        for coll in self.conditions.items[cType.collinear]:
            premise = [self.conditions.get_index(coll, cType.collinear)]
            for i in range(0, len(coll) - 2):
                sym_of_angle = self.get_sym_of_attr(coll[i:i + 3], aType.MA)
                self.set_value_of_sym(sym_of_angle, 180, premise, -2)

    """------------解方程相关------------"""

    def solve_target(self, target_expr):  # 求解target_expr的值
        # 无需求解的情况
        if target_expr in self.conditions.items[cType.equation]:  # 如果是已知方程，那么target_expr=0
            return 0.0, [self.conditions.get_index(target_expr, cType.equation)]
        if -target_expr in self.conditions.items[cType.equation]:
            return 0.0, [self.conditions.get_index(-target_expr, cType.equation)]

        # 简单替换就可求解的情况
        premise = []
        for sym in target_expr.free_symbols:  # 替换掉target_expr中符号值已知的符号
            if self.value_of_sym[sym] is not None:
                premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])
        if len(target_expr.free_symbols) == 0:
            return float(target_expr), premise

        # 需要求解的情况
        equations, eq_premise = self.get_minimum_equations(target_expr)  # 最小依赖方程组
        premise += eq_premise  # 上面替换掉的符号的premise
        # print("高级化简之前:", end="  ")
        # print(equations, end=",  ")
        # print(target_expr)
        equations = self.high_level_simplify(equations, target_expr)  # 高级化简
        target_sym = symbols("t_s")
        equations[-1] = target_sym - equations[-1]  # 添加目标方程
        # print("高级化简之后:", end="  ")
        # print(equations)
        solved_result = solve(equations)  # 求解最小方程组
        # print(solved_result)
        # print()

        if len(solved_result) > 0 and isinstance(solved_result, list):  # 若解不唯一，选择第一个
            solved_result = solved_result[0]

        if len(solved_result) > 0 and \
                target_sym in solved_result.keys() and \
                (isinstance(solved_result[target_sym], Float) or
                 isinstance(solved_result[target_sym], Integer)):
            return float(solved_result[target_sym]), list(set(premise))  # 有实数解，返回解

        return None, None  # 无解，返回None

    def get_minimum_equations(self, target_expr):  # 返回求解target_expr依赖的最小方程组
        sym_set = target_expr.free_symbols  # 方程组涉及到的符号
        min_equations = []  # 最小方程组
        premise = []  # 每个方程的index，作为target_expr求解结果的premise

        # 循环添加依赖方程，得到最小方程组
        update = True
        while update:
            update = False
            for sym in sym_set:
                if self.value_of_sym[sym] is None:  # 如果sym的值未求出，需要添加包含sym的依赖方程
                    for key in self.basic_equations.keys():  # 添加简单依赖方程
                        if sym in self.basic_equations[key].free_symbols and \
                                self.basic_equations[key] not in min_equations:
                            min_equations.append(self.basic_equations[key])
                            premise.append(self.conditions.get_index(key, cType.equation))
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号(未化简的原方程的所有符号)
                            update = True
                    for key in self.complex_equations.keys():  # 添加复杂依赖方程
                        if sym in self.complex_equations[key].free_symbols and \
                                self.complex_equations[key] not in min_equations:
                            min_equations.append(self.complex_equations[key])
                            premise.append(self.conditions.get_index(key, cType.equation))
                            sym_set = sym_set.union(key.free_symbols)  # 添加新方程会引入新符号
                            update = True
        # 化简最小方程组
        for sym in sym_set:
            if self.value_of_sym[sym] is not None:
                premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                for i in range(len(min_equations)):  # 替换方程中的已知sym
                    min_equations[i] = min_equations[i].subs(sym, self.value_of_sym[sym])
                target_expr = target_expr.subs(sym, self.value_of_sym[sym])  # 替换target_expr中的已知sym

        return min_equations, premise  # 返回化简的target_expr、最小依赖方程组和前提

    def simplify_equations(self):  # 化简 basic、complex equations
        update = True
        while update:
            update = False
            remove_lists = []  # 要删除的 basic equation 列表
            for key in self.basic_equations.keys():
                for sym in self.basic_equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if self.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        self.basic_equations[key] = self.basic_equations[key].subs(sym, self.value_of_sym[sym])
                        update = True

                if len(self.basic_equations[key].free_symbols) == 0:  # 没有未知符号：删除方程
                    remove_lists.append(key)

                if len(self.basic_equations[key].free_symbols) == 1:  # 只剩一个符号：求得符号值，然后删除方程
                    target_sym = list(self.basic_equations[key].free_symbols)[0]
                    value = solve(self.basic_equations[key])[0]
                    premise = [self.conditions.get_index(key, cType.equation)]  # 前提
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                    self.set_value_of_sym(target_sym, value, premise, -3, True)
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                self.basic_equations.pop(remove_eq)

            remove_lists = []  # 要删除的 complex equation 列表
            for key in self.complex_equations.keys():
                for sym in self.complex_equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if self.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        self.complex_equations[key] = self.complex_equations[key].subs(sym, self.value_of_sym[sym])
                        update = True

                if len(self.complex_equations[key].free_symbols) == 0:  # 没有未知符号：删除方程
                    remove_lists.append(key)

                if len(self.complex_equations[key].free_symbols) == 1:  # 只剩一个符号：求得符号值，然后删除方程
                    target_sym = list(self.complex_equations[key].free_symbols)[0]
                    value = solve(self.complex_equations[key])[0]
                    premise = [self.conditions.get_index(key, cType.equation)]  # 前提
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                    self.set_value_of_sym(target_sym, value, premise, -3, True)
                    remove_lists.append(key)

            for remove_eq in remove_lists:  # 删除所有符号值已知的方程
                self.complex_equations.pop(remove_eq)

    @staticmethod
    def high_level_simplify(equations, target_expr):    # 基于替换的高级化简
        update = True
        while update:
            update = False
            for equation in equations:    # 替换符号
                if len(equation.free_symbols) == 2:
                    result = solve(equation)
                    if len(result) > 0:  # 有解
                        if isinstance(result, list):  # 若解不唯一，选择第一个
                            result = result[0]
                        sym = list(result.keys())[0]
                        target_expr = target_expr.subs(sym, result[sym])    # 符号替换
                        for i in range(len(equations)):
                            equations[i] = equations[i].subs(sym, result[sym])    # 符号替换
                        update = True

        equations.append(target_expr)

        return equations

    @func_set_timeout(8)  # 限时8s
    def solve_equations(self):  # 求解basic、complex equations
        if self.equation_solved:  # equations没有更新，不用重复求解
            return

        update = True
        while update:
            update = False
            # solve前先化简方程
            self.simplify_equations()

            # 得到所有值未知的符号
            sym_set = []
            for equation in list(self.basic_equations.values()) + list(self.complex_equations.values()):
                sym_set += equation.free_symbols
            sym_set = list(set(sym_set))  # 快速去重

            # 方程求解
            for sym in sym_set:
                equations, premise = self.get_minimum_equations(sym)
                result = solve(equations)  # 求解最小方程组

                if len(result) > 0:  # 有解
                    if isinstance(result, list):  # 若解不唯一，选择第一个
                        result = result[0]
                    for key in result.keys():  # 遍历并保存所有解
                        if self.value_of_sym[key] is None \
                                and (isinstance(result[key], Float) or isinstance(result[key], Integer)):
                            self.set_value_of_sym(key, float(result[key]), premise, -3, True)
                            update = True

        self.equation_solved = True
        # 注释掉下列语句相当于合并complex和basic
        self.complex_equations = {}  # 清空 complex equations

    """------------辅助功能: 初始化新问题------------"""
    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):
        """-------------------题目输入-------------------"""
        self.problem_index = problem_index
        self.fl = FormalLanguage(construction_fls, text_fls, image_fls, target_fls, problem_index)
        self.theorem_seqs = theorem_seqs
        self.answer = answer

        """-------------------辅助功能-------------------"""
        self.solve_time_list = []
        self.dot = None  # 以下是求解树相关数据结构
        self.nodes = []  # 点集 (node_name, node_type, c_index) 三元组
        self.node_count = 0  # 结点计数
        self.edges = {}  # 边集 (node_index, node_index) 二元组
        self.used = []  # 解题过程中用到的 problem 条件的 index

        """------Entity, Entity Relation, Equation------"""
        self.conditions.clean()

        """------------symbols and equation------------"""
        self.sym_of_attr = {}  # (entity, aType): sym
        self.value_of_sym = {}  # sym: value
        self.basic_equations = {}
        self.complex_equations = {}
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        """----------Target----------"""
        self.target_count = 0  # 目标个数
        self.targets = []  # 解题目标

    """------------辅助功能: 反向生成形式化语句------------"""
    def anti_generate_all_fl_by_step(self):
        """---------construction---------"""
        # i = 0
        # while i < len(self.conditions.items[cType.shape]):    # shape 没必要生成FL
        #     item = self.conditions.items[cType.shape][i]
        i = 0
        while i < len(self.conditions.items[cType.collinear]):
            item = self.conditions.items[cType.collinear][i]
            self.fl.add(("Collinear", item))
            i = i + rep.count_collinear

        """---------entity---------"""
        # i = 0
        # while i < len(self.conditions.items[cType.point]):    # point 没必要生成FL
        #     item = self.conditions.items[cType.point][i]
        #     i = i + rep.count_point
        # i = 0
        # while i < len(self.conditions.items[cType.line]):    # line 暂无必要生成FL
        #     item = self.conditions.items[cType.line][i]
        #     self.fl.add(("Line", item))
        #     i = i + rep.count_line
        # i = 0
        # while i < len(self.conditions.items[cType.angle]):    # angle 暂无必要生成FL
        #     item = self.conditions.items[cType.angle][i]
        #     self.fl.add(("Angle", item))
        #     i = i + rep.count_angle
        i = 0
        while i < len(self.conditions.items[cType.triangle]):
            item = self.conditions.items[cType.triangle][i]
            self.fl.add(("Triangle", item))
            i = i + rep.count_triangle
        i = 0
        while i < len(self.conditions.items[cType.right_triangle]):
            item = self.conditions.items[cType.right_triangle][i]
            self.fl.add(("RightTriangle", item))
            i = i + rep.count_right_triangle
        i = 0
        while i < len(self.conditions.items[cType.isosceles_triangle]):
            item = self.conditions.items[cType.isosceles_triangle][i]
            self.fl.add(("IsoscelesTriangle", item))
            i = i + rep.count_isosceles_triangle
        i = 0
        while i < len(self.conditions.items[cType.equilateral_triangle]):
            item = self.conditions.items[cType.equilateral_triangle][i]
            self.fl.add(("EquilateralTriangle", item))
            i = i + rep.count_equilateral_triangle
        # i = 0
        # while i < len(self.conditions.items[cType.polygon]):    # polygon 没必要生成FL
        #     item = self.conditions.items[cType.polygon][i]

        """---------entity relation---------"""
        i = 0
        while i < len(self.conditions.items[cType.midpoint]):
            item = self.conditions.items[cType.midpoint][i]
            self.fl.add(("Midpoint", item[0], item[1]))
            i = i + rep.count_midpoint
        i = 0
        while i < len(self.conditions.items[cType.intersect]):
            item = self.conditions.items[cType.intersect][i]
            self.fl.add(("Intersect", item[0], item[1], item[2]))
            i = i + rep.count_intersect
        i = 0
        while i < len(self.conditions.items[cType.parallel]):
            item = self.conditions.items[cType.parallel][i]
            self.fl.add(("Parallel", item[0], item[1]))
            i = i + rep.count_parallel
        i = 0
        while i < len(self.conditions.items[cType.disorder_parallel]):
            item = self.conditions.items[cType.disorder_parallel][i]
            self.fl.add(("DisorderParallel", item[0], item[1]))
            i = i + rep.count_disorder_parallel
        i = 0
        while i < len(self.conditions.items[cType.perpendicular]):
            item = self.conditions.items[cType.perpendicular][i]
            self.fl.add(("Perpendicular", item[0], item[1], item[2]))
            i = i + rep.count_perpendicular
        i = 0
        while i < len(self.conditions.items[cType.perpendicular_bisector]):
            item = self.conditions.items[cType.perpendicular_bisector][i]
            self.fl.add(("PerpendicularBisector", item[0], item[1], item[2]))
            i = i + rep.count_perpendicular_bisector
        i = 0
        while i < len(self.conditions.items[cType.bisector]):
            item = self.conditions.items[cType.bisector][i]
            self.fl.add(("Bisector", item[0], item[1]))
            i = i + rep.count_bisector
        i = 0
        while i < len(self.conditions.items[cType.median]):
            item = self.conditions.items[cType.median][i]
            self.fl.add(("Median", item[0], item[1]))
            i = i + rep.count_median
        i = 0
        while i < len(self.conditions.items[cType.is_altitude]):
            item = self.conditions.items[cType.is_altitude][i]
            self.fl.add(("IsAltitude", item[0], item[1]))
            i = i + rep.count_is_altitude
        i = 0
        while i < len(self.conditions.items[cType.neutrality]):
            item = self.conditions.items[cType.neutrality][i]
            self.fl.add(("Neutrality", item[0], item[1]))
            i = i + rep.count_neutrality
        i = 0
        while i < len(self.conditions.items[cType.circumcenter]):
            item = self.conditions.items[cType.circumcenter][i]
            self.fl.add(("Circumcenter", item[0], item[1]))
            i = i + rep.count_circumcenter
        i = 0
        while i < len(self.conditions.items[cType.incenter]):
            item = self.conditions.items[cType.incenter][i]
            self.fl.add(("Incenter", item[0], item[1]))
            i = i + rep.count_incenter
        i = 0
        while i < len(self.conditions.items[cType.centroid]):
            item = self.conditions.items[cType.centroid][i]
            self.fl.add(("Centroid", item[0], item[1]))
            i = i + rep.count_centroid
        i = 0
        while i < len(self.conditions.items[cType.orthocenter]):
            item = self.conditions.items[cType.orthocenter][i]
            self.fl.add(("Orthocenter", item[0], item[1]))
            i = i + rep.count_orthocenter
        i = 0
        while i < len(self.conditions.items[cType.congruent]):
            item = self.conditions.items[cType.congruent][i]
            self.fl.add(("Congruent", item[0], item[1]))
            i = i + rep.count_congruent
        i = 0
        while i < len(self.conditions.items[cType.similar]):
            item = self.conditions.items[cType.similar][i]
            self.fl.add(("Similar", item[0], item[1]))
            i = i + rep.count_similar
        i = 0
        while i < len(self.conditions.items[cType.mirror_congruent]):
            item = self.conditions.items[cType.mirror_congruent][i]
            self.fl.add(("MirrorCongruent", item[0], item[1]))
            i = i + rep.count_mirror_congruent
        i = 0
        while i < len(self.conditions.items[cType.mirror_similar]):
            item = self.conditions.items[cType.mirror_similar][i]
            self.fl.add(("MirrorSimilar", item[0], item[1]))
            i = i + rep.count_mirror_similar

        """---------attribute---------"""
        processed = []
        for key in self.sym_of_attr.keys():
            sym = self.sym_of_attr[key]
            if self.value_of_sym[sym] is not None and sym not in processed:
                entity, a_type = key
                if a_type is aType.LL:
                    self.fl.add(("Length", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                elif a_type is aType.MA:
                    self.fl.add(("Measure", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                elif a_type is aType.AS:
                    self.fl.add(("Area", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                elif a_type is aType.PT:
                    self.fl.add(("Perimeter", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                elif a_type is aType.AT:
                    self.fl.add(("Altitude", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                elif a_type is aType.F:
                    self.fl.add(("Free", entity, "{:.3f}".format(float(self.value_of_sym[sym]))))
                processed.append(sym)

        """---------equation---------"""
        for equation in self.conditions.items[cType.equation]:
            if len(equation.free_symbols) > 1:
                self.fl.add(("Equation", str(equation)))

        self.fl.step()  # step

    def anti_generate_one_fl_by_index(self, index):
        if index == -1:
            return tuple(["Premise", "-1"])

        item, c_type = self.conditions.item_list[index]
        if c_type is cType.equation:
            result = ("Equation", str(item))
        elif c_type is cType.shape:
            result = ("Shape", item)
        elif c_type is cType.collinear:
            result = ("Collinear", item)
        elif c_type is cType.point:
            result = ("Point", item)
        elif c_type is cType.line:
            result = ("Line", item)
        elif c_type is cType.angle:
            result = ("Angle", item)
        elif c_type is cType.triangle:
            result = ("Triangle", item)
        elif c_type is cType.right_triangle:
            result = ("RightTriangle", item)
        elif c_type is cType.isosceles_triangle:
            result = ("IsoscelesTriangle", item)
        elif c_type is cType.equilateral_triangle:
            result = ("EquilateralTriangle", item)
        elif c_type is cType.polygon:
            result = ("Polygon", item)
        elif c_type is cType.midpoint:
            result = ("MidPoint", item[0], item[1])
        elif c_type is cType.intersect:
            result = ("Intersect", item[0], item[1], item[2])
        elif c_type is cType.parallel:
            result = ("Parallel", item[0], item[1])
        elif c_type is cType.disorder_parallel:
            result = ("DisorderParallel", item[0], item[1])
        elif c_type is cType.perpendicular:
            result = ("Perpendicular", item[0], item[1], item[2])
        elif c_type is cType.perpendicular_bisector:
            result = ("PerpendicularBisector", item[0], item[1], item[2])
        elif c_type is cType.bisector:
            result = ("Bisector", item[0], item[1])
        elif c_type is cType.median:
            result = ("Median", item[0], item[1])
        elif c_type is cType.is_altitude:
            result = ("IsAltitude", item[0], item[1])
        elif c_type is cType.neutrality:
            result = ("Neutrality", item[0], item[1])
        elif c_type is cType.circumcenter:
            result = ("Circumcenter", item[0], item[1])
        elif c_type is cType.incenter:
            result = ("Incenter", item[0], item[1])
        elif c_type is cType.centroid:
            result = ("Centroid", item[0], item[1])
        elif c_type is cType.orthocenter:
            result = ("Orthocenter", item[0], item[1])
        elif c_type is cType.congruent:
            result = ("Congruent", item[0], item[1])
        elif c_type is cType.similar:
            result = ("Similar", item[0], item[1])
        elif c_type is cType.mirror_congruent:
            result = ("MirrorCongruent", item[0], item[1])
        elif c_type is cType.mirror_similar:
            result = ("MirrorSimilar", item[0], item[1])
        else:
            result = None

        return result

    """------------辅助功能: show------------"""
    def show(self):
        # Formal Language
        print("\033[36mproblem_index:\033[0m", end=" ")
        print(self.problem_index)
        print("\033[36mconstruction_fls:\033[0m")
        for construction_fl in self.fl.construction_fls:
            print(construction_fl)
        print("\033[36mtext_fls:\033[0m")
        for text_fl in self.fl.text_fls:
            print(text_fl)
        print("\033[36mimage_fls:\033[0m")
        for image_fl in self.fl.image_fls:
            print(image_fl)
        print("\033[36mtarget_fls:\033[0m")
        for target_fl in self.fl.target_fls:  # 解析 formal language
            print(target_fl)
        print("\033[36mtheorem_seqs:\033[0m", end=" ")
        for theorem in self.theorem_seqs:
            print(theorem, end=" ")
        print()
        print("\033[36mreasoning_fls:\033[0m")
        for step in self.fl.reasoning_fls.keys():
            for fl in self.fl.reasoning_fls[step]:
                print("step {}:".format(step), end="  ")
                print(fl)

        for i in range(self.target_count):  # 找到所有解题需要的条件
            target = self.targets[i]
            if target.target_solved:
                self.used += target.premise
        self.used = list(set(self.used))
        update = True
        while update:
            update = False
            for i in self.used:
                for j in self.conditions.get_premise_by_index(i):
                    if j not in self.used:
                        self.used.append(j)
                        update = True

        # Logic-Construction
        print("\033[33mConstruction:\033[0m")
        for entity in Condition.construction_list:
            if len(self.conditions.items[entity]) > 0:
                print("{}:".format(entity.name))
                for item in self.conditions.items[entity]:
                    if self.conditions.get_index(item, entity) not in self.used:
                        print_str = "{0:^6}{1:^15}{2:^25}{3:^6}"
                    else:
                        print_str = "\033[35m{0:^6}{1:^15}{2:^25}{3:^6}\033[0m"
                    print(print_str.format(self.conditions.get_index(item, entity),
                                           str(item),
                                           str(self.conditions.get_premise(item, entity)),
                                           self.conditions.get_theorem(item, entity)))

        # Logic-Entity
        print("\033[33mEntity:\033[0m")
        for entity in Condition.entity_list:
            if len(self.conditions.items[entity]) > 0:
                print("{}:".format(entity.name))
                for item in self.conditions.items[entity]:
                    if self.conditions.get_index(item, entity) not in self.used:
                        print_str = "{0:^6}{1:^15}{2:^25}{3:^6}"
                    else:
                        print_str = "\033[35m{0:^6}{1:^15}{2:^25}{3:^6}\033[0m"
                    print(print_str.format(self.conditions.get_index(item, entity),
                                           str(item),
                                           str(self.conditions.get_premise(item, entity)),
                                           self.conditions.get_theorem(item, entity)))
        # Logic-EntityRelation
        print("\033[33mEntity Relation:\033[0m")
        for entity_relation in Condition.entity_relation_list:
            if len(self.conditions.items[entity_relation]) > 0:
                print("{}:".format(entity_relation.name))
                for item in self.conditions.items[entity_relation]:
                    if self.conditions.get_index(item, entity_relation) not in self.used:
                        print_str = "{0:^6}{1:^25}{2:^25}{3:^6}"
                    else:
                        print_str = "\033[35m{0:^6}{1:^25}{2:^25}{3:^6}\033[0m"
                    print(print_str.format(self.conditions.get_index(item, entity_relation),
                                           str(item),
                                           str(self.conditions.get_premise(item, entity_relation)),
                                           self.conditions.get_theorem(item, entity_relation)))
        # Logic-Attribution&Symbol&Equation
        print("\033[33mEntity\'s Symbol and Value:\033[0m")
        for attr in self.sym_of_attr.keys():
            if isinstance(self.value_of_sym[self.sym_of_attr[attr]], Float):
                print("{0:^10}{1:^10}{2:^15.3f}".format(attr[0],
                                                        str(self.sym_of_attr[attr]),
                                                        self.value_of_sym[self.sym_of_attr[attr]]))
            else:
                print("{0:^10}{1:^10}{2:^15}".format(attr[0],
                                                     str(self.sym_of_attr[attr]),
                                                     str(self.value_of_sym[self.sym_of_attr[attr]])))

        print("\033[33mEquations:\033[0m")
        if len(self.conditions.items[Condition.equation]) > 0:
            print("{}:".format(Condition.equation.name))
            for item in self.conditions.items[Condition.equation]:
                if self.conditions.get_index(item, Condition.equation) not in self.used:
                    print_str = "{0:^6}{1:^70}{2:^25}{3:>6}"
                else:
                    print_str = "\033[35m{0:^6}{1:^70}{2:^25}{3:>6}\033[0m"
                if len(self.conditions.get_premise(item, Condition.equation)) > 5:
                    print(print_str.format(self.conditions.get_index(item, Condition.equation),
                                           str(item),
                                           str(self.conditions.get_premise(item, Condition.equation)[0:5]) + "...",
                                           self.conditions.get_theorem(item, Condition.equation)))
                else:
                    print(print_str.format(self.conditions.get_index(item, Condition.equation),
                                           str(item),
                                           str(self.conditions.get_premise(item, Condition.equation)),
                                           self.conditions.get_theorem(item, Condition.equation)))

        # target and answer
        print("\033[34mTarget Count:\033[0m {}".format(self.target_count))
        for i in range(self.target_count):
            target = self.targets[i]
            print("\033[34m{}:\033[0m".format(target.target_type.name), end="  ")
            print(target.target, end="  ")
            print(target.solved_answer, end="  ")
            print(target.premise, end="  ")
            print(target.theorem, end="  ")
            print("\033[34mcorrect answer:\033[0m {}".format(self.answer[i]), end="  ")
            if target.target_solved:
                print("\033[32msolved\033[0m")
            else:
                print("\033[31munsolved\033[0m")

        # 求解时间
        for solve_time in self.solve_time_list:
            print(solve_time)

    def simpel_show(self):
        print("\033[36mproblem_index:\033[0m", end=" ")
        print(self.problem_index)

        for i in range(self.target_count):
            target = self.targets[i]
            print("\033[34m{}:\033[0m".format(target.target_type.name), end="  ")
            print(target.target, end="  ")
            print(target.solved_answer, end="  ")
            print(target.premise, end="  ")
            print(target.theorem, end="  ")
            print("\033[34mcorrect answer:\033[0m {}".format(self.answer[i]), end="  ")
            if target.target_solved:
                print("\033[32msolved\033[0m")
            else:
                print("\033[31munsolved\033[0m")

    """------------辅助功能: 求解树生成和保存相关------------"""
    def generate_tree(self):
        self.dot = Digraph(name=str(self.problem_index))  # 求解树

        group = {}    # 将题目条件分组，有相同前提和定理的放一块
        fl = []    # 条件反向解析为形式化语句
        for i in range(len(self.conditions.item_list)):
            fl.append(self.anti_generate_one_fl_by_index(i))
            p = tuple(self.conditions.get_premise_by_index(i))
            t = self.conditions.get_theorem_by_index(i)
            if (p, t) not in group.keys():
                group[(p, t)] = [i]
            else:
                group[(p, t)].append(i)
        fl.append(self.anti_generate_one_fl_by_index(-1))

        for key in group.keys():    # 生成求解树
            premise, theorem = key
            condition = group[key]
            t_index = self.add_node(TheoremMap.get_theorem_name(theorem))
            for p in premise:
                p_index = self.add_node(fl[p])
                self.add_edge(p_index, t_index)
            for c in condition:
                c_index = self.add_node(fl[c])
                self.add_edge(t_index, c_index)

        for i in range(self.target_count):   # 添加解题目标到求解树
            target = self.targets[i]
            if target.target_solved:
                t_index = self.add_node(TheoremMap.get_theorem_name(target.theorem))  # 定理
                target_index = self.add_node(("Target", str(target.target)))  # 目标
                self.add_edge(t_index, target_index)
                for premise in target.premise:
                    p_index = self.add_node(self.anti_generate_one_fl_by_index(premise))  # 前提
                    self.add_edge(p_index, t_index)

    def add_node(self, node):    # 添加点
        if node in self.nodes:    # node 已经添加
            return self.nodes.index(node)

        new_node_index = self.node_count    # 新node，添加进点集
        self.nodes.append(node)
        self.node_count += 1
        if isinstance(node, tuple):
            self.dot.node(str(new_node_index), str(node), shape='box')  # 条件结点
        else:
            self.dot.node(str(new_node_index), str(node))    # 定理结点
        return new_node_index

    def add_edge(self, node_start_index, node_end_index):    # 添加边
        if self.nodes[node_start_index] not in self.edges:
            self.edges[self.nodes[node_start_index]] = [self.nodes[node_end_index]]
        else:
            self.edges[self.nodes[node_start_index]].append(self.nodes[node_end_index])
        self.dot.edge(str(node_start_index), str(node_end_index))

    def save(self, file_dir):    # 保存求解树
        if "{}_graph.pk".format(self.problem_index) in os.listdir(file_dir):    # 已经求解过就不用再求解了
            return

        if self.dot is None:    # 若未生成求解树，先生成
            self.generate_tree()

        with open(file_dir + "{}_graph.pk".format(self.problem_index), "wb") as f:    # 保存求解树
            pickle.dump(self.edges, f)
        self.dot.render(directory=file_dir, view=False, format="png")
        os.remove(file_dir + "{}.gv".format(self.problem_index))    # 这个文件不需要
        os.rename(file_dir + "{}.gv.png".format(self.problem_index), file_dir + "{}.png".format(self.problem_index))
