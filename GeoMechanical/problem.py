from utility import Representation as rep
from facts import AttributionType as aType
from facts import EquationType as eType
from facts import ConditionType as cType
from facts import Condition
from sympy import symbols, solve, Float
from func_timeout import func_set_timeout


class ProblemLogic:

    def __init__(self):
        """------Entity, Entity Relation, Equation------"""
        self.conditions = Condition()  # 题目条件

        """------------symbols and equation------------"""
        self.sym_of_attr = {}  # 属性的符号表示 (ConditionType, "name"): sym
        self.value_of_sym = {}  # 符号的值 sym: value
        self.basic_equations = {}
        self.theorem_equations = {}
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        """----------Target----------"""
        self.target_count = 0  # 目标个数
        self.target_type = []  # 解题目标的类型
        self.target = []  # 解题目标
        self.target_solved = []  # 条件求解情况
        self.answer = []  # 答案
        self.premise = []  # 前提条件集合

    """------------define Entity------------"""

    def define_point(self, point, premise, theorem):  # 点
        return self.conditions.add(point, cType.point, premise, theorem)

    def define_line(self, line, premise, theorem, root=True):  # 线
        if self.conditions.add(line, cType.line, premise, theorem):
            if root:  # 如果是 definition tree 的根节点，那子节点都使用当前节点的 premise
                premise = [self.conditions.get_index(line, cType.line)]
            self.conditions.add(line[::-1], cType.line, premise, -2)  # 一条线2种表示
            self.conditions.add(line[0], cType.point, premise, -2)  # 定义线上的点
            self.conditions.add(line[1], cType.point, premise, -2)
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

    def define_arc(self, arc, premise, theorem, root=True):  # 弧
        if self.conditions.add(arc, cType.arc, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.conditions.get_index(arc, cType.arc)]
            self.conditions.add(arc[0], cType.point, premise, -2)  # 构成弧的两个点
            self.conditions.add(arc[1], cType.point, premise, -2)
            return True
        return False

    def define_shape(self, shape, premise, theorem, root=True):
        if self.conditions.add(shape, cType.shape, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.conditions.get_index(shape, cType.shape)]
            for point in shape:  # 添加构成shape的点。因为shape形状不确定，只能知道有这些点。
                self.conditions.add(point, cType.point, premise, -2)
            for shape in rep.shape(shape):  # n种表示
                self.conditions.add(shape, cType.shape, premise, -2)
            return True
        return False

    def define_circle(self, circle, premise, theorem, root=True):  # 圆
        if self.conditions.add(circle, cType.circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(circle, cType.circle)]
            self.conditions.add(circle, cType.point, premise, -2)  # 圆心
            return True
        return False

    def define_sector(self, sector, premise, theorem, root=True):  # 扇形
        if self.conditions.add(sector, cType.sector, premise, theorem):
            if root:
                premise = [self.conditions.get_index(sector, cType.sector)]
            self.define_angle(sector[2] + sector[0] + sector[1], premise, -2, False)  # 构成扇形的角
            self.define_arc(sector[1:3], premise, -2, False)  # 扇形的弧
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
            self.conditions.add(triangle[::-1], cType.right_triangle, premise, -2)  # 两种表示
            self.define_triangle(triangle, premise, -2, False)  # RT三角形也是普通三角形
            return True
        return False

    def define_isosceles_triangle(self, triangle, premise, theorem, root=True):  # 等腰三角形
        if self.conditions.add(triangle, cType.isosceles_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.isosceles_triangle)]
            self.conditions.add(triangle[0] + triangle[2] + triangle[1], cType.isosceles_triangle, premise, -2)  # 两种表示
            self.define_triangle(triangle, premise, -2, False)  # 等腰三角形也是普通三角形
            return True
        return False

    def define_regular_triangle(self, triangle, premise, theorem, root=True):  # 正三角形
        if self.conditions.add(triangle, cType.regular_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(triangle, cType.regular_triangle)]
            for triangle in rep.shape(triangle):
                self.conditions.add(triangle, cType.regular_triangle, premise, -2)
                self.define_isosceles_triangle(triangle, premise, -2, False)  # 等边也是等腰
            return True
        return False

    def define_quadrilateral(self, shape, premise, theorem, root=True):  # 四边形
        if self.conditions.add(shape, cType.quadrilateral, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.quadrilateral)]

            for quadrilateral in rep.shape(shape):
                self.conditions.add(quadrilateral, cType.quadrilateral, premise, -2)
                self.define_angle(quadrilateral[0:3], premise, -2, False)  # 四边形由角组成
            return True
        return False

    def define_trapezoid(self, shape, premise, theorem, root=True):  # 梯形
        if self.conditions.add(shape, cType.trapezoid, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.trapezoid)]

            self.conditions.add(shape[2] + shape[3] + shape[0] + shape[1], cType.trapezoid, premise, -2)
            self.define_quadrilateral(shape, premise, -2, False)  # 梯形也是四边形
            return True
        return False

    def define_isosceles_trapezoid(self, shape, premise, theorem, root=True):  # 等腰梯形
        if self.conditions.add(shape, cType.isosceles_trapezoid, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.isosceles_trapezoid)]

            self.conditions.add(shape[2] + shape[3] + shape[0] + shape[1], cType.isosceles_trapezoid, premise, -2)
            self.define_trapezoid(shape, premise, -2, False)  # 等腰梯形也是梯形
            return True
        return False

    def define_parallelogram(self, shape, premise, theorem, root=True):  # 平行四边形
        if self.conditions.add(shape, cType.parallelogram, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.parallelogram)]

            for parallelogram in rep.shape(shape):
                self.conditions.add(parallelogram, cType.parallelogram, premise, -2)
                self.define_isosceles_trapezoid(parallelogram, premise, -2, False)  # 平行四边形也是梯形
            return True
        return False

    def define_rectangle(self, shape, premise, theorem, root=True):  # 长方形
        if self.conditions.add(shape, cType.rectangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.rectangle)]
            for rectangle in rep.shape(shape):
                self.conditions.add(cType.rectangle, rectangle, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 长方形也是平行四边形
            return True
        return False

    def define_kite(self, shape, premise, theorem, root=True):  # 风筝形
        if self.conditions.add(shape, cType.kite, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.kite)]
            self.conditions.add(shape[2] + shape[3] + shape[0] + shape[1], cType.kite, premise, -2)
            self.define_quadrilateral(shape, premise, -2, False)  # Kite也是四边形
            return True
        return False

    def define_rhombus(self, shape, premise, theorem, root=True):  # 菱形
        if self.conditions.add(shape, cType.rhombus, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.rhombus)]
            for rhombus in rep.shape(shape):
                self.conditions.add(rhombus, cType.rhombus, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 菱形也是平行四边形
            self.define_kite(shape, premise, -2, False)  # 菱形也是Kite
            return True
        return False

    def define_square(self, shape, premise, theorem, root=True):  # 正方形
        if self.conditions.add(shape, cType.square, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.square)]
            for square in rep.shape(shape):
                self.conditions.add(square, cType.square, premise, -2)
            self.define_rectangle(shape, premise, -2, False)  # 正方形也是长方形
            self.define_rhombus(shape, premise, -2, False)  # 正方形也是菱形
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

    def define_regular_polygon(self, shape, premise, theorem, root=True):  # 正多边形
        if self.conditions.add(shape, cType.regular_polygon, premise, theorem):
            if root:
                premise = [self.conditions.get_index(shape, cType.regular_polygon)]

            for polygon in rep.shape(shape):
                self.conditions.add(polygon, cType.regular_polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            self.define_polygon(shape, premise, -2, False)  # 正多边形也是多边形
            return True
        return False

    """------------define Relation------------"""

    def define_collinear(self, points, premise, theorem):
        if len(points) > 2 and self.conditions.add(points, cType.collinear, premise, theorem):
            return True
        return False

    def define_point_on_line(self, ordered_pair, premise, theorem, root=True):  # 点在线上
        point, line = ordered_pair
        if self.conditions.add(ordered_pair, cType.point_on_line, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.point_on_line)]
            self.conditions.add((point, line[::-1]), cType.point_on_line, premise, theorem)  # 点在线上两种表示
            self.define_point(point, premise, -2)  # 定义点和线
            self.define_line(line, premise, -2, False)
            if point != line[0] and point != line[1]:
                self.define_line(line[0] + point, premise, -2, False)  # 定义子线段
                self.define_line(line[1] + point, premise, -2, False)
            return True
        return False

    def define_point_on_arc(self, ordered_pair, premise, theorem, root=True):  # 点在弧上
        point, arc = ordered_pair
        if self.conditions.add(ordered_pair, cType.point_on_arc, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.point_on_arc)]
            self.define_point(point, premise, -2)  # 定义点和弧
            self.define_arc(arc, premise, -2, False)
            return True
        return False

    def define_point_on_circle(self, ordered_pair, premise, theorem, root=True):  # 点在圆上
        point, circle = ordered_pair
        if self.conditions.add(ordered_pair, cType.point_on_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.point_on_circle)]
            self.define_point(point, premise, -2)  # 定义点和弧
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_midpoint(self, ordered_pair, premise, theorem, root=True):  # 中点
        point, line = ordered_pair
        if self.conditions.add(ordered_pair, cType.midpoint, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.midpoint)]
            self.conditions.add((point, line[::-1]), cType.midpoint, premise, -2)  # 中点有两中表示形式
            self.define_point(point, premise, -2)  # 定义点和弧
            self.define_line(line, premise, -2, False)
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

    def define_parallel(self, ordered_pair, premise, theorem, root=True):  # 线平行
        line1, line2, = ordered_pair
        if self.conditions.add(ordered_pair, cType.parallel, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.parallel)]

            for parallel in rep.parallel(ordered_pair):  # 平行的4种表示
                self.conditions.add(parallel, cType.parallel, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            return True
        return False

    def define_intersect_line_line(self, ordered_pair, premise, theorem, root=True):  # 线相交
        point, line1, line2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.intersect, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.intersect)]

            for all_form in rep.intersect(ordered_pair):  # 相交有4种表示
                self.conditions.add(all_form, cType.intersect, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            self.define_point_on_line((point, line1), premise, -2, False)  # 定义点在线上
            self.define_point_on_line((point, line2), premise, -2, False)
            return True
        return False

    def define_perpendicular(self, ordered_pair, premise, theorem, root=True):
        point, line1, line2 = ordered_pair

        if self.conditions.add(ordered_pair, cType.perpendicular, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.perpendicular)]

            for all_form in rep.intersect(ordered_pair):
                self.conditions.add(all_form, cType.perpendicular, premise, -2)  # 垂直有4种表示

            self.define_intersect_line_line(ordered_pair, premise, -2, False)  # 垂直也是相交

            sym = []
            if len(set(point + line1 + line2)) == 3:
                if line1[0] == line2[1]:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2 + line1[1])))
                elif line1[1] == line2[1]:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1 + line2[0])))
                elif line1[1] == line2[0]:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2[::-1] + line1[0])))
                elif line1[0] == line2[0]:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1[::-1] + line2[1])))
            elif len(set(point + line1 + line2)) == 4:
                if line2[1] == point:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1[0] + line2[::-1])))
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2 + line1[1])))
                elif line1[1] == point:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2[1] + line1[::-1])))
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1 + line2[0])))
                elif line2[0] == point:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1[1] + line2)))
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2[::-1] + line1[0])))
                elif line1[0] == point:
                    sym.append(self.get_sym_of_attr((aType.DA.name, line2[0] + line1)))
                    sym.append(self.get_sym_of_attr((aType.DA.name, line1[::-1] + line2[1])))
            elif len(set(point + line1 + line2)) == 5:
                sym.append(self.get_sym_of_attr((aType.DA.name, line1[0] + point + line2[0])))
                sym.append(self.get_sym_of_attr((aType.DA.name, line2[0] + point + line1[1])))
                sym.append(self.get_sym_of_attr((aType.DA.name, line1[1] + point + line2[1])))
                sym.append(self.get_sym_of_attr((aType.DA.name, line2[1] + point + line1[0])))

            for s in sym:  # 设置直角为90°
                self.set_value_of_sym(s, 90, premise, -2)

            return True
        return False

    def define_perpendicular_bisector(self, ordered_pair, premise, theorem, root=True):  # 垂直平分
        point, line1, line2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.perpendicular_bisector, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.perpendicular_bisector)]
            self.define_perpendicular((point, line1, line2), premise, -2, False)  # 垂直平分也是垂直
            return True
        return False

    def define_bisects_angle(self, ordered_pair, premise, theorem, root=True):  # 角平分线
        line, angle = ordered_pair
        if self.conditions.add(ordered_pair, cType.bisects_angle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.bisects_angle)]
            self.define_angle(angle, premise, -2, False)  # 定义角和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_disjoint_line_circle(self, ordered_pair, premise, theorem, root=True):  # 线圆相离
        line, circle = ordered_pair
        if self.conditions.add(ordered_pair, cType.disjoint_line_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.disjoint_line_circle)]
            self.define_line(line, premise, -2, False)  # 定义和线圆
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_disjoint_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 圆圆相离
        circle1, circle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.disjoint_circle_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.disjoint_circle_circle)]
            self.conditions.add((circle2, circle1), cType.disjoint_circle_circle, premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            return True
        return False

    def define_tangent_line_circle(self, ordered_pair, premise, theorem, root=True):  # 相切
        point, line, circle = ordered_pair
        if self.conditions.add(ordered_pair, cType.tangent_line_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.tangent_line_circle)]
            self.define_line(line, premise, -2, False)  # 定义线和圆
            self.define_circle(circle, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.define_point(point, premise, -2)
                self.define_point_on_line((point, line), premise, -2, False)
                self.define_point_on_circle((point, circle), premise, -2, False)
            return True
        return False

    def define_tangent_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 相切
        point, circle1, circle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.tangent_circle_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.tangent_circle_circle)]
            self.conditions.add((point, circle2, circle1), cType.tangent_circle_circle, premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.define_point(point, premise, -2)
                self.define_point_on_circle((point, circle1), premise, -2, False)
                self.define_point_on_circle((point, circle2), premise, -2, False)
            return True
        return False

    def define_intersect_line_circle(self, ordered_pair, premise, theorem, root=True):  # 相交
        point1, point2, line, circle = ordered_pair
        if self.conditions.add(ordered_pair, cType.intersect_line_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.intersect_line_circle)]
            self.define_line(line, premise, -2, False)  # 定义线
            self.define_circle(circle, premise, -2, False)  # 定义圆
            if point1 != "$":  # 如果给出交点
                self.define_point(point1, premise, -2)
                self.define_point_on_line((point1, line), premise, -2, False)
                self.define_point_on_circle((point1, circle), premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.define_point(point2, premise, -2)
                self.define_point_on_line((point2, line), premise, -2, False)
                self.define_point_on_circle((point2, circle), premise, -2, False)
            return True
        return False

    def define_intersect_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 相交
        point1, point2, circle1, circle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.intersect_circle_circle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.intersect_circle_circle)]
            self.conditions.add((point2, point1, circle2, circle1), cType.intersect_circle_circle, premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point1 != "$":  # 如果给出交点
                self.define_point(point1, premise, -2)
                self.define_point_on_circle((point1, circle1), premise, -2, False)
                self.define_point_on_circle((point1, circle2), premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.define_point(point2, premise, -2)
                self.define_point_on_circle((point2, circle1), premise, -2, False)
                self.define_point_on_circle((point2, circle2), premise, -2, False)
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

    def define_height_triangle(self, ordered_pair, premise, theorem, root=True):  # 高
        height, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.height_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.height_triangle)]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_height_trapezoid(self, ordered_pair, premise, theorem, root=True):  # 高
        height, trapezoid = ordered_pair
        if self.conditions.add(ordered_pair, cType.height_trapezoid, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.height_trapezoid)]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_trapezoid(trapezoid, premise, -2, False)
            return True
        return False

    def define_internally_tangent(self, ordered_pair, premise, theorem, root=True):  # 内切 circle2是大的
        point, circle1, circle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.internally_tangent, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.internally_tangent)]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            if point != "$":
                self.define_point(point, premise, -2)
                self.define_point_on_circle((point, circle1), premise, -2, False)
                self.define_point_on_circle((point, circle2), premise, -2, False)
            return True
        return False

    def define_contain(self, ordered_pair, premise, theorem, root=True):  # 内含 circle2是大的
        circle1, circle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.contain, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.contain)]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            return True
        return False

    def define_circumscribed_to_triangle(self, ordered_pair, premise, theorem, root=True):  # 外接圆
        circle, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.circumscribed_to_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.circumscribed_to_triangle)]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_inscribed_in_triangle(self, ordered_pair, premise, theorem, root=True):
        point1, point2, point3, circle, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.inscribed_in_triangle, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.inscribed_in_triangle)]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            if point1 != "$":
                self.define_point(point1, premise, -2)
                self.define_point_on_line((point1, triangle[0:2]), premise, -2, False)
                self.define_point_on_circle((point1, circle), premise, -2, False)
            if point2 != "$":
                self.define_point(point2, premise, -2)
                self.define_point_on_line((point2, triangle[1:3]), premise, -2, False)
                self.define_point_on_circle((point2, circle), premise, -2, False)
            if point3 != "$":
                self.define_point(point3, premise, -2)
                self.define_point_on_line((point3, triangle[2] + triangle[0]), premise, -2, False)
                self.define_point_on_circle((point3, circle), premise, -2, False)
            return True
        return False

    def define_congruent(self, ordered_pair, premise, theorem, root=True):  # 全等
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.congruent, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.congruent)]
            triangle1_all = rep.shape(triangle1)  # 6种
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):
                self.conditions.add((triangle1_all[i], triangle2_all[i]), cType.congruent, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_similar(self, ordered_pair, premise, theorem, root=True):  # 相似
        triangle1, triangle2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.similar, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.similar)]
            triangle1_all = rep.shape(triangle1)  # 6种表示方式
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):
                self.conditions.add((triangle1_all[i], triangle2_all[i]), cType.similar, premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_chord(self, ordered_pair, premise, theorem, root=True):  # 弦
        line, circle = ordered_pair
        if self.conditions.add(ordered_pair, cType.chord, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.chord)]
            self.conditions.add((line[::-1], circle), cType.chord, premise, -2)  # 两种表示
            self.define_line(line, premise, -2, False)  # 定义实体
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    """------------Equation------------"""

    def define_equation(self, equation, equation_type, premise, theorem):  # 定义方程
        if equation_type is eType.basic and equation not in self.basic_equations.keys():
            self.basic_equations[equation] = equation  # 由题目条件和构图语句得到的方程
            self.equation_solved = False
        elif equation_type is eType.theorem and equation not in self.theorem_equations.keys():
            self.theorem_equations[equation] = equation  # 应用定理时得到的方程
            self.equation_solved = False

        return self.conditions.add(equation, cType.equation, premise, theorem)  # 返回是否添加了新条件

    """------------Attr's Symbol------------"""

    def get_sym_of_attr(self, attr):
        if attr[0] == aType.T.name:  # 表示目标/中间值类型的符号，不用存储在符号库
            return symbols(attr[0].lower() + "_" + attr[1])

        if attr not in self.sym_of_attr.keys():  # 若无符号，新建符号
            sym = symbols(attr[0].lower() + "_" + attr[1].lower(), positive=True)  # 属性值没有负数
            self.sym_of_attr[attr] = sym  # 符号
            self.value_of_sym[sym] = None  # 值

            # 图形有多种表示形式的
            if attr[0] == aType.LL.name \
                    or attr[0] == aType.PT.name \
                    or attr[0] == aType.PQ.name \
                    or attr[0] == aType.PP.name \
                    or attr[0] == aType.AT.name \
                    or attr[0] == aType.AQ.name \
                    or attr[0] == aType.AP.name:
                for all_form in rep.shape(attr[1]):
                    self.sym_of_attr[(attr[0], all_form)] = sym

        else:  # 有符号就返回符号
            sym = self.sym_of_attr[attr]

        return sym

    def set_value_of_sym(self, sym, value, premise, theorem):  # 设置符号的值
        if self.value_of_sym[sym] is None:
            self.value_of_sym[sym] = value
            self.define_equation(sym - value, eType.value, premise, theorem)
        return False


class Problem(ProblemLogic):

    def __init__(self, problem_index, construction_fls, text_fls, image_fls, theorem_seqs, answer):
        super().__init__()
        self.problem_index = problem_index
        self.construction_fls = construction_fls
        self.text_fls = text_fls
        self.image_fls = image_fls
        self.theorem_seqs = theorem_seqs
        self.answer = answer

    """------------构图辅助函数------------"""

    def find_all_triangle(self):  # 通过三角形，找到所有的三角形/四边形
        update = True
        while update:
            update = False
            for tri1 in self.conditions.items[cType.triangle]:
                for tri2 in self.conditions.items[cType.triangle]:
                    if tri1[0] == tri2[0] and tri1[2] == tri2[1]:  # 两个相邻三角形拼接为更大的图形
                        area1 = self.get_sym_of_attr((aType.AT.name, tri1))  # 相邻图形面积相加构成更大面积
                        area2 = self.get_sym_of_attr((aType.AT.name, tri2))
                        break_coll = False
                        for coll in self.conditions.items[cType.collinear]:  # 遍历所有共线
                            if tri1[1] in coll and tri1[2] in coll and tri2[2] in coll:
                                update = self.define_triangle(tri1[0:2] + tri2[2], [-1], -1) or update
                                area3 = self.get_sym_of_attr((aType.AT.name, tri1[0:2] + tri2[2]))
                                self.define_equation(area1 + area2 - area3, eType.basic, [-1], -1)
                                break_coll = True
                                break
                            elif tri1[1] in coll and tri1[0] in coll and tri2[2] in coll:
                                update = self.define_triangle(tri1[1:3] + tri2[2], [-1], -1) or update
                                area3 = self.get_sym_of_attr((aType.AT.name, tri1[1:3] + tri2[2]))
                                self.define_equation(area1 + area2 - area3, eType.basic, [-1], -1)
                                break_coll = True
                                break
                        if not break_coll:  # 没有共线的，是四边形
                            update = self.define_quadrilateral(tri1 + tri2[2], [-1], -1) or update
                            area3 = self.get_sym_of_attr((aType.AQ.name, tri1 + tri2[2]))
                            self.define_equation(area1 + area2 - area3, eType.basic, [-1], -1)

    def angle_representation_alignment(self):  # 使角的表示符号一致
        for angle in self.conditions.items[cType.angle]:
            if (aType.DA.name, angle) in self.sym_of_attr.keys():  # 有符号了就不用再赋予了
                continue

            coll_a = None  # 与AO共线的点
            coll_b = None  # 与OB共线的点
            for coll in self.conditions.items[cType.collinear]:
                if angle[0] in coll and angle[1] in coll:
                    coll_a = coll
                if angle[1] in coll and angle[2] in coll:
                    coll_b = coll

            sym = self.get_sym_of_attr((aType.DA.name, angle))
            a_points = angle[0]
            o_point = angle[1]
            b_points = angle[2]
            if coll_a is not None:  # 与AO共线的点
                a_index = coll_a.find(angle[0])
                o_index = coll_a.find(angle[1])
                a_points = coll_a[0:a_index + 1] if a_index < o_index else coll_a[a_index:len(coll_a)]
            if coll_b is not None:  # 与OB共线的点
                o_index = coll_b.find(angle[1])
                b_index = coll_b.find(angle[2])
                b_points = coll_b[0:b_index + 1] if b_index < o_index else coll_b[b_index:len(coll_b)]

            for a_point in a_points:  # 本质上相同的角安排一样的符号
                for b_point in b_points:
                    self.sym_of_attr[(aType.DA.name, a_point + o_point + b_point)] = sym

    def find_all_angle_addition(self):  # 所有的角的相加关系
        for angle1 in self.conditions.items[cType.angle]:
            for angle2 in self.conditions.items[cType.angle]:
                if angle1[0] == angle2[2] and angle1[1] == angle2[1]:
                    angle3 = angle2[0:2] + angle1[2]
                    sym1 = self.get_sym_of_attr((aType.DA.name, angle1))
                    sym2 = self.get_sym_of_attr((aType.DA.name, angle2))
                    sym3 = self.get_sym_of_attr((aType.DA.name, angle3))
                    self.define_equation(sym1 + sym2 - sym3, eType.basic, [-1], -1)

    def find_all_line_addition(self):  # 所有共线边长度的相加关系、平角大小
        for coll in self.conditions.items[cType.collinear]:
            premise = [self.conditions.get_index(coll, cType.collinear)]
            for i in range(0, len(coll) - 2):
                for j in range(i + 1, len(coll) - 1):
                    for k in range(j + 1, len(coll)):
                        sym1 = self.get_sym_of_attr((aType.LL.name, coll[i] + coll[j]))
                        sym2 = self.get_sym_of_attr((aType.LL.name, coll[j] + coll[k]))
                        sym3 = self.get_sym_of_attr((aType.LL.name, coll[i] + coll[k]))
                        sym_of_angle = self.get_sym_of_attr((aType.DA.name, coll[i] + coll[j] + coll[k]))
                        self.set_value_of_sym(sym_of_angle, 180, premise, -2)  # 平角为180°
                        self.define_equation(sym1 + sym2 - sym3, eType.basic, premise, -2)  # 共线边长度的相加关系

    """------------解方程相关------------"""

    @func_set_timeout(15)  # 限时15s
    def solve_equations(self, target_sym=None, target_equation=None):  # 求解方程
        self.simplify_equations()  # solve前先化简方程
        if target_sym is None:  # 只涉及basic、value、theorem
            if self.equation_solved:  # basic、theorem没有更新，不用重复求解
                return

            equations = list(self.basic_equations.values()) + list(self.theorem_equations.values())
            # sym_set = []
            # for equation in equations:
            #     sym_set += equation.free_symbols
            # print(list(set(sym_set)))
            # a = input("")
            #
            # update = True
            # while update:    # 循环求解变量值
            #     update = False
            #     self.simplify_equations()  # solve前先化简方程
            #     for sym in list(set(sym_set)):  # 快速去重
            #         if self.value_of_sym[sym] is None:  # 符号值未知，尝试求解
            #             min_equations, premise = self.get_minimum_equations(set(), sym)
            #             print(sym)
            #             for i in min_equations:
            #                 print(i)
            #             print()
            #             solved_result = solve(min_equations)  # 求解min_equations
            #             if len(solved_result) > 0:  # 有解
            #                 if isinstance(solved_result, list):  # 解不唯一，选第一个(涉及三角函数时可能有多个解)
            #                     solved_result = solved_result[0]
            #                 if sym in solved_result.keys() and isinstance(solved_result[sym], Float):  # sym有实数解
            #                     self.set_value_of_sym(sym, solved_result[sym], premise, -3)
            #                     update = True

            solved_result = solve(equations)  # 求解equations
            # for i in equations:
            #     print(i)
            # print(solved_result)
            # print("???")
            if len(solved_result) == 0:  # 没有解，返回(一般都是有解的)
                return
            if isinstance(solved_result, list):  # 解不唯一，选第一个(涉及三角函数时可能有多个解)
                solved_result = solved_result[0]

            saved_results = []
            for sym in solved_result.keys():  # 遍历所有的解
                if isinstance(solved_result[sym], Float) and self.value_of_sym[sym] is None:  # 有新解，且解是实数
                    _, premise = self.get_minimum_equations(set(), sym)  # 得到求解sym的值所需要的最小方程组
                    saved_results.append([sym, solved_result[sym], premise])

            for saved_result in saved_results:  # 保存求解结果
                self.set_value_of_sym(saved_result[0], saved_result[1], saved_result[2], -3)

            self.theorem_equations = {}  # 清空 theorem_equations
            self.equation_solved = True  # 更新方程求解状态
        else:  # 求解target
            equations, premise = self.get_minimum_equations({target_sym}, target_equation)  # 使用value + basic
            equations.append(target_equation)
            solved_result = solve(equations)  # 求解target+value+basic equation
            if len(solved_result) > 0 and isinstance(solved_result, list):  # 若解不唯一，选择第一个
                solved_result = solved_result[0]

            if len(solved_result) > 0 and \
                    target_sym in solved_result.keys() and \
                    isinstance(solved_result[target_sym], Float):
                return float(solved_result[target_sym]), premise  # 有实数解，返回解

            return None, None  # 无解，返回None

    def simplify_equations(self):  # 化简basic、theorem equation
        update = True
        while update:
            update = False
            remove_lists = []  # 要删除equation列表
            for key in self.basic_equations.keys():
                for sym in self.basic_equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if self.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        self.basic_equations[key] = self.basic_equations[key].subs(sym, self.value_of_sym[sym])

                if len(self.basic_equations[key].free_symbols) == 0:  # 没有未知变量，删除这个方程
                    remove_lists.append(key)

                if len(self.basic_equations[key].free_symbols) == 1:  # 化简后只剩一个符号，自然求得符号值
                    target_sym = list(self.basic_equations[key].free_symbols)[0]
                    value = solve(self.basic_equations[key])[0]
                    premise = [self.conditions.get_index(key, cType.equation)]    # 前提
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                    self.set_value_of_sym(target_sym, value, premise, -3)
                    remove_lists.append(key)
                    update = True    # 得到了新的sym值，需要再次循环替换掉basic中此sym的值

            for remove_eq in remove_lists:  # 如果方程符号值都是已知的，删除这个方程
                self.basic_equations.pop(remove_eq)

            remove_lists = []  # 要删除equation列表
            for key in self.theorem_equations.keys():
                for sym in self.theorem_equations[key].free_symbols:  # 遍历方程中的符号，检查其值是否都是已知的
                    if self.value_of_sym[sym] is not None:  # sym值已知，替换掉
                        self.theorem_equations[key] = self.theorem_equations[key].subs(sym, self.value_of_sym[sym])

                if len(self.theorem_equations[key].free_symbols) == 0:  # 如果方程符号值都是已知的，删除这个方程
                    remove_lists.append(key)

                if len(self.theorem_equations[key].free_symbols) == 1:  # 化简后只剩一个符号，自然求得符号值
                    target_sym = list(self.theorem_equations[key].free_symbols)[0]
                    value = solve(self.theorem_equations[key])[0]
                    premise = [self.conditions.get_index(key, cType.equation)]  # 前提
                    for sym in key.free_symbols:
                        if self.value_of_sym[sym] is not None:
                            premise.append(self.conditions.get_index(sym - self.value_of_sym[sym], cType.equation))
                    self.set_value_of_sym(target_sym, value, premise, -3)
                    remove_lists.append(key)
                    update = True  # 得到了新的sym值，需要再次循环替换掉basic中此sym的值

            for remove_eq in remove_lists:  # 如果方程符号值都是已知的，删除这个方程
                self.theorem_equations.pop(remove_eq)

    def get_minimum_equations(self, target_sym, target_equation):  # 找到与求解目标方程相关的最小(basic、value)方程组
        sym_set = target_equation.free_symbols.difference(target_sym)  # 去掉求解的目标符号
        min_equations = []
        premise = []

        update = True
        while update:
            update = False
            for sym in sym_set:
                if self.value_of_sym[sym] is not None:  # sym的值已经求出，只用添加 equation:sym-value
                    equation = sym - self.value_of_sym[sym]
                    if equation not in min_equations:
                        min_equations.append(equation)  # 未知数方程经过了simplify，不可能含有已知量
                        premise.append(self.conditions.get_index(equation, cType.equation))  # 方程序号作为前提
                        update = True
                else:  # sym的值未求出，寻找basic和theorem中的最小方程组
                    for key in self.basic_equations.keys():
                        if sym in self.basic_equations[key].free_symbols and \
                                self.basic_equations[key] not in min_equations:  # basic方程包含sym
                            min_equations.append(self.basic_equations[key])
                            premise.append(self.conditions.get_index(key, cType.equation))
                            sym_set = sym_set.union(key.free_symbols)  # 添加basic方程可能会引入新符号
                            update = True
                    for key in self.theorem_equations.keys():
                        if sym in self.theorem_equations[key].free_symbols and \
                                self.theorem_equations[key] not in min_equations:  # theorem方程包含sym
                            min_equations.append(self.theorem_equations[key])
                            premise.append(self.conditions.get_index(key, cType.equation))
                            sym_set = sym_set.union(key.free_symbols)  # 添加basic方程可能会引入新符号
                            update = True

        return min_equations, premise

    def show_equations(self):
        """
        simplify_equations 核查完毕
        get_minimum_equations 未核查
        solve_equations 未核查
        show_equations 未编写
        """
        pass
    """------------辅助功能------------"""

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, theorem_seqs, answer):  # 新问题
        self.problem_index = problem_index
        self.construction_fls = construction_fls
        self.text_fls = text_fls
        self.image_fls = image_fls
        self.theorem_seqs = theorem_seqs
        self.answer = answer

        """------Entity, Entity Relation, Equation------"""
        self.conditions.clean()

        """------------symbols and equation------------"""
        self.sym_of_attr = {}  # (ConditionType, "name"): sym
        self.value_of_sym = {}  # sym: value
        self.basic_equations = {}
        self.theorem_equations = {}
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        """----------Target----------"""
        self.target_count = 0  # 目标个数
        self.target_type = []  # 解题目标的类型
        self.target = []  # 解题目标
        self.target_solved = []  # 条件求解情况
        self.answer = []  # 答案
        self.premise = []  # 前提条件集合

    def get_premise(self):  # 从结果往前遍历，找到所有所需条件
        for i in range(self.target_count):  # 所有解题目标需要的条件
            if self.target[i][3] is not None:
                self.premise += self.target[i][3]
        self.premise = list(set(self.premise))  # 快速去重

        while True:  # 向上遍历，寻找添加其他条件
            length = len(self.premise)
            for index in self.premise:
                if index != -1:
                    item = self.conditions.item_list[index]
                    self.premise = self.premise + self.conditions.get_premise(item[0], item[1])
            self.premise = list(set(self.premise))  # 快速去重
            if len(self.premise) == length:  # 如果没有更新，结束循环
                break

    def show(self):
        # Formal Language
        print("\033[36mproblem_index:\033[0m", end=" ")
        print(self.problem_index)
        print("\033[36mconstruction_fls:\033[0m")
        for construction_fl in self.construction_fls:  # 解析 formal language
            print(construction_fl)
        print("\033[36mtext_fls:\033[0m")
        for text_fl in self.text_fls:  # 解析 formal language
            print(text_fl)
        print("\033[36mimage_fls:\033[0m")
        for image_fl in self.image_fls:  # 解析 formal language
            print(image_fl)
        print("\033[36mtheorem_seqs:\033[0m", end=" ")
        for theorem in self.theorem_seqs:
            print(theorem, end=" ")
        print()

        self.get_premise()  # 生成条件树

        # Logic-Entity
        print("\033[33mEntity:\033[0m")
        for entity in Condition.entity_list:
            if len(self.conditions.items[entity]) > 0:
                print("{}:".format(entity.name))
                for item in self.conditions.items[entity]:
                    if self.conditions.get_index(item, entity) not in self.premise:
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
                    if self.conditions.get_index(item, entity_relation) not in self.premise:
                        print_str = "{0:^6}{1:^25}{2:^25}{3:^6}"
                    else:
                        print_str = "\033[35m{0:^6}{1:^25}{2:^25}{3:^6}\033[0m"
                    print(print_str.format(self.conditions.get_index(item, entity_relation),
                                           str(item),
                                           str(self.conditions.get_premise(item, entity_relation)),
                                           self.conditions.get_theorem(item, entity_relation)))
        # Logic-Attribution&Symbol&Equation
        print("\033[33mAttribution, Symbol and Value:\033[0m")
        for attr in self.sym_of_attr.keys():
            print("{0:^15}{1:^10}{2:^20}".format(str(attr),
                                                 str(self.sym_of_attr[attr]),
                                                 str(self.value_of_sym[self.sym_of_attr[attr]])))

        print("\033[33mEquations:\033[0m")
        if len(self.conditions.items[Condition.equation]) > 0:
            print("{}:".format(Condition.equation.name))
            for item in self.conditions.items[Condition.equation]:
                if self.conditions.get_index(item, Condition.equation) not in self.premise:
                    print_str = "{0:^6}{1:^65}{2:^25}{3:>6}"
                else:
                    print_str = "\033[35m{0:^6}{1:^65}{2:^25}{3:>6}\033[0m"
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
        for i in range(0, self.target_count):
            print("\033[34m{}:\033[0m {}".format(self.target_type[i].name, str(self.target[i])), end="  ")
            print("\033[34mcorrect answer:\033[0m {}".format(self.answer[i]), end="  ")
            if self.target_solved[i] == "solved":
                print("\033[32m{}\033[0m".format(self.target_solved[i]))
            else:
                print("\033[31m{}\033[0m".format(self.target_solved[i]))
