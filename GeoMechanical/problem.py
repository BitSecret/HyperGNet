from utility import Representation as rep
from facts import AttributionType as aType
from facts import EquationType as eType
from facts import Condition
from sympy import symbols, solve, Float
from func_timeout import func_set_timeout


class ProblemLogic:

    def __init__(self):
        """-------------Entity-------------"""
        self.point = Condition()  # 点、线、角、弧
        self.line = Condition()
        self.angle = Condition()
        self.arc = Condition()
        self.shape = Condition()  # 形状
        self.circle = Condition()  # 圆和扇形
        self.sector = Condition()
        self.triangle = Condition()  # 三角形
        self.right_triangle = Condition()
        self.isosceles_triangle = Condition()
        self.regular_triangle = Condition()
        self.quadrilateral = Condition()  # 四边形
        self.trapezoid = Condition()
        self.isosceles_trapezoid = Condition()
        self.parallelogram = Condition()
        self.rectangle = Condition()
        self.kite = Condition()
        self.rhombus = Condition()
        self.square = Condition()
        self.polygon = Condition()  # 多边形
        self.regular_polygon = Condition()
        self.entities = {"Point": self.point,
                         "Line": self.line,
                         "Angle": self.angle,
                         "Arc": self.arc,
                         "Shape": self.shape,
                         "Circle": self.circle,
                         "Sector": self.sector,
                         "Triangle": self.triangle,
                         "RightTriangle": self.right_triangle,
                         "IsoscelesTriangle": self.isosceles_triangle,
                         "RegularTriangle": self.regular_triangle,
                         "Quadrilateral": self.quadrilateral,
                         "Trapezoid": self.trapezoid,
                         "IsoscelesTrapezoid": self.isosceles_trapezoid,
                         "Parallelogram": self.parallelogram,
                         "Rectangle": self.rectangle,
                         "Kite": self.kite,
                         "Rhombus": self.rhombus,
                         "Square": self.square,
                         "Polygon": self.polygon,
                         "RegularPolygon": self.regular_polygon}

        """------------Positional Relation------------"""
        self.collinear = Condition()    # 共线
        self.point_on_line = Condition()  # 点的关系
        self.point_on_arc = Condition()
        self.point_on_circle = Condition()
        self.midpoint = Condition()
        self.circumcenter = Condition()
        self.incenter = Condition()
        self.centroid = Condition()
        self.orthocenter = Condition()
        self.parallel = Condition()  # 线的关系
        self.intersect = Condition()
        self.perpendicular = Condition()
        self.perpendicular_bisector = Condition()
        self.bisects_angle = Condition()
        self.disjoint_line_circle = Condition()
        self.disjoint_circle_circle = Condition()
        self.tangent_line_circle = Condition()
        self.tangent_circle_circle = Condition()
        self.intersect_line_circle = Condition()
        self.intersect_circle_circle = Condition()
        self.median = Condition()
        self.height_triangle = Condition()
        self.height_trapezoid = Condition()
        self.internally_tangent = Condition()  # 图形的关系
        self.contain = Condition()
        self.circumscribed_to_triangle = Condition()
        self.inscribed_in_triangle = Condition()
        self.congruent = Condition()
        self.similar = Condition()
        self.chord = Condition()
        self.relations = {"Collinear": self.collinear,
                          "PointOnLine": self.point_on_line,
                          "PointOnArc": self.point_on_arc,
                          "PointOnCircle": self.point_on_circle,
                          "Midpoint": self.midpoint,
                          "Circumcenter": self.circumcenter,
                          "Incenter": self.incenter,
                          "Centroid": self.centroid,
                          "Orthocenter": self.orthocenter,
                          "Parallel": self.parallel,
                          "Perpendicular": self.perpendicular,
                          "PerpendicularBisector": self.perpendicular_bisector,
                          "BisectsAngle": self.bisects_angle,
                          "DisjointLineCircle": self.disjoint_line_circle,
                          "DisjointCircleCircle": self.disjoint_circle_circle,
                          "TangentLineCircle": self.tangent_line_circle,
                          "TangentCircleCircle": self.tangent_circle_circle,
                          "IntersectLineLine": self.intersect,
                          "IntersectLineCircle": self.intersect_line_circle,
                          "IntersectCircleCircle": self.intersect_circle_circle,
                          "Median": self.median,
                          "HeightTriangle": self.height_triangle,
                          "HeightTrapezoid": self.height_trapezoid,
                          "InternallyTangent": self.internally_tangent,
                          "Contain": self.contain,
                          "CircumscribedToTriangle": self.circumscribed_to_triangle,
                          "InscribedInTriangle": self.inscribed_in_triangle,
                          "Congruent": self.congruent,
                          "Similar": self.similar,
                          "Chord": self.chord}

        """------------Algebraic Relation------------"""
        self.sym_of_attr = {}  # (ConditionType, "name"): sym
        self.value_of_sym = {}  # sym: value
        self.equations = Condition()  # 代数方程
        self.basic_equations = []
        self.value_equations = []
        self.theorem_equations = []
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        """----------Target----------"""
        self.target_count = 0  # 目标个数
        self.target_type = []  # 解题目标的类型
        self.target = []  # 解题目标
        self.target_solved = []  # 条件求解情况

    """------------define Entity------------"""
    def define_point(self, point, premise, theorem):  # 点
        return self.point.add(point, premise, theorem)

    def define_line(self, line, premise, theorem, root=True):  # 线
        if self.line.add(line, premise, theorem):
            if root:  # 如果是 definition tree 的根节点，那子节点都使用当前节点的 premise
                premise = [self.line.indexes[line]]
            self.line.add(line[::-1], premise, -2)  # 一条线2种表示
            self.point.add(line[0], premise, -2)  # 定义线上的点
            self.point.add(line[1], premise, -2)
            return True
        return False

    def define_angle(self, angle, premise, theorem, root=True):  # 角
        if self.angle.add(angle, premise, theorem):
            if root:
                premise = [self.angle.indexes[angle]]
            self.define_line(angle[0:2], premise, -2, False)  # 构成角的两个线
            self.define_line(angle[1:3], premise, -2, False)
            return True
        return False

    def define_arc(self, arc, premise, theorem, root=True):  # 弧
        if self.arc.add(arc, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.arc.indexes[arc]]
            self.point.add(arc[0], premise, -2)  # 构成弧的两个点
            self.point.add(arc[1], premise, -2)
            return True
        return False

    def define_shape(self, shape, premise, theorem, root=True):
        if self.shape.add(shape, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.shape.indexes[shape]]
            for point in shape:  # 添加构成shape的点。因为shape形状不确定，只能知道有这些点。
                self.point.add(point, premise, -2)
            shape_all = rep.shape(shape)  # n种表示
            for shape in shape_all:
                self.shape.add(shape, premise, -2)

            return True
        return False

    def define_circle(self, circle, premise, theorem, root=True):  # 圆
        if self.circle.add(circle, premise, theorem):
            if root:
                premise = [self.circle.indexes[circle]]
            self.point.add(circle, premise, -2)  # 圆心
            return True
        return False

    def define_sector(self, sector, premise, theorem, root=True):  # 扇形
        if self.sector.add(sector, premise, theorem):
            if root:
                premise = [self.sector.indexes[sector]]
            self.define_angle(sector[2] + sector[0] + sector[1], premise, -2, False)  # 构成扇形的角
            self.define_arc(sector[1:3], premise, -2, False)  # 扇形的弧
            return True
        return False

    def define_triangle(self, triangle, premise, theorem, root=True):  # 三角形
        if self.triangle.add(triangle, premise, theorem):
            if root:
                premise = [self.triangle.indexes[triangle]]

            for triangle in rep.shape(triangle):  # 所有表示形式
                self.triangle.add(triangle, premise, -2)
                self.define_angle(triangle, premise, -2, False)  # 定义3个角
            return True
        return False

    def define_right_triangle(self, triangle, premise, theorem, root=True):  # 直角三角形
        if self.right_triangle.add(triangle, premise, theorem):
            if root:
                premise = [self.right_triangle.indexes[triangle]]
            self.right_triangle.add(triangle[::-1], premise, -2)  # 两种表示
            self.define_triangle(triangle, premise, -2, False)  # RT三角形也是普通三角形
            return True
        return False

    def define_isosceles_triangle(self, triangle, premise, theorem, root=True):  # 等腰三角形
        if self.isosceles_triangle.add(triangle, premise, theorem):
            if root:
                premise = [self.isosceles_triangle.indexes[triangle]]
            self.isosceles_triangle.add(triangle[0] + triangle[2] + triangle[1], premise, -2)  # 两种表示
            self.define_triangle(triangle, premise, -2, False)  # 等腰三角形也是普通三角形
            return True
        return False

    def define_regular_triangle(self, triangle, premise, theorem, root=True):  # 正三角形
        if self.regular_triangle.add(triangle, premise, theorem):
            if root:
                premise = [self.regular_triangle.indexes[triangle]]
            triangle_all = rep.shape(triangle)  # 6种表示
            for triangle in triangle_all:
                self.regular_triangle.add(triangle, premise, -2)
                self.define_isosceles_triangle(triangle, premise, -2, False)  # 等边也是等腰
            return True
        return False

    def define_quadrilateral(self, shape, premise, theorem, root=True):  # 四边形
        if self.quadrilateral.add(shape, premise, theorem):
            if root:
                premise = [self.quadrilateral.indexes[shape]]
            quadrilateral_all = rep.shape(shape)  # 8种表示
            for quadrilateral in quadrilateral_all:
                self.quadrilateral.add(quadrilateral, premise, -2)
                self.define_angle(quadrilateral[0:3], premise, -2, False)  # 四边形由角组成
            return True
        return False

    def define_trapezoid(self, shape, premise, theorem, root=True):  # 梯形
        if self.trapezoid.add(shape, premise, theorem):
            if root:
                premise = [self.trapezoid.indexes[shape]]
            shape_inverse = shape[2] + shape[3] + shape[0] + shape[1]  # 一个梯形4种表示
            self.trapezoid.add(shape[::-1], premise, -2)
            self.trapezoid.add(shape_inverse, premise, -2)
            self.trapezoid.add(shape_inverse[::-1], premise, -2)
            self.define_quadrilateral(shape, premise, -2, False)  # 梯形也是四边形
            return True
        return False

    def define_isosceles_trapezoid(self, shape, premise, theorem, root=True):  # 等腰梯形
        if self.isosceles_trapezoid.add(shape, premise, theorem):
            if root:
                premise = [self.isosceles_trapezoid.indexes[shape]]
            shape_inverse = shape[2] + shape[3] + shape[0] + shape[1]  # 一个梯形4种表示
            self.isosceles_trapezoid.add(shape[::-1], premise, -2)
            self.isosceles_trapezoid.add(shape_inverse, premise, -2)
            self.isosceles_trapezoid.add(shape_inverse[::-1], premise, -2)
            self.define_trapezoid(shape, premise, -2, False)  # 等腰梯形也是梯形
            return True
        return False

    def define_parallelogram(self, shape, premise, theorem, root=True):  # 平行四边形
        if self.parallelogram.add(shape, premise, theorem):
            if root:
                premise = [self.parallelogram.indexes[shape]]
            parallelogram_all = rep.shape(shape)  # 8种表示
            for parallelogram in parallelogram_all:
                self.parallelogram.add(parallelogram, premise, -2)
            self.define_trapezoid(shape, premise, -2, False)  # 平行四边形也是梯形
            self.define_trapezoid(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_rectangle(self, shape, premise, theorem, root=True):  # 长方形
        if self.rectangle.add(shape, premise, theorem):
            if root:
                premise = [self.rectangle.indexes[shape]]
            rectangle_all = rep.shape(shape)  # 8种表示
            for rectangle in rectangle_all:
                self.rectangle.add(rectangle, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 长方形也是平行四边形
            self.define_isosceles_trapezoid(shape, premise, -2, False)  # 长方形也是等腰梯形
            self.define_isosceles_trapezoid(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_kite(self, shape, premise, theorem, root=True):  # 风筝形
        if self.kite.add(shape, premise, theorem):
            if root:
                premise = [self.kite.indexes[shape]]
            shape_inverse = shape[2] + shape[3] + shape[0] + shape[1]  # 4种表示
            self.kite.add(shape[::-1], premise, -2)
            self.kite.add(shape_inverse, premise, -2)
            self.kite.add(shape_inverse[::-1], premise, -2)
            self.define_quadrilateral(shape, premise, -2, False)  # Kite也是四边形
            return True
        return False

    def define_rhombus(self, shape, premise, theorem, root=True):  # 菱形
        if self.rhombus.add(shape, premise, theorem):
            if root:
                premise = [self.rhombus.indexes[shape]]
            rhombus_all = rep.shape(shape)  # 8种表示
            for rhombus in rhombus_all:
                self.rhombus.add(rhombus, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 菱形也是平行四边形
            self.define_kite(shape, premise, -2, False)  # 菱形也是Kite
            self.define_kite(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_square(self, shape, premise, theorem, root=True):  # 正方形
        if self.square.add(shape, premise, theorem):
            if root:
                premise = [self.square.indexes[shape]]
            square_all = rep.shape(shape)  # 8种表示
            for square in square_all:
                self.square.add(square, premise, -2)
            self.define_rectangle(shape, premise, -2, False)  # 正方形也是长方形
            self.define_rhombus(shape, premise, -2, False)  # 正方形也是菱形
            return True
        return False

    def define_polygon(self, shape, premise, theorem, root=True):  # 多边形
        if self.polygon.add(shape, premise, theorem):
            if root:
                premise = [self.polygon.indexes[shape]]
            polygon_all = rep.shape(shape)  # 所有表示
            for polygon in polygon_all:
                self.polygon.add(polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            return True
        return False

    def define_regular_polygon(self, shape, premise, theorem, root=True):  # 正多边形
        if self.regular_polygon.add(shape, premise, theorem):
            if root:
                premise = [self.regular_polygon.indexes[shape]]
            polygon_all = rep.shape(shape)  # 所有表示
            for polygon in polygon_all:
                self.regular_polygon.add(polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            self.define_polygon(shape, premise, -2, False)  # 正多边形也是多边形
            return True
        return False

    """------------define Relation------------"""
    def define_collinear(self, points, premise, theorem):
        if len(points) > 2 and self.collinear.add(points, premise, theorem):
            return True
        return False

    def define_point_on_line(self, ordered_pair, premise, theorem, root=True):  # 点在线上
        point, line = ordered_pair
        if self.point_on_line.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.point_on_line.indexes[ordered_pair]]
            self.point_on_line.add((point, line[::-1]), premise, theorem)  # 点在线上两种表示
            self.point.add(point, premise, -2)  # 定义点和线
            self.define_line(line, premise, -2, False)
            if point != line[0] and point != line[1]:
                self.define_line(line[0] + point, premise, -2, False)  # 定义子线段
                self.define_line(line[1] + point, premise, -2, False)
            return True
        return False

    def define_point_on_arc(self, ordered_pair, premise, theorem, root=True):  # 点在弧上
        point, arc = ordered_pair
        if self.point_on_arc.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.point_on_arc.indexes[ordered_pair]]
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_arc(arc, premise, -2, False)
            return True
        return False

    def define_point_on_circle(self, ordered_pair, premise, theorem, root=True):  # 点在圆上
        point, circle = ordered_pair
        if self.point_on_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.point_on_circle.indexes[ordered_pair]]
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_midpoint(self, ordered_pair, premise, theorem, root=True):  # 中点
        point, line = ordered_pair
        if self.midpoint.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.midpoint.indexes[ordered_pair]]
            self.midpoint.add((point, line[::-1]), premise, -2)  # 中点有两中表示形式
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_circumcenter(self, ordered_pair, premise, theorem, root=True):  # 外心
        point, triangle = ordered_pair
        if self.circumcenter.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.circumcenter.indexes[ordered_pair]]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.circumcenter.add((point, tri), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_incenter(self, ordered_pair, premise, theorem, root=True):  # 内心
        point, triangle = ordered_pair
        if self.incenter.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.incenter.indexes[ordered_pair]]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.incenter.add((point, tri), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_centroid(self, ordered_pair, premise, theorem, root=True):  # 重心
        point, triangle = ordered_pair
        if self.centroid.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.centroid.indexes[ordered_pair]]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.centroid.add((point, tri), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_orthocenter(self, ordered_pair, premise, theorem, root=True):  # 垂心
        point, triangle = ordered_pair
        if self.orthocenter.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.orthocenter.indexes[ordered_pair]]
            triangle_all = rep.shape(triangle)  # 一个三角形三种表示
            for tri in triangle_all:
                self.orthocenter.add((point, tri), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_parallel(self, ordered_pair, premise, theorem, root=True):  # 线平行
        line1, line2, = ordered_pair
        if self.parallel.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.parallel.indexes[ordered_pair]]

            for all_form in rep.parallel(ordered_pair):  # 平行的4种表示
                self.parallel.add(all_form, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            return True
        return False

    def define_intersect_line_line(self, ordered_pair, premise, theorem, root=True):  # 线相交
        point, line1, line2 = ordered_pair
        if self.intersect.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.intersect.indexes[ordered_pair]]

            for all_form in rep.intersect(ordered_pair):  # 相交有4种表示
                self.intersect.add(all_form, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            # if point != "$":  # 如果给出交点
            #     self.point.add(point, premise, -2)
            #     self.define_point_on_line((point, line1), premise, -2, False)
            #     self.define_point_on_line((point, line2), premise, -2, False)
            return True
        return False

    def define_perpendicular(self, ordered_pair, premise, theorem, root=True):
        point, line1, line2 = ordered_pair

        if self.perpendicular.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.perpendicular.indexes[ordered_pair]]

            for all_form in rep.intersect(ordered_pair):
                self.perpendicular.add(all_form, premise, -2)  # 垂直有4种表示

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

            for s in sym:    # 设置直角为90°
                self.set_value_of_sym(s, 90, [-1], -1)

            return True
        return False

    def define_perpendicular_bisector(self, ordered_pair, premise, theorem, root=True):  # 垂直平分
        point, line1, line2 = ordered_pair
        if self.perpendicular_bisector.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.perpendicular_bisector.indexes[ordered_pair]]
            self.perpendicular_bisector.add((point, line1[::-1], line2[::-1]), premise, -2)  # 垂直平分有2种表示
            self.define_perpendicular((point, line1, line2), premise, -2, False)  # 垂直平分也是垂直
            return True
        return False

    def define_bisects_angle(self, ordered_pair, premise, theorem, root=True):  # 角平分线
        line, angle = ordered_pair
        if self.bisects_angle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.bisects_angle.indexes[ordered_pair]]
            self.define_angle(angle, premise, -2, False)  # 定义角和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_disjoint_line_circle(self, ordered_pair, premise, theorem, root=True):  # 线圆相离
        line, circle = ordered_pair
        if self.disjoint_line_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.disjoint_line_circle.indexes[ordered_pair]]
            self.define_line(line, premise, -2, False)  # 定义和线圆
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_disjoint_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 圆圆相离
        circle1, circle2 = ordered_pair
        if self.disjoint_circle_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.disjoint_circle_circle.indexes[ordered_pair]]
            self.disjoint_circle_circle.add((circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            return True
        return False

    def define_tangent_line_circle(self, ordered_pair, premise, theorem, root=True):  # 相切
        point, line, circle = ordered_pair
        if self.tangent_line_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.tangent_line_circle.indexes[ordered_pair]]
            self.define_line(line, premise, -2, False)  # 定义线和圆
            self.define_circle(circle, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.point.add(point, premise, -2)
                self.define_point_on_line((point, line), premise, -2, False)
                self.define_point_on_circle((point, circle), premise, -2, False)
            return True
        return False

    def define_tangent_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 相切
        point, circle1, circle2 = ordered_pair
        if self.tangent_circle_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.tangent_circle_circle.indexes[ordered_pair]]
            self.tangent_circle_circle.add((point, circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.point.add(point, premise, -2)
                self.define_point_on_circle((point, circle1), premise, -2, False)
                self.define_point_on_circle((point, circle2), premise, -2, False)
            return True
        return False

    def define_intersect_line_circle(self, ordered_pair, premise, theorem, root=True):  # 相交
        point1, point2, line, circle = ordered_pair
        if self.intersect_line_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.intersect_line_circle.indexes[ordered_pair]]
            self.define_line(line, premise, -2, False)  # 定义线
            self.define_circle(circle, premise, -2, False)  # 定义圆
            if point1 != "$":  # 如果给出交点
                self.point.add(point1, premise, -2)
                self.define_point_on_line((point1, line), premise, -2, False)
                self.define_point_on_circle((point1, circle), premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.point.add(point2, premise, -2)
                self.define_point_on_line((point2, line), premise, -2, False)
                self.define_point_on_circle((point2, circle), premise, -2, False)
            return True
        return False

    def define_intersect_circle_circle(self, ordered_pair, premise, theorem, root=True):  # 相交
        point1, point2, circle1, circle2 = ordered_pair
        if self.intersect_circle_circle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.intersect_circle_circle.indexes[ordered_pair]]
            self.intersect_circle_circle.add((point2, point1, circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point1 != "$":  # 如果给出交点
                self.point.add(point1, premise, -2)
                self.define_point_on_circle((point1, circle1), premise, -2, False)
                self.define_point_on_circle((point1, circle2), premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.point.add(point2, premise, -2)
                self.define_point_on_circle((point2, circle1), premise, -2, False)
                self.define_point_on_circle((point2, circle2), premise, -2, False)
            return True
        return False

    def define_median(self, ordered_pair, premise, theorem, root=True):  # 中线
        line, triangle = ordered_pair
        if self.median.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.median.indexes[ordered_pair]]
            self.define_line(line, premise, -2)  # 定义实体
            self.define_triangle(triangle, premise, -2)
            self.define_midpoint((line[1], triangle[1:3]), premise, -2, False)  # 底边中点
            return True
        return False

    def define_height_triangle(self, ordered_pair, premise, theorem, root=True):  # 高
        height, triangle = ordered_pair
        if self.height_triangle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.height_triangle.indexes[ordered_pair]]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_height_trapezoid(self, ordered_pair, premise, theorem, root=True):  # 高
        height, trapezoid = ordered_pair
        if self.height_trapezoid.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.height_trapezoid.indexes[ordered_pair]]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_trapezoid(trapezoid, premise, -2, False)
            return True
        return False

    def define_internally_tangent(self, ordered_pair, premise, theorem, root=True):  # 内切 circle2是大的
        point, circle1, circle2 = ordered_pair
        if self.internally_tangent.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.internally_tangent.indexes[ordered_pair]]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            if point != "$":
                self.point.add(point, premise, -2)
                self.define_point_on_circle((point, circle1), premise, -2, False)
                self.define_point_on_circle((point, circle2), premise, -2, False)
            return True
        return False

    def define_contain(self, ordered_pair, premise, theorem, root=True):  # 内含 circle2是大的
        circle1, circle2 = ordered_pair
        if self.contain.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.contain.indexes[ordered_pair]]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            return True
        return False

    def define_circumscribed_to_triangle(self, ordered_pair, premise, theorem, root=True):  # 外接圆
        circle, triangle = ordered_pair
        if self.circumscribed_to_triangle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.circumscribed_to_triangle.indexes[ordered_pair]]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_inscribed_in_triangle(self, ordered_pair, premise, theorem, root=True):
        point1, point2, point3, circle, triangle = ordered_pair
        if self.inscribed_in_triangle.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.inscribed_in_triangle.indexes[ordered_pair]]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            if point1 != "$":
                self.point.add(point1, premise, -2)
                self.define_point_on_line((point1, triangle[0:2]), premise, -2, False)
                self.define_point_on_circle((point1, circle), premise, -2, False)
            if point2 != "$":
                self.point.add(point2, premise, -2)
                self.define_point_on_line((point2, triangle[1:3]), premise, -2, False)
                self.define_point_on_circle((point2, circle), premise, -2, False)
            if point3 != "$":
                self.point.add(point3, premise, -2)
                self.define_point_on_line((point3, triangle[2] + triangle[0]), premise, -2, False)
                self.define_point_on_circle((point3, circle), premise, -2, False)
            return True
        return False

    def define_congruent(self, ordered_pair, premise, theorem, root=True):  # 全等
        triangle1, triangle2 = ordered_pair
        if self.congruent.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.congruent.indexes[ordered_pair]]
            triangle1_all = rep.shape(triangle1)  # 6种
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):
                self.congruent.add((triangle1_all[i], triangle2_all[i]), premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_similar(self, ordered_pair, premise, theorem, root=True):  # 相似
        triangle1, triangle2 = ordered_pair
        if self.similar.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.similar.indexes[ordered_pair]]
            triangle1_all = rep.shape(triangle1)  # 6种表示方式
            triangle2_all = rep.shape(triangle2)
            for i in range(len(triangle1_all)):
                self.similar.add((triangle1_all[i], triangle2_all[i]), premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_chord(self, ordered_pair, premise, theorem, root=True):  # 弦
        line, circle = ordered_pair
        if self.chord.add(ordered_pair, premise, theorem):
            if root:
                premise = [self.chord.indexes[ordered_pair]]
            self.chord.add((line[::-1], circle), premise, -2)  # 两种表示
            self.define_line(line, premise, -2, False)  # 定义实体
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    """------------Equation------------"""
    def define_equation(self, equation, equation_type, premise, theorem):    # 定义方程
        if equation_type is eType.basic and equation not in self.basic_equations:
            self.basic_equations.append(equation)  # 由题目条件和构图语句得到的方程
        elif equation_type is eType.value and equation not in self.value_equations:
            self.value_equations.append(equation)  # 已经求解的值，使用方程表示，便于后续求解
        elif equation_type is eType.theorem and equation not in self.theorem_equations:
            self.theorem_equations.append(equation)  # 应用定理时得到的方程

        if self.equations.add(equation, premise, theorem) and equation_type is not eType.value:
            self.equation_solved = False    # solved类型的equation不会引入新的信息
            return True
        return False

    def simplify_basic_equations(self):    # 化简basic equation
        for equation in self.basic_equations:
            remove = True
            for sym in equation.free_symbols:    # 遍历方程中的符号，检查其值是否都是已知的
                remove = remove and self.value_of_sym[sym] is not None
            if remove:    # 如果都是已知的，删除这个方程
                self.basic_equations.remove(equation)

    def get_minimum_solvable_equations(self, target_equation, all_equations):    # 找到与求解目标方程相关的最小方程组
        sym_set = target_equation.free_symbols
        min_equations = [target_equation]
        premise = []
        update = True
        while update:
            update = False
            for equation in all_equations:
                if len(sym_set.intersection(equation.free_symbols)) > 0 and equation not in min_equations:  # 含有目标符号
                    min_equations.append(equation)
                    premise.append(self.equations.indexes[equation])
                    sym_set = sym_set.union(equation.free_symbols)
                    update = True
        return min_equations, premise

    """------------Attr's Symbol------------"""
    def get_sym_of_attr(self, attr):
        if attr[0] == aType.T.name:  # 表示目标/中间值类型的符号，不用存储在符号库
            return symbols(attr[0].lower() + "_" + attr[1])

        if attr not in self.sym_of_attr.keys():  # 若无符号，新建符号
            sym = symbols(attr[0].lower() + "_" + attr[1].lower(), positive=True)    # 属性值没有负数
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

    def set_value_of_sym(self, sym, value, premise, theorem):    # 设置符号的值
        if self.value_of_sym[sym] is None:
            self.value_of_sym[sym] = value
            self.define_equation(sym - value, eType.value, premise, theorem)
            return True
        return False


class Problem(ProblemLogic):

    def __init__(self, problem_index, construction_fls, text_fls, image_fls, theorem_seqs):
        super().__init__()
        self.problem_index = problem_index
        self.construction_fls = construction_fls
        self.text_fls = text_fls
        self.image_fls = image_fls
        self.theorem_seqs = theorem_seqs

    """------------构图辅助函数------------"""
    def find_all_triangle(self):  # 通过三角形，找到所有的三角形/四边形
        update = True
        while update:
            update = False
            for tri1 in self.triangle.items:
                for tri2 in self.triangle.items:
                    if tri1[0] == tri2[0] and tri1[2] == tri2[1]:    # 两个相邻三角形拼接为更大的图形
                        area1 = self.get_sym_of_attr((aType.AT.name, tri1))    # 相邻图形面积相加构成更大面积
                        area2 = self.get_sym_of_attr((aType.AT.name, tri2))
                        break_coll = False
                        for coll in self.collinear.items:    # 遍历所有共线
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
                        if not break_coll:    # 没有共线的，是四边形
                            update = self.define_quadrilateral(tri1 + tri2[2], [-1], -1) or update
                            area3 = self.get_sym_of_attr((aType.AQ.name, tri1 + tri2[2]))
                            self.define_equation(area1 + area2 - area3, eType.basic, [-1], -1)

    def angle_representation_alignment(self):    # 使角的表示符号一致
        for angle in self.angle.items:
            if (aType.DA.name, angle) in self.sym_of_attr.keys():    # 有符号了就不用再赋予了
                continue

            coll_a = None    # 与AO共线的点
            coll_b = None    # 与OB共线的点
            for coll in self.collinear.items:
                if angle[0] in coll and angle[1] in coll:
                    coll_a = coll
                if angle[1] in coll and angle[2] in coll:
                    coll_b = coll

            sym = self.get_sym_of_attr((aType.DA.name, angle))
            a_points = angle[0]
            o_point = angle[1]
            b_points = angle[2]
            if coll_a is not None:    # 与AO共线的点
                a_index = coll_a.find(angle[0])
                o_index = coll_a.find(angle[1])
                a_points = coll_a[0:a_index + 1] if a_index < o_index else coll_a[a_index:len(coll_a)]
            if coll_b is not None:    # 与OB共线的点
                o_index = coll_b.find(angle[1])
                b_index = coll_b.find(angle[2])
                b_points = coll_b[0:b_index + 1] if b_index < o_index else coll_b[b_index:len(coll_b)]

            for a_point in a_points:    # 本质上相同的角安排一样的符号
                for b_point in b_points:
                    self.sym_of_attr[(aType.DA.name, a_point + o_point + b_point)] = sym

    def find_all_angle_addition(self):    # 所有的角的相加关系
        for angle1 in self.angle.items:
            for angle2 in self.angle.items:
                if angle1[0] == angle2[2] and angle1[1] == angle2[1]:
                    angle3 = angle2[0:2] + angle1[2]
                    sym1 = self.get_sym_of_attr((aType.DA.name, angle1))
                    sym2 = self.get_sym_of_attr((aType.DA.name, angle2))
                    sym3 = self.get_sym_of_attr((aType.DA.name, angle3))
                    self.define_equation(sym1 + sym2 - sym3, eType.basic, [-1], -1)

    def find_all_line_addition(self):    # 所有共线边长度的相加关系、平角大小
        for coll in self.collinear.items:
            for i in range(0, len(coll) - 2):
                for j in range(i + 1, len(coll) - 1):
                    for k in range(j + 1, len(coll)):
                        # self.define_line(coll[i] + coll[j], [-1], -1)    # 不用再定义line，在前面三角形构图时已经define了
                        sym1 = self.get_sym_of_attr((aType.LL.name, coll[i] + coll[j]))
                        # self.define_line(coll[j] + coll[k], [-1], -1)
                        sym2 = self.get_sym_of_attr((aType.LL.name, coll[j] + coll[k]))
                        # self.define_line(coll[i] + coll[k], [-1], -1)
                        sym3 = self.get_sym_of_attr((aType.LL.name, coll[i] + coll[k]))
                        sym_of_angle = self.get_sym_of_attr((aType.DA.name, coll[i] + coll[j] + coll[k]))
                        self.set_value_of_sym(sym_of_angle, 180, [-1], -1)    # 平角为180°
                        self.define_equation(sym1 + sym2 - sym3, eType.basic, [-1], -1)    # 共线边长度的相加关系

    """------------解方程------------"""
    @func_set_timeout(15)  # 限时10s
    def solve_equations(self, target_sym=None, target_equation=None):    # 求解方程
        if target_sym is None:    # 只涉及basic、value、theorem
            if self.equation_solved:  # basic、theorem没有更新，不用重复求解
                return

            if len(self.theorem_equations) > 0:    # 求解basic+value+theorem
                equations = self.basic_equations + self.value_equations + self.theorem_equations
                premise = [-1]
                for eq in self.theorem_equations:
                    premise.append(self.equations.indexes[eq])
            else:    # 求解basic+value，一个题目中只有最开始会调用一次这个else
                equations = self.basic_equations + self.value_equations
                premise = [-1]

            solved_result = solve(equations)  # 求解equations
            if len(solved_result) == 0:  # 没有解，返回(一般都是有解的)
                return
            if isinstance(solved_result, list):  # 解不唯一，选第一个(涉及三角函数时可能有多个解)
                solved_result = solved_result[0]
            for sym in solved_result.keys():  # 遍历所有的解
                if isinstance(solved_result[sym], Float):  # 如果解是实数，保存
                    self.set_value_of_sym(sym, solved_result[sym], premise, -3)    # 保存求解结果

            self.theorem_equations = []    # 清空 theorem_equations
            self.simplify_basic_equations()    # 简化 basic_equations
            self.equation_solved = True    # 更新方程求解状态

        else:    # 求解target+value、target+value+basic
            equations, premise = self.get_minimum_solvable_equations(target_equation, self.value_equations)
            solved_result = solve(equations)
            if len(solved_result) > 0 and isinstance(solved_result, list):    # 若解不唯一，选择第一个
                solved_result = solved_result[0]

            # 若没有解，尝试添加basic后再求解
            if len(solved_result) == 0 or (len(solved_result) > 0 and target_sym not in solved_result.keys()):
                equations, premise = self.get_minimum_solvable_equations(target_equation,
                                                                         self.value_equations + self.basic_equations)
                solved_result = solve(equations)  # 求解target+value+basic equation
                if len(solved_result) > 0 and isinstance(solved_result, list):    # 若解不唯一，选择第一个
                    solved_result = solved_result[0]

            if len(solved_result) > 0 and \
                    target_sym in solved_result.keys() and \
                    isinstance(solved_result[target_sym], Float):
                return float(solved_result[target_sym]), premise  # 有实数解，返回解

            return None, None  # 无解，返回None

    """------------辅助功能------------"""
    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, theorem_seqs):  # 新问题
        self.problem_index = problem_index
        self.construction_fls = construction_fls
        self.text_fls = text_fls
        self.image_fls = image_fls
        self.theorem_seqs = theorem_seqs

        for key in self.entities.keys():
            self.entities[key].clean()

        for key in self.relations.keys():
            self.relations[key].clean()

        self.sym_of_attr = {}  # (ConditionType, "name"): sym
        self.value_of_sym = {}  # sym: value
        self.equations = Condition()  # 代数方程
        self.equation_solved = True  # 记录有没有求解过方程，避免重复计算

        self.target_count = 0  # 目标个数
        self.target_type = []  # 解题目标的类型
        self.target = []  # 解题目标
        self.target_solved = []  # 条件求解情况

    def show(self):
        # Formal Language
        print("\033[32mproblem_index:\033[0m", end=" ")
        print(self.problem_index)
        print("\033[32mconstruction_fls:\033[0m")
        for construction_fl in self.construction_fls:  # 解析 formal language
            print(construction_fl)
        print("\033[32mtext_fls:\033[0m")
        for text_fl in self.text_fls:  # 解析 formal language
            print(text_fl)
        print("\033[32mimage_fls:\033[0m")
        for image_fl in self.image_fls:  # 解析 formal language
            print(image_fl)

        print("\033[32mtheorem_seqs:\033[0m", end=" ")
        for theorem in self.theorem_seqs:
            print(theorem, end=" ")
        print()

        # Logic-Entity
        print("\033[33mEntities:\033[0m")
        for entity in self.entities.keys():
            if len(self.entities[entity].items) > 0:
                print("{}:".format(entity))
                for item in self.entities[entity].items:
                    print("{0:^6}{1:^15}{2:^25}{3:^6}".format(self.entities[entity].indexes[item],
                                                              item,
                                                              str(self.entities[entity].premises[item]),
                                                              self.entities[entity].theorems[item]))
        # Logic-Relation
        print("\033[33mRelations:\033[0m")
        for relation in self.relations.keys():
            if len(self.relations[relation].items) > 0:
                print("{}:".format(relation))
                for item in self.relations[relation].items:
                    print("{0:^6}{1:^25}{2:^25}{3:^6}".format(self.relations[relation].indexes[item],
                                                              str(item),
                                                              str(self.relations[relation].premises[item]),
                                                              self.relations[relation].theorems[item]))
        # Logic-Attribution&Symbol
        print("\033[33mSymbol Of Attr:\033[0m")
        for attr in self.sym_of_attr.keys():
            print(attr, end=": ")
            print(self.sym_of_attr[attr])
        print("\033[33mValue Of Symbol:\033[0m")
        for sym in self.value_of_sym.keys():
            print(sym, end=": ")
            print(self.value_of_sym[sym])
        print("\033[33mEquations:\033[0m")
        for equation in self.equations.items:
            print("{0:^6}{1:^40}{2:^25}{3:^6}".format(self.equations.indexes[equation],
                                                      str(equation),
                                                      str(self.equations.premises[equation]),
                                                      self.equations.theorems[equation]))

        # target
        print("\033[34mTarget Count:\033[0m", end=" ")
        print(self.target_count)
        for i in range(0, self.target_count):
            print("\033[34m{}:\033[0m  {}  ".format(self.target_type[i].name,
                                                    str(self.target[i])), end="")
            if self.target_solved[i] == "solved":
                print("\033[32m{}\033[0m".format(self.target_solved[i]))
            else:
                print("\033[31m{}\033[0m".format(self.target_solved[i]))
