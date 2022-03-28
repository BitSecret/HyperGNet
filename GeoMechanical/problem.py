from utility import get_all_representation_of_shape
from facts import Condition, ConditionType, AttributionSymbol, AttributionType, TargetType


class ProblemLogic:

    def __init__(self):
        """-------------Condition:Entity-------------"""
        self.point = Condition(ConditionType.Entity, "Point")  # 点、线、角、弧
        self.line = Condition(ConditionType.Entity, "Line")
        self.angle = Condition(ConditionType.Entity, "Angle")
        self.arc = Condition(ConditionType.Entity, "Arc")
        self.shape = Condition(ConditionType.Entity, "Shape")  # 形状
        self.circle = Condition(ConditionType.Entity, "Circle")  # 圆和扇形
        self.sector = Condition(ConditionType.Entity, "Sector")
        self.triangle = Condition(ConditionType.Entity, "Triangle")  # 三角形
        self.right_triangle = Condition(ConditionType.Entity, "Right Triangle")
        self.isosceles_triangle = Condition(ConditionType.Entity, "Isosceles Triangle")
        self.regular_triangle = Condition(ConditionType.Entity, "Regular Triangle")
        self.quadrilateral = Condition(ConditionType.Entity, "Quadrilateral")  # 四边形
        self.trapezoid = Condition(ConditionType.Entity, "Trapezoid")
        self.isosceles_trapezoid = Condition(ConditionType.Entity, "Isosceles Trapezoid")
        self.parallelogram = Condition(ConditionType.Entity, "Parallelogram")
        self.rectangle = Condition(ConditionType.Entity, "Rectangle")
        self.kite = Condition(ConditionType.Entity, "Kite")
        self.rhombus = Condition(ConditionType.Entity, "Rhombus")
        self.square = Condition(ConditionType.Entity, "Square")
        self.polygon = Condition(ConditionType.Entity, "Polygon")  # 多边形
        self.regular_polygon = Condition(ConditionType.Entity, "Regular Polygon")

        """------------Condition:Relation------------"""
        self.point_on_line = Condition(ConditionType.Relation, "Point On Line")  # 点的关系
        self.point_on_arc = Condition(ConditionType.Relation, "Point On Arc")
        self.point_on_circle = Condition(ConditionType.Relation, "Point On Circle")
        self.midpoint = Condition(ConditionType.Relation, "Midpoint")
        self.circumcenter = Condition(ConditionType.Relation, "Circumcenter")
        self.incenter = Condition(ConditionType.Relation, "Incenter")
        self.centroid = Condition(ConditionType.Relation, "Centroid")
        self.orthocenter = Condition(ConditionType.Relation, "Orthocenter")
        self.parallel = Condition(ConditionType.Relation, "Parallel")  # 线的关系
        self.intersect = Condition(ConditionType.Relation, "Intersect")
        self.perpendicular = Condition(ConditionType.Relation, "Perpendicular")
        self.perpendicular_bisector = Condition(ConditionType.Relation, "Perpendicular Bisector")
        self.bisects_angle = Condition(ConditionType.Relation, "Bisects Angle")
        self.disjoint_line_circle = Condition(ConditionType.Relation, "Disjoint Line Circle")
        self.disjoint_circle_circle = Condition(ConditionType.Relation, "Disjoint Circle Circle")
        self.tangent_line_circle = Condition(ConditionType.Relation, "Tangent Line Circle")
        self.tangent_circle_circle = Condition(ConditionType.Relation, "Tangent Circle Circle")
        self.intersect_line_circle = Condition(ConditionType.Relation, "Intersect Line Circle")
        self.intersect_circle_circle = Condition(ConditionType.Relation, "Intersect Circle Circle")
        self.median = Condition(ConditionType.Relation, "Median")
        self.height_triangle = Condition(ConditionType.Relation, "Height Triangle")
        self.height_trapezoid = Condition(ConditionType.Relation, "Height Trapezoid")
        self.internally_tangent = Condition(ConditionType.Relation, "Internally Tangent")  # 图形的关系
        self.contain = Condition(ConditionType.Relation, "Contain")
        self.circumscribed_to_triangle = Condition(ConditionType.Relation, "Circumscribed To Triangle")
        self.inscribed_in_triangle = Condition(ConditionType.Relation, "Inscribed In Triangle")
        self.congruent = Condition(ConditionType.Relation, "Congruent")
        self.similar = Condition(ConditionType.Relation, "Similar")

        """------------Condition:Expression------------"""
        self.expression = Condition(ConditionType.Expression, "Expression")

        """------Attribution's Symbol------"""
        self.length_of_line = AttributionSymbol(AttributionType.LengthOfLine)
        self.length_of_arc = AttributionSymbol(AttributionType.LengthOfArc)
        self.degree_of_angle = AttributionSymbol(AttributionType.DegreeOfAngle)
        self.degree_of_sector = AttributionSymbol(AttributionType.DegreeOfSector)
        self.radius_of_arc = AttributionSymbol(AttributionType.RadiusOfArc)
        self.radius_of_circle = AttributionSymbol(AttributionType.RadiusOfCircle)
        self.radius_of_sector = AttributionSymbol(AttributionType.RadiusOfSector)
        self.diameter_of_circle = AttributionSymbol(AttributionType.DiameterOfCircle)
        self.perimeter_of_triangle = AttributionSymbol(AttributionType.PerimeterOfTriangle)
        self.perimeter_of_circle = AttributionSymbol(AttributionType.PerimeterOfCircle)
        self.perimeter_of_sector = AttributionSymbol(AttributionType.PerimeterOfSector)
        self.perimeter_of_quadrilateral = AttributionSymbol(AttributionType.PerimeterOfQuadrilateral)
        self.perimeter_of_polygon = AttributionSymbol(AttributionType.PerimeterOfPolygon)
        self.area_of_triangle = AttributionSymbol(AttributionType.AreaOfTriangle)
        self.area_of_circle = AttributionSymbol(AttributionType.AreaOfCircle)
        self.area_of_sector = AttributionSymbol(AttributionType.AreaOfSector)
        self.area_of_quadrilateral = AttributionSymbol(AttributionType.AreaOfQuadrilateral)
        self.area_of_polygon = AttributionSymbol(AttributionType.AreaOfPolygon)

        """----------汇总----------"""
        self.target_type = None  # 解题目标的类型
        self.target = None  # 解题目标

    """------------define Entity------------"""

    def define_point(self, point, premise=-1, theorem=-1):  # 点
        return self.point.add(point, premise, theorem)

    def define_line(self, line, premise=-1, theorem=-1, root=True):  # 线
        if self.line.add(line, premise, theorem):
            if root:  # 如果是 definition tree 的根节点，那子节点都使用当前节点的 premise
                premise = self.line.indexes[line]
            self.line.add(line[::-1], premise, -2)  # 一条线2种表示
            self.point.add(line[0], premise, -2)  # 定义线上的点
            self.point.add(line[1], premise, -2)
            return True
        return False

    def define_angle(self, angle, premise=-1, theorem=-1, root=True):  # 角
        if self.angle.add(angle, premise, theorem):
            if root:
                premise = self.angle.indexes[angle]
            self.define_line(angle[0:2], premise, -2, False)  # 构成角的两个线
            self.define_line(angle[1:3], premise, -2, False)
            return True
        return False

    def define_arc(self, arc, premise=-1, theorem=-1, root=True):  # 弧
        if self.arc.add(arc, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = self.arc.indexes[arc]
            self.point.add(arc[0], premise, -2)  # 构成弧的两个点
            self.point.add(arc[1], premise, -2)
            return True
        return False

    def define_shape(self, shape, premise=-1, theorem=-1, root=True):
        if self.shape.add(shape, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = self.shape.indexes[shape]
            for point in shape:
                # 添加构成shape的点。因为shape形状不确定，只能知道有这些点。
                self.point.add(point, premise, -2)
            return True
        return False

    def define_circle(self, circle, premise=-1, theorem=-1, root=True):  # 圆
        if self.circle.add(circle, premise, theorem):
            if root:
                premise = self.circle.indexes[circle]
            self.point.add(circle, premise, -2)  # 圆心
            return True
        return False

    def define_sector(self, sector, premise=-1, theorem=-1, root=True):  # 扇形
        if self.sector.add(sector, premise, theorem):
            if root:
                premise = self.sector.indexes[sector]
            self.define_angle(sector[2] + sector[0] + sector[1], premise, -2, False)  # 构成扇形的角
            self.define_arc(sector[1:3], premise, -2, False)  # 扇形的弧
            # 之后注意先把实体定义完，在定义属性相关的equal关系
            # sym1 = self.get_sym_length_of_line(sector[0] + sector[1])  # 得到线长度的符号表示
            # sym2 = self.get_sym_length_of_line(sector[0] + sector[2])  # 得到线长度的符号表示
            # self.equal(sym1, sym2, premise, -2)  # 扇形的两边相等，使用equal添加到expression
            return True
        return False

    def define_triangle(self, triangle, premise=-1, theorem=-1, root=True):  # 三角形
        if self.triangle.add(triangle, premise, theorem):
            if root:
                premise = self.triangle.indexes[triangle]
            triangle_all = get_all_representation_of_shape(triangle)  # 3种表示
            for triangle in range(triangle_all):
                self.triangle.add(triangle, premise, -2)
                self.define_angle(triangle, premise, -2)  # 定义3个角
            return True
        return False

    def define_right_triangle(self, triangle, premise=-1, theorem=-1, root=True):  # 直角三角形
        if self.right_triangle.add(triangle, premise, theorem):
            if root:
                premise = self.right_triangle.indexes[triangle]
            self.define_triangle(triangle, premise, -2, False)  # RT三角形也是普通三角形
            # sym = self.get_sym_degree_of_angle(triangle)  # 得到直角角度属性的符号表示
            # self.equal(sym, 90, premise, -2)  # 直角等于90° 使用equal添加到expression
            return True
        return False

    def define_isosceles_triangle(self, triangle, premise=-1, theorem=-1, root=True):  # 等腰三角形
        if self.isosceles_triangle.add(triangle, premise, theorem):
            if root:
                premise = self.isosceles_triangle.indexes[triangle]
            self.define_triangle(triangle, premise, -2, False)  # 等腰三角形也是普通三角形
            # sym1 = self.get_sym_degree_of_angle(triangle)  # 等腰三角形两底角相等
            # sym2 = self.get_sym_degree_of_angle(triangle[0] + triangle[2] + triangle[1])
            # self.equal(sym1, sym2, premise, -2)
            # sym1 = self.get_sym_length_of_line(triangle[0] + triangle[1])  # 等腰三角形两腰相等
            # sym2 = self.get_sym_length_of_line(triangle[0] + triangle[2])
            # self.equal(sym1, sym2, premise, -2)
            return True
        return False

    def define_regular_triangle(self, triangle, premise=-1, theorem=-1, root=True):  # 正三角形
        if self.regular_triangle.add(triangle, premise, theorem):
            if root:
                premise = self.regular_triangle.indexes[triangle]
            triangle_all = get_all_representation_of_shape(triangle)  # 3种表示
            for triangle in range(triangle_all):
                self.regular_triangle.add(triangle, premise, -2)
                self.define_isosceles_triangle(triangle, premise, -2, False)  # 等边也是等腰
            return True
        return False

    def define_quadrilateral(self, shape, premise=-1, theorem=-1, root=True):  # 四边形
        if self.quadrilateral.add(shape, premise, theorem):
            if root:
                premise = self.quadrilateral.indexes[shape]
            quadrilateral_all = get_all_representation_of_shape(shape)  # 4种表示
            for quadrilateral in range(quadrilateral_all):
                self.quadrilateral.add(quadrilateral, premise, -2)
                self.define_angle(quadrilateral[0:3], premise, -2, False)  # 四边形由角组成
            return True
        return False

    def define_trapezoid(self, shape, premise=-1, theorem=-1, root=True):  # 梯形
        if self.trapezoid.add(shape, premise, theorem):
            if root:
                premise = self.trapezoid.indexes[shape]
            self.trapezoid.add(shape[2] + shape[3] + shape[0] + shape[1], premise, -2)  # 一个梯形两种表示
            self.define_quadrilateral(shape, premise, -2, False)  # 梯形也是四边形
            # self.define_parallel(shape[0] + shape[1], shape[2] + shape[3], premise, -2)   # 平行边
            return True
        return False

    def define_isosceles_trapezoid(self, shape, premise=-1, theorem=-1, root=True):  # 等腰梯形
        if self.isosceles_trapezoid.add(shape, premise, theorem):
            if root:
                premise = self.isosceles_trapezoid.indexes[shape]
            self.isosceles_trapezoid.add(shape[2] + shape[3] + shape[0] + shape[1], premise, -2)  # 一个等腰梯形两种表示
            self.define_trapezoid(shape, premise, -2, False)  # 等腰梯形也是梯形
            # sym1 = self.get_sym_length_of_line(shape[1] + shape[2])    # 两腰相等
            # sym2 = self.get_sym_length_of_line(shape[3] + shape[0])
            # self.equal(sym1, sym2, premise, theorem)
            # sym1 = self.get_sym_degree_of_angle(shape[0] + shape[1] + shape[2])  # 两顶角相等
            # sym2 = self.get_sym_degree_of_angle(shape[1] + shape[0] + shape[3])
            # self.equal(sym1, sym2, premise, theorem)
            # sym1 = self.get_sym_degree_of_angle(shape[1] + shape[2] + shape[3])  # 两底角相等
            # sym2 = self.get_sym_degree_of_angle(shape[2] + shape[3] + shape[0])
            # self.equal(sym1, sym2, premise, theorem)
            return True
        return False

    def define_parallelogram(self, shape, premise=-1, theorem=-1, root=True):  # 平行四边形
        if self.parallelogram.add(shape, premise, theorem):
            if root:
                premise = self.parallelogram.indexes[shape]
            parallelogram_all = get_all_representation_of_shape(shape)  # 4种表示
            for parallelogram in range(parallelogram_all):
                self.parallelogram.add(parallelogram, premise, -2)
            self.define_trapezoid(shape, premise, -2, False)  # 平行四边形也是梯形
            self.define_trapezoid(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_rectangle(self, shape, premise=-1, theorem=-1, root=True):  # 长方形
        if self.rectangle.add(shape, premise, theorem):
            if root:
                premise = self.rectangle.indexes[shape]
            rectangle_all = get_all_representation_of_shape(shape)  # 4种表示
            for rectangle in range(rectangle_all):
                self.rectangle.add(rectangle, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 长方形也是平行四边形
            self.define_isosceles_trapezoid(shape, premise, -2, False)  # 长方形也是等腰梯形
            self.define_isosceles_trapezoid(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_kite(self, shape, premise=-1, theorem=-1, root=True):  # 风筝形
        if self.kite.add(shape, premise, theorem):
            if root:
                premise = self.kite.indexes[shape]
            self.kite.add(shape[2] + shape[3] + shape[0] + shape[1], premise, -2)  # 2种表示
            self.define_quadrilateral(shape, premise, -2, False)  # Kite也是四边形
            return True
        return False

    def define_rhombus(self, shape, premise=-1, theorem=-1, root=True):  # 菱形
        if self.rhombus.add(shape, premise, theorem):
            if root:
                premise = self.rhombus.indexes[shape]
            rhombus_all = get_all_representation_of_shape(shape)  # 4种表示
            for rhombus in range(rhombus_all):
                self.rhombus.add(rhombus, premise, -2)
            self.define_parallelogram(shape, premise, -2, False)  # 菱形也是平行四边形
            self.define_kite(shape, premise, -2, False)  # 菱形也是Kite
            self.define_kite(shape[1] + shape[2] + shape[3] + shape[0], premise, -2, False)
            return True
        return False

    def define_square(self, shape, premise=-1, theorem=-1, root=True):  # 正方形
        if self.square.add(shape, premise, theorem):
            if root:
                premise = self.square.indexes[shape]
            square_all = get_all_representation_of_shape(shape)  # 4种表示
            for square in range(square_all):
                self.square.add(square, premise, -2)
            self.define_rectangle(shape, premise, -2, False)  # 正方形也是长方形
            self.define_rhombus(shape, premise, -2, False)  # 正方形也是菱形
            return True
        return False

    def define_polygon(self, shape, premise=-1, theorem=-1, root=True):  # 多边形
        if self.polygon.add(shape, premise, theorem):
            if root:
                premise = self.polygon.indexes[shape]
            polygon_all = get_all_representation_of_shape(shape)  # 所有表示
            for polygon in range(polygon_all):
                self.polygon.add(polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            return True
        return False

    def define_regular_polygon(self, shape, premise=-1, theorem=-1, root=True):  # 正多边形
        if self.regular_polygon.add(shape, premise, theorem):
            if root:
                premise = self.regular_polygon.indexes[shape]
            polygon_all = get_all_representation_of_shape(shape)  # 所有表示
            for polygon in range(polygon_all):
                self.regular_polygon.add(polygon, premise, -2)
                self.define_angle(polygon[0:3], premise, -2, False)  # 由角组成
            self.define_polygon(shape, premise, -2, False)  # 正多边形也是多边形
            return True
        return False

    """------------define Relation------------"""

    def define_point_on_line(self, point, line, premise=-1, theorem=-1, root=True):  # 点在线上
        if self.point_on_line.add((point, line), premise, theorem):
            if root:
                premise = self.point_on_line.indexes[(point, line)]
            self.point_on_line.add((point, line[::-1]), premise, theorem)  # 点在线上两种表示
            self.point.add(point, premise, -2)  # 定义点和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_point_on_arc(self, point, arc, premise=-1, theorem=-1, root=True):  # 点在弧上
        if self.point_on_arc.add((point, arc), premise, theorem):
            if root:
                premise = self.point_on_arc.indexes[(point, arc)]
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_arc(arc, premise, -2, False)
            return True
        return False

    def define_point_on_circle(self, point, circle, premise=-1, theorem=-1, root=True):  # 点在圆上
        if self.point_on_circle.add((point, circle), premise, theorem):
            if root:
                premise = self.point_on_circle.indexes[(point, circle)]
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_midpoint(self, point, line, premise=-1, theorem=-1, root=True):  # 中点
        if self.midpoint.add((point, line), premise, theorem):
            if root:
                premise = self.midpoint.indexes[(point, line)]
            self.midpoint.add((point, line[::-1]), premise, -2)  # 中点有两中表示形式
            self.point.add(point, premise, -2)  # 定义点和弧
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_circumcenter(self, point, triangle, premise=-1, theorem=-1, root=True):  # 外心
        if self.circumcenter.add((point, triangle), premise, theorem):
            if root:
                premise = self.circumcenter.indexes[(point, triangle)]
            triangle_all = get_all_representation_of_shape(triangle)  # 一个三角形三种表示
            for triangle in triangle_all:
                self.circumcenter.add((point, triangle_all), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_incenter(self, point, triangle, premise=-1, theorem=-1, root=True):  # 内心
        if self.incenter.add((point, triangle), premise, theorem):
            if root:
                premise = self.incenter.indexes[(point, triangle)]
            triangle_all = get_all_representation_of_shape(triangle)  # 一个三角形三种表示
            for triangle in triangle_all:
                self.incenter.add((point, triangle_all), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_centroid(self, point, triangle, premise=-1, theorem=-1, root=True):  # 重心
        if self.centroid.add((point, triangle), premise, theorem):
            if root:
                premise = self.centroid.indexes[(point, triangle)]
            triangle_all = get_all_representation_of_shape(triangle)  # 一个三角形三种表示
            for triangle in triangle_all:
                self.centroid.add((point, triangle_all), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_orthocenter(self, point, triangle, premise=-1, theorem=-1, root=True):  # 垂心
        if self.orthocenter.add((point, triangle), premise, theorem):
            if root:
                premise = self.orthocenter.indexes[(point, triangle)]
            triangle_all = get_all_representation_of_shape(triangle)  # 一个三角形三种表示
            for triangle in triangle_all:
                self.orthocenter.add((point, triangle_all), premise, -2)
            self.point.add(point, premise, -2)  # 定义点和三角形
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_parallel(self, line1, line2, premise=-1, theorem=-1, root=True):  # 线平行
        if self.parallel.add((line1, line2), premise, theorem):
            if root:
                premise = self.parallel.indexes[(line1, line2)]
            self.parallel.add((line2, line1), premise, theorem)  # 平行有4种表示
            self.parallel.add((line1[::-1], line2[::-1]), premise, theorem)
            self.parallel.add((line2[::-1], line1[::-1]), premise, theorem)
            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            return True
        return False

    def define_intersect(self, point, line1, line2, premise=-1, theorem=-1, root=True):  # 线相交
        if self.intersect.add((point, line1, line2), premise, theorem):
            if root:
                premise = self.intersect.indexes[(point, line1, line2)]
            self.intersect.add((point, line2[::-1], line1), premise, -2)  # 相交有4种表示
            self.intersect.add((point, line1[::-1], line2[::-1]), premise, -2)
            self.intersect.add((point, line2, line1[::-1]), premise, -2)
            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
            if point != "$":  # 如果给出交点
                self.point.add(point, premise, -2)
                self.define_point_on_line(point, line1, premise, -2, False)
                self.define_point_on_line(point, line2, premise, -2, False)
            return True
        return False

    def define_perpendicular(self, point, line1, line2, premise=-1, theorem=-1, root=True):
        if self.perpendicular.add((point, line1, line2), premise, theorem):
            if root:
                premise = self.perpendicular.indexes[(point, line1, line2)]
            self.perpendicular.add((point, line2[::-1], line1), premise, -2)  # 垂直有4种表示
            self.perpendicular.add((point, line1[::-1], line2[::-1]), premise, -2)
            self.perpendicular.add((point, line2, line1[::-1]), premise, -2)
            self.define_intersect(point, line1, line2, premise, -2, False)  # 垂直也是相交
            return True
        return False

    def define_perpendicular_bisector(self, point, line1, line2, premise=-1, theorem=-1, root=True):  # 垂直平分
        if self.perpendicular_bisector.add((point, line1, line2), premise, theorem):
            if root:
                premise = self.perpendicular_bisector.indexes[(point, line1, line2)]
            self.perpendicular_bisector.add((point, line1[::-1], line2[::-1]), premise, -2)  # 垂直平分有2种表示
            self.define_perpendicular(point, line1, line2, premise, -2, False)  # 垂直平分也是垂直
            return True
        return False

    def define_bisects_angle(self, line, angle, premise=-1, theorem=-1, root=True):  # 角平分线
        if self.bisects_angle.add((line, angle), premise, theorem):
            if root:
                premise = self.bisects_angle.indexes[(line, angle)]
            self.define_angle(angle, premise, -2, False)  # 定义角和线
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_disjoint_line_circle(self, line, circle, premise=-1, theorem=-1, root=True):  # 线圆相离
        if self.disjoint_line_circle.add((line, circle), premise, theorem):
            if root:
                premise = self.disjoint_line_circle.indexes[(line, circle)]
            self.define_line(line, premise, -2, False)  # 定义和线圆
            self.define_circle(circle, premise, -2, False)
            return True
        return False

    def define_disjoint_circle_circle(self, circle1, circle2, premise=-1, theorem=-1, root=True):  # 圆圆相离
        if self.disjoint_circle_circle.add((circle1, circle2), premise, theorem):
            if root:
                premise = self.disjoint_circle_circle.indexes[(circle1, circle2)]
            self.disjoint_circle_circle.add((circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            return True
        return False

    def define_tangent_line_circle(self, point, line, circle, premise=-1, theorem=-1, root=True):  # 相切
        if self.tangent_line_circle.add((point, line, circle), premise, theorem):
            if root:
                premise = self.tangent_line_circle.indexes[(point, line, circle)]
            self.define_line(line, premise, -2, False)  # 定义线和圆
            self.define_circle(circle, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.point.add(point, premise, -2)
                self.define_point_on_line(point, line, premise, -2, False)
                self.define_point_on_circle(point, circle, premise, -2, False)
            return True
        return False

    def define_tangent_circle_circle(self, point, circle1, circle2, premise=-1, theorem=-1, root=True):  # 相切
        if self.tangent_circle_circle.add((point, circle1, circle2), premise, theorem):
            if root:
                premise = self.tangent_line_circle.indexes[(point, circle1, circle2)]
            self.tangent_line_circle.add((point, circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point != "$":  # 如果给出切点
                self.point.add(point, premise, -2)
                self.define_point_on_circle(point, circle1, premise, -2, False)
                self.define_point_on_circle(point, circle2, premise, -2, False)
            return True
        return False

    def define_intersect_line_circle(self, point1, point2, line, circle, premise=-1, theorem=-1, root=True):  # 相交
        if self.intersect_line_circle.add((point1, point2, line, circle), premise, theorem):
            if root:
                premise = self.intersect_line_circle.indexes[(point1, point2, line, circle)]
            self.define_line(line, premise, -2, False)  # 定义线
            self.define_circle(circle, premise, -2, False)  # 定义圆
            if point1 != "$":  # 如果给出交点
                self.point.add(point1, premise, -2)
                self.define_point_on_line(point1, line, premise, -2, False)
                self.define_point_on_circle(point1, circle, premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.point.add(point2, premise, -2)
                self.define_point_on_line(point2, line, premise, -2, False)
                self.define_point_on_circle(point2, circle, premise, -2, False)
            return True
        return False

    def define_intersect_circle_circle(self, point1, point2, circle1, circle2, premise=-1, theorem=-1, root=True):  # 相交
        if self.intersect_circle_circle.add((point1, point2, circle1, circle2), premise, theorem):
            if root:
                premise = self.intersect_line_circle.indexes[(point1, point2, circle1, circle2)]
            self.intersect_line_circle.add((point2, point1, circle2, circle1), premise, -2)  # 2种表示
            self.define_circle(circle1, premise, -2, False)  # 定义圆
            self.define_circle(circle2, premise, -2, False)
            if point1 != "$":  # 如果给出交点
                self.point.add(point1, premise, -2)
                self.define_point_on_circle(point1, circle1, premise, -2, False)
                self.define_point_on_circle(point1, circle2, premise, -2, False)
            if point2 != "$":  # 如果给出交点
                self.point.add(point2, premise, -2)
                self.define_point_on_circle(point2, circle1, premise, -2, False)
                self.define_point_on_circle(point2, circle2, premise, -2, False)
            return True
        return False

    def define_median(self, line, triangle, premise=-1, theorem=-1, root=True):  # 中线
        if self.median.add((line, triangle), premise, theorem):
            if root:
                premise = self.median.indexes[(line, triangle)]
            self.define_line(line, premise, -2)  # 定义实体
            self.define_triangle(triangle, premise, -2)
            self.define_point_on_line(line[1], triangle[1:3], premise, -2, False)  # 子关系
            return True
        return False

    def define_height_triangle(self, height, triangle, premise=-1, theorem=-1, root=True):  # 高
        if self.height_triangle.add((height, triangle), premise, theorem):
            if root:
                premise = self.height_triangle.indexes[(height, triangle)]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_height_trapezoid(self, height, trapezoid, premise=-1, theorem=-1, root=True):  # 高
        if self.height_trapezoid.add((height, trapezoid), premise, theorem):
            if root:
                premise = self.height_trapezoid.indexes[(height, trapezoid)]
            self.define_line(height, premise, -2, False)  # 定义实体
            self.define_trapezoid(trapezoid, premise, -2, False)
            return True
        return False

    def define_internally_tangent(self, point, circle1, circle2, premise=-1, theorem=-1, root=True):  # 内切 circle1是大的
        if self.internally_tangent.add((point, circle1, circle2), premise, theorem):
            if root:
                premise = self.internally_tangent.indexes[(point, circle1, circle2)]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            if point != "$":
                self.point.add(point, premise, -2)
            return True
        return False

    def define_contain(self, point, circle1, circle2, premise=-1, theorem=-1, root=True):  # 内含 circle1是大的
        if self.contain.add((point, circle1, circle2), premise, theorem):
            if root:
                premise = self.contain.indexes[(point, circle1, circle2)]
            self.define_circle(circle1, premise, -2, False)  # 定义实体
            self.define_circle(circle2, premise, -2, False)
            if point != "$":
                self.point.add(point, premise, -2)
            return True
        return False

    def define_circumscribed_to_triangle(self, circle, triangle, premise=-1, theorem=-1, root=True):  # 外接圆
        if self.circumscribed_to_triangle.add((circle, triangle), premise, theorem):
            if root:
                premise = self.circumscribed_to_triangle.indexes[(circle, triangle)]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            return True
        return False

    def define_inscribed_in_triangle(self, point1, point2, point3, circle, triangle, premise=-1, theorem=-1, root=True):
        if self.inscribed_in_triangle.add((point1, point2, point3, circle, triangle), premise, theorem):
            if root:
                premise = self.inscribed_in_triangle.indexes[(point1, point2, point3, circle, triangle)]
            self.define_circle(circle, premise, -2, False)
            self.define_triangle(triangle, premise, -2, False)
            if point1 != "$":
                self.point.add(point1, premise, -2)
                self.define_point_on_line(point1, triangle[0:2], premise, -2, False)
                self.define_point_on_circle(point1, circle, premise, -2, False)
            if point2 != "$":
                self.point.add(point2, premise, -2)
                self.define_point_on_line(point2, triangle[1:3], premise, -2, False)
                self.define_point_on_circle(point2, circle, premise, -2, False)
            if point3 != "$":
                self.point.add(point3, premise, -2)
                self.define_point_on_line(point3, triangle[2] + triangle[0], premise, -2, False)
                self.define_point_on_circle(point3, circle, premise, -2, False)
            return True
        return False

    def define_congruent(self, triangle1, triangle2, premise=-1, theorem=-1, root=True):  # 全等
        if self.congruent.add((triangle1, triangle2), premise, theorem):
            if root:
                premise = self.congruent.indexes[(triangle1, triangle2)]
            triangle1_all = get_all_representation_of_shape(triangle1)  # 6种
            triangle2_all = get_all_representation_of_shape(triangle2)
            for i in range(len(triangle1_all)):
                self.congruent.add((triangle1_all[i], triangle2_all[i]), premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    def define_similar(self, triangle1, triangle2, premise=-1, theorem=-1, root=True):  # 相似
        if self.similar.add((triangle1, triangle2), premise, theorem):
            if root:
                premise = self.similar.indexes[(triangle1, triangle2)]
            triangle1_all = get_all_representation_of_shape(triangle1)  # 6种表示方式
            triangle2_all = get_all_representation_of_shape(triangle2)
            for i in range(len(triangle1_all)):
                self.similar.add((triangle1_all[i], triangle2_all[i]), premise, -2)
            self.define_triangle(triangle1, premise, -2, False)  # 定义实体
            self.define_triangle(triangle2, premise, -2, False)
            return True
        return False

    """------------Expression related------------"""

    def define_expression(self, expression, premise=-1, theorem=-1):  # 定义表达式
        return self.expression.add(expression, premise, theorem)

    """------------Attr's Symbol------------"""

    def get_sym_length_of_line(self, attr):
        if attr not in self.length_of_line.attr:
            sym = attr.lower() + "_" + "ll"
            self.length_of_line.add(attr, sym)
            self.length_of_line.add(attr[::-1], sym)
        else:
            sym = self.length_of_line.attr[attr][0]
        return sym

    def get_sym_length_of_arc(self, attr):
        pass

    def get_sym_degree_of_angle(self, attr):
        if attr not in self.degree_of_angle.attr:
            sym = attr.lower() + "_" + "da"
            self.degree_of_angle.add(attr, sym)
            self.degree_of_angle.add(attr[::-1], sym)
        else:
            sym = self.degree_of_angle.attr[attr][0]
        return sym

    def get_sym_degree_of_sector(self, attr):
        pass

    def get_sym_radius_of_arc(self, attr):
        pass

    def get_sym_radius_of_circle(self, attr):
        pass

    def get_sym_radius_of_sector(self, attr):
        pass

    def get_sym_diameter_of_circle(self, attr):
        pass

    def get_sym_perimeter_of_triangle(self, attr):
        pass

    def get_sym_perimeter_of_circle(self, attr):
        pass

    def get_sym_perimeter_of_sector(self, attr):
        pass

    def get_sym_perimeter_of_quadrilateral(self, attr):
        pass

    def get_sym_perimeter_of_polygon(self, attr):
        pass

    def get_sym_area_of_triangle(self, attr):
        pass

    def get_sym_area_of_circle(self, attr):
        pass

    def get_sym_area_of_sector(self, attr):
        pass

    def get_sym_area_of_quadrilateral(self, attr):
        pass

    def get_sym_area_of_polygon(self, attr):
        pass

    # 处理等式left=right，并将其添加到expression中
    def equal(self, left, right, premise=-1, theorem=-1):
        return self.define_expression("{}={}".format(left, right), premise, theorem)

    # 设置解题目标
    def set_target(self, target_type, target):
        self.target_type = target_type
        self.target = target

    """------------high level function------------"""

    def find_(self):
        pass

    def high_level_func(self):
        pass


class Problem(ProblemLogic):

    def __init__(self, problem_index, formal_languages, theorem_seqs):
        super().__init__()
        self.problem_index = problem_index
        self.formal_languages = formal_languages
        self.theorem_seqs = theorem_seqs

    def show_problem(self):
        conditions = [self.point, self.line, self.angle, self.arc, self.shape, self.circle, self.sector,
                      self.triangle, self.right_triangle, self.isosceles_triangle, self.regular_triangle,
                      self.quadrilateral, self.trapezoid, self.isosceles_trapezoid, self.parallelogram,
                      self.rectangle, self.kite, self.rhombus,
                      self.square, self.polygon, self.regular_polygon, self.point_on_line,
                      self.point_on_arc, self.point_on_circle, self.midpoint, self.circumcenter, self.incenter,
                      self.centroid, self.orthocenter, self.parallel, self.intersect, self.perpendicular,
                      self.perpendicular_bisector, self.bisects_angle, self.disjoint_line_circle,
                      self.disjoint_circle_circle, self.tangent_line_circle, self.tangent_circle_circle,
                      self.intersect_line_circle, self.intersect_circle_circle, self.median, self.height_triangle,
                      self.height_trapezoid, self.internally_tangent, self.contain,
                      self.circumscribed_to_triangle, self.inscribed_in_triangle, self.congruent, self.similar,
                      self.expression]
        sym_of_attr = [self.length_of_line, self.degree_of_angle, self.length_of_arc, self.radius_of_arc,
                       self.radius_of_circle, self.radius_of_sector, self.degree_of_sector,
                       self.diameter_of_circle,
                       self.perimeter_of_triangle, self.perimeter_of_circle, self.perimeter_of_sector,
                       self.perimeter_of_quadrilateral, self.perimeter_of_polygon, self.area_of_triangle,
                       self.area_of_circle, self.area_of_sector, self.area_of_quadrilateral, self.area_of_polygon]
        # Formal Language
        print("problem_index: {}".format(self.problem_index))
        print("formal_languages:")
        for formal_language in self.formal_languages:  # 解析 formal language
            print(formal_language)
        print("theorem_seqs: ", end="")
        for theorem in self.theorem_seqs:
            print(theorem, end=" ")
        print("\n")

        # Logic
        for condition in conditions:  # 条件
            if len(condition.items) > 0:
                print("{}: {}".format(condition.type.name, condition.name))
                for item in condition.items:
                    if condition.type is ConditionType.Entity:
                        print("{0:^4}{1:^9}{2:^4}{3:^4}".format(condition.indexes[item], item,
                                                                condition.premises[item],
                                                                condition.theorems[item]))
                    else:
                        print("{0:^4}{1:^20}{2:^4}{3:^4}".format(condition.indexes[item], str(item),
                                                                 condition.premises[item],
                                                                 condition.theorems[item]))
        for sym in sym_of_attr:  # 属性的符号表示
            if len(sym.attr) > 0:
                print("Attr's Symbol: {}".format(sym.type.name))
                for key in sym.attr.keys():
                    print(key, end=": ")
                    print(sym.attr[key])
