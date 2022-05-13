from enum import Enum


class ConditionType(Enum):  # 条件的类型
    point = 1  # 点、线、角、弧
    line = 2
    angle = 3
    arc = 4
    shape = 5  # 形状
    circle = 6  # 圆和扇形
    sector = 7
    triangle = 8  # 三角形
    right_triangle = 9
    isosceles_triangle = 10
    regular_triangle = 11
    quadrilateral = 12  # 四边形
    trapezoid = 13
    isosceles_trapezoid = 14
    parallelogram = 15
    rectangle = 16
    kite = 17
    rhombus = 18
    square = 19
    polygon = 20  # 多边形
    regular_polygon = 21
    collinear = 22  # 共线
    point_on_line = 23  # 点的关系
    point_on_arc = 24
    point_on_circle = 25
    midpoint = 26
    circumcenter = 27
    incenter = 28
    centroid = 29
    orthocenter = 30
    parallel = 31  # 线的关系
    intersect = 32
    perpendicular = 33
    perpendicular_bisector = 34
    bisects_angle = 35
    disjoint_line_circle = 36
    disjoint_circle_circle = 37
    tangent_line_circle = 38
    tangent_circle_circle = 39
    intersect_line_circle = 40
    intersect_circle_circle = 41
    median = 42
    height_triangle = 43
    height_trapezoid = 44
    internally_tangent = 45  # 图形的关系
    contain = 46
    circumscribed_to_triangle = 47
    inscribed_in_triangle = 48
    congruent = 49
    similar = 50
    chord = 51
    equation = 52  # 代数方程


class Condition:  # 条件
    entity_list = [ConditionType.point, ConditionType.line, ConditionType.angle, ConditionType.arc, ConditionType.shape,
                   ConditionType.circle, ConditionType.sector, ConditionType.triangle, ConditionType.right_triangle,
                   ConditionType.isosceles_triangle, ConditionType.regular_triangle, ConditionType.quadrilateral,
                   ConditionType.trapezoid, ConditionType.isosceles_trapezoid, ConditionType.parallelogram,
                   ConditionType.rectangle, ConditionType.kite, ConditionType.rhombus, ConditionType.square,
                   ConditionType.polygon, ConditionType.regular_polygon]
    entity_relation_list = [ConditionType.collinear, ConditionType.point_on_line, ConditionType.point_on_arc,
                            ConditionType.point_on_circle, ConditionType.midpoint, ConditionType.circumcenter,
                            ConditionType.incenter, ConditionType.centroid, ConditionType.orthocenter,
                            ConditionType.parallel, ConditionType.intersect, ConditionType.perpendicular,
                            ConditionType.perpendicular_bisector, ConditionType.bisects_angle,
                            ConditionType.disjoint_line_circle, ConditionType.disjoint_circle_circle,
                            ConditionType.tangent_line_circle, ConditionType.tangent_circle_circle,
                            ConditionType.intersect_line_circle, ConditionType.intersect_circle_circle,
                            ConditionType.median, ConditionType.height_triangle, ConditionType.height_trapezoid,
                            ConditionType.internally_tangent, ConditionType.contain,
                            ConditionType.circumscribed_to_triangle, ConditionType.inscribed_in_triangle,
                            ConditionType.congruent, ConditionType.similar, ConditionType.chord]
    equation = ConditionType.equation

    def __init__(self):
        self.count = 0  # 条件计数
        self.items = {}  # 条件集  key:ConditionType  value:[]
        self.item_list = []  # 按序号顺序存储条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

        for entity in Condition.entity_list:   # 初始化
            self.items[entity] = []
            self.indexes[entity] = {}
            self.premises[entity] = {}
            self.theorems[entity] = {}
        for entity_relation in Condition.entity_relation_list:
            self.items[entity_relation] = []
            self.indexes[entity_relation] = {}
            self.premises[entity_relation] = {}
            self.theorems[entity_relation] = {}
        self.items[ConditionType.equation] = []
        self.indexes[ConditionType.equation] = {}
        self.premises[ConditionType.equation] = {}
        self.theorems[ConditionType.equation] = {}

    def add(self, item, condition_type, premise, theorem):
        if item not in self.items[condition_type]:  # 如果是新条件，添加
            self.items[condition_type].append(item)
            self.item_list.append([item, condition_type])
            self.indexes[condition_type][item] = self.count
            self.premises[condition_type][item] = sorted(premise)
            self.theorems[condition_type][item] = theorem
            self.count += 1  # 更新条件总数
            return True
        return False

    def get_index(self, item, condition_type):
        return self.indexes[condition_type][item]

    def get_premise(self, item, condition_type):
        return self.premises[condition_type][item]

    def get_theorem(self, item, condition_type):
        return self.theorems[condition_type][item]

    def clean(self):
        self.count = 0  # 条件计数
        self.items = {}  # 条件集  key:ConditionType  value:[]
        self.item_list = []  # 按序号顺序存储条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

        for entity in Condition.entity_list:   # 初始化
            self.items[entity] = []
            self.indexes[entity] = {}
            self.premises[entity] = {}
            self.theorems[entity] = {}
        for entity_relation in Condition.entity_relation_list:
            self.items[entity_relation] = []
            self.indexes[entity_relation] = {}
            self.premises[entity_relation] = {}
            self.theorems[entity_relation] = {}
        self.items[ConditionType.equation] = []
        self.indexes[ConditionType.equation] = {}
        self.premises[ConditionType.equation] = {}
        self.theorems[ConditionType.equation] = {}


class AttributionType(Enum):  # 属性的类型
    LL = 1  # LengthOfLine 线长度
    LA = 2  # LengthOfArc 弧长
    DA = 3  # DegreeOfAngle 角度
    DS = 4  # DegreeOfSector 扇形圆心角度数
    RA = 5  # RadiusOfArc 弧半径长度
    RC = 6  # RadiusOfCircle 圆半径长度
    RS = 7  # RadiusOfSector 扇形版经常堵
    DC = 8  # DiameterOfCircle 圆的直径
    PT = 9  # PerimeterOfTriangle 三角形的周长
    PC = 10  # PerimeterOfCircle 圆的周长
    PS = 11  # PerimeterOfSector 扇形的周长
    PQ = 12  # PerimeterOfQuadrilateral 四边形的周长
    PP = 13  # PerimeterOfPolygon 多边形的周长
    AT = 14  # AreaOfTriangle 三角形的面积
    AC = 15  # AreaOfCircle 圆的面积
    AS = 16  # AreaOfSector 扇形的面积
    AQ = 17  # AreaOfQuadrilateral 四边形的面积
    AP = 18  # AreaOfPolygon 多边形的面积
    F = 19  # Free 自由符号
    T = 20  # Target 代数型解题目标


class TargetType(Enum):  # 解题目标类型
    value = 1  # 代数关系，求值
    equal = 2  # 代数关系，验证
    entity = 3  # 位置关系，实体
    relation = 4  # 位置关系，联系
    symbol = 5  # 代数关系，符号形式，以后再实现


class EquationType(Enum):  # 方程的类型
    basic = 1  # 由构图语句和初始条件扩充来的方程
    value = 2  # 已经求出的符号，用方程形式表示，便于求解后续方程
    theorem = 3  # 定理得到的方程
