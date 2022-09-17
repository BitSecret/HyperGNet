from enum import Enum


class ConditionType(Enum):  # 条件的类型
    shape = 1  # 形状
    collinear = 2  # 共线

    point = 3  # 点、线、角、弧
    line = 4
    angle = 5
    triangle = 6  # 三角形
    right_triangle = 7
    isosceles_triangle = 8
    equilateral_triangle = 9
    polygon = 10  # 多边形

    midpoint = 11
    intersect = 12
    parallel = 13
    disorder_parallel = 14
    perpendicular = 15
    perpendicular_bisector = 16
    bisector = 17
    median = 18
    is_altitude = 19
    neutrality = 20
    circumcenter = 21
    incenter = 22
    centroid = 23
    orthocenter = 24
    congruent = 25
    similar = 26
    mirror_congruent = 27
    mirror_similar = 28

    equation = 29  # 代数方程


class AttributionType(Enum):  # 属性的类型
    LL = 1  # LengthOfLine 线长
    MA = 2  # MeasureOfAngle 角度
    AS = 3  # AreaOfShape 面积
    PT = 4  # PerimeterOfTriangle 三角形周长
    AT = 5  # AltitudeOfTriangle 三角形高

    F = 6  # Free 自由符号

    M = 7  # Middle 解题中间过程需要引入的符号


class TargetType(Enum):  # 解题目标类型
    value = 1  # 代数关系，求值
    equal = 2  # 代数关系，验证
    entity = 3  # 位置关系，实体
    relation = 4  # 位置关系，联系
    symbol = 5  # 代数关系，符号形式，以后再实现


class EquationType(Enum):  # 方程的类型
    basic = 1  # 简单方程，整个解题过程都存在
    complex = 2  # 复杂方程，只存在解题中的一步
    value = 3  # sym的值，用方程存储


class Condition:  # 条件
    # 与FormalLanguage一一对应
    construction_list = [ConditionType.shape, ConditionType.collinear]
    entity_list = [ConditionType.point, ConditionType.line, ConditionType.angle, ConditionType.triangle,
                   ConditionType.right_triangle, ConditionType.isosceles_triangle,
                   ConditionType.equilateral_triangle, ConditionType.polygon]
    entity_relation_list = [ConditionType.midpoint, ConditionType.intersect, ConditionType.parallel,
                            ConditionType.disorder_parallel, ConditionType.perpendicular,
                            ConditionType.perpendicular_bisector, ConditionType.bisector,
                            ConditionType.median, ConditionType.is_altitude, ConditionType.neutrality,
                            ConditionType.circumcenter, ConditionType.incenter, ConditionType.centroid,
                            ConditionType.orthocenter, ConditionType.congruent, ConditionType.similar,
                            ConditionType.mirror_congruent, ConditionType.mirror_similar]
    equation = ConditionType.equation
    all = construction_list + entity_list + entity_relation_list + [ConditionType.equation]

    def __init__(self):
        self.count = 0  # 条件计数
        self.items = {}  # 条件集  key:ConditionType  value:[]
        self.item_list = []  # 按序号顺序存储条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

        for entity in Condition.all:  # 初始化
            self.items[entity] = []
            self.indexes[entity] = {}
            self.premises[entity] = {}
            self.theorems[entity] = {}

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

        for entity in Condition.all:  # 初始化
            self.items[entity] = []
            self.indexes[entity] = {}
            self.premises[entity] = {}
            self.theorems[entity] = {}


class FormalLanguage:
    # 与Condition一一对应
    construction_predicates = ["Shape", "Collinear"]
    entity_predicates = ["Point", "Line", "Angle", "Triangle", "RightTriangle", "IsoscelesTriangle",
                         "EquilateralTriangle", "Polygon"]
    entity_relation_predicates = ["Midpoint", "Intersect", "Parallel", "DisorderParallel", "Perpendicular",
                                  "PerpendicularBisector", "Bisector", "Median", "IsAltitude", "Neutrality",
                                  "Circumcenter", "Incenter", "Centroid", "Orthocenter", "Congruent", "Similar",
                                  "MirrorCongruent", "MirrorSimilar"]
    attribute_predicates = ["Length", "Measure", "Area", "Perimeter", "Altitude"]
    equation = "Equation"
    all = construction_predicates + entity_predicates + entity_relation_predicates + attribute_predicates + [equation]

    def __init__(self, construction_fls, text_fls, image_fls, target_fls):
        self.construction_fls = construction_fls
        self.text_fls = text_fls
        self.image_fls = image_fls
        self.target_fls = target_fls

        self.reasoning_fls = []
        self.reasoning_fls_steps = []
        self.step_count = 0

    def add(self, fl):
        if fl not in self.reasoning_fls:
            self.reasoning_fls.append(fl)
            self.reasoning_fls_steps.append(self.step_count)

    def step(self):
        self.step_count += 1
