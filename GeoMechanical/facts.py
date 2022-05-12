from enum import Enum


class Condition:  # 条件
    index_count = {}  # 静态变量，给每个条件唯一序号  key:problem_id  value:condition_count

    # 以下两个数据以及他们的操作，都是为可视化服务的，并不需要在高性能解题过程出现
    conditions = {}  # 通过序号查找条件  key:problem_id  value:{index: condition}
    indexes = {}    # 通过条件查找序号  key:problem_id  value:{condition: index}

    def __init__(self, problem_index):
        self.problem_index = problem_index   # 所属的问题
        self.items = []  # 每一个条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

    def add(self, item, premise, theorem):
        if item not in self.items:
            self.items.append(item)
            Condition.conditions[self.problem_index].append(item)
            self.indexes[item] = Condition.index_count[self.problem_index]
            self.premises[item] = premise
            self.theorems[item] = theorem
            Condition.index_count[self.problem_index] += 1    # 更新条件总数
            return True
        return False

    def clean(self, new_problem_index):    # 更新问题
        self.problem_index = new_problem_index
        self.items = []  # 每一个条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理


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


class TargetType(Enum):    # 解题目标类型
    value = 1       # 代数关系，求值
    equal = 2       # 代数关系，验证
    entity = 3    # 位置关系，实体
    relation = 4    # 位置关系，联系
    symbol = 5    # 代数关系，符号形式，以后再实现


class EquationType(Enum):    # 方程的类型
    basic = 1      # 由构图语句和初始条件扩充来的方程
    value = 2      # 已经求出的符号，用方程形式表示，便于求解后续方程
    theorem = 3    # 定理得到的方程


class ConditionType(Enum):   # 条件的类型
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
    equations = 52  # 代数方程


class NewCondition:  # 条件
    def __init__(self):
        self.count = 0    # 条件计数
        self.items = {}    # 条件集  key:cType  value:
        self.item_list = []  # 按序号顺序存储条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

    def add(self, condition_type, item, premise, theorem):
        if condition_type not in self.items.keys():    # 若之前没有添加过此类条件，初始化
            self.items[condition_type] = []

        if item not in self.items[condition_type]:    # 如果是新条件，添加
            self.items[condition_type].append(item)
            self.item_list.append(item)
            self.indexes[item] = self.count
            self.premises[item] = premise
            self.theorems[item] = theorem
            self.count += 1    # 更新条件总数
            return True
        return False

    def clean(self):
        self.count = 0
        self.items = {}
        self.item_list = []
        self.indexes = {}
        self.premises = {}
        self.theorems = {}
