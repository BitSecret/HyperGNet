from enum import Enum


class Condition:  # 条件
    index = 0  # 静态变量，给每个条件唯一序号

    def __init__(self):
        self.items = []  # 每一个条件
        self.indexes = {}  # 每个条件的序号
        self.premises = {}  # 推出条件需要的前提
        self.theorems = {}  # 推出条件应用的定理

    def add(self, item, premise, theorem):
        if item not in self.items:
            self.items.append(item)
            self.indexes[item] = Condition.index
            self.premises[item] = premise
            self.theorems[item] = theorem
            Condition.index = Condition.index + 1
            return True
        return False


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

