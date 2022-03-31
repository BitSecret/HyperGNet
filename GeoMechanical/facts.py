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
    LengthOfLine = 1  # 线长度
    LengthOfArc = 2  # 弧长
    DegreeOfAngle = 3  # 角度
    DegreeOfSector = 4  # 扇形圆心角度数
    RadiusOfArc = 5  # 弧半径长度
    RadiusOfCircle = 6  # 圆半径长度
    RadiusOfSector = 7  # 扇形版经常堵
    DiameterOfCircle = 8  # 圆的直径
    PerimeterOfTriangle = 9  # 三角形的周长
    PerimeterOfCircle = 10  # 圆的周长
    PerimeterOfSector = 11  # 扇形的周长
    PerimeterOfQuadrilateral = 12  # 四边形的周长
    PerimeterOfPolygon = 13  # 多边形的周长
    AreaOfTriangle = 14  # 三角形的面积
    AreaOfCircle = 15  # 圆的面积
    AreaOfSector = 16  # 扇形的面积
    AreaOfQuadrilateral = 17  # 四边形的面积
    AreaOfPolygon = 18  # 多边形的面积
    Free = 19  # 自由的符号


class TargetType(Enum):    # 解题目标类型
    value = 1       # 求值
    relation = 2    # 求关系

