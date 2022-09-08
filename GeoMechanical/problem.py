from utility import Representation as rep
from facts import AttributionType as aType
from facts import EquationType as eType
from facts import ConditionType as cType
from facts import Condition
from facts import FormalLanguage
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

    """------------Construction------------"""

    def define_shape(self, shape, premise, theorem, root=True):
        if self.conditions.add(shape, cType.shape, premise, theorem):  # 因为规定了方向，一个弧的表示方式是唯一的
            if root:
                premise = [self.conditions.get_index(shape, cType.shape)]

            for point in shape:   # 添加构成shape的点
                self.define_point(point, premise, -2)

            if len(shape) == 3:    # 三角形
                self.define_triangle(shape, premise, -2)
            else:    # 多边形
                self.define_polygon(shape, premise, -2)

            for shape in rep.shape(shape):  # shape多种表示
                self.conditions.add(shape, cType.shape, premise, -2)
            return True
        return False

    def define_collinear(self, points, premise, theorem):    # 共线
        if len(points) > 2 and self.conditions.add(points, cType.collinear, premise, theorem):
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
            self.conditions.add(triangle[0] + triangle[2] + triangle[1], cType.isosceles_triangle, premise, -2)  # 两种表示
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
            self.define_point(point, premise, -2)  # 定义点和弧
            self.define_line(line, premise, -2, False)
            return True
        return False

    def define_intersect(self, ordered_pair, premise, theorem, root=True):  # 线相交
        point, line1, line2 = ordered_pair
        if self.conditions.add(ordered_pair, cType.intersect, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.intersect)]

            for all_form in rep.intersect(ordered_pair):  # 相交有4种表示
                self.conditions.add(all_form, cType.intersect, premise, -2)

            self.define_line(line1, premise, -2, False)  # 定义线
            self.define_line(line2, premise, -2, False)
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

    def define_perpendicular(self, ordered_pair, premise, theorem, root=True):
        point, line1, line2 = ordered_pair

        if self.conditions.add(ordered_pair, cType.perpendicular, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.perpendicular)]

            for all_form in rep.intersect(ordered_pair):
                self.conditions.add(all_form, cType.perpendicular, premise, -2)  # 垂直有4种表示

            self.define_intersect(ordered_pair, premise, -2, False)  # 垂直也是相交

            sym = []
            if len(set(point + line1 + line2)) == 3:
                if line1[0] == line2[1]:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2 + line1[1])))
                elif line1[1] == line2[1]:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1 + line2[0])))
                elif line1[1] == line2[0]:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2[::-1] + line1[0])))
                elif line1[0] == line2[0]:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1[::-1] + line2[1])))
            elif len(set(point + line1 + line2)) == 4:
                if line2[1] == point:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1[0] + line2[::-1])))
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2 + line1[1])))
                elif line1[1] == point:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2[1] + line1[::-1])))
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1 + line2[0])))
                elif line2[0] == point:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1[1] + line2)))
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2[::-1] + line1[0])))
                elif line1[0] == point:
                    sym.append(self.get_sym_of_attr((aType.MA.name, line2[0] + line1)))
                    sym.append(self.get_sym_of_attr((aType.MA.name, line1[::-1] + line2[1])))
            elif len(set(point + line1 + line2)) == 5:
                sym.append(self.get_sym_of_attr((aType.MA.name, line1[0] + point + line2[0])))
                sym.append(self.get_sym_of_attr((aType.MA.name, line2[0] + point + line1[1])))
                sym.append(self.get_sym_of_attr((aType.MA.name, line1[1] + point + line2[1])))
                sym.append(self.get_sym_of_attr((aType.MA.name, line2[1] + point + line1[0])))

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

    def define_bisector(self, ordered_pair, premise, theorem, root=True):  # 角平分线
        line, angle = ordered_pair
        if self.conditions.add(ordered_pair, cType.bisector, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.bisector)]
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

    def define_neutrality(self, ordered_pair, premise, theorem, root=True):    # 中位线
        line, triangle = ordered_pair
        if self.conditions.add(ordered_pair, cType.neutrality, premise, theorem):
            if root:
                premise = [self.conditions.get_index(ordered_pair, cType.is_altitude)]
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

    def get_sym_of_attr(self, attr):    # attr: (aType, entity_name)
        if attr[0] == aType.T:  # 表示目标/中间值类型的符号，不用存储在符号库
            return symbols(attr[0].name.lower() + "_" + attr[1])

        if attr not in self.sym_of_attr.keys():  # 若无符号，新建符号
            if attr[0] in [aType.MA, aType.F]:
                sym = symbols(attr[0].lower() + "_" + attr[1].lower())    # 角、自由变量可以有负的
            else:
                sym = symbols(attr[0].lower() + "_" + attr[1].lower(), positive=True)  # 属性值没有负数

            self.value_of_sym[sym] = None  # 符号值

            if attr[0] in [aType.LL, aType.MA, aType.AS, aType.PT]:    # entity 有多种表示形式
                for all_form in rep.shape(attr[1]):
                    self.sym_of_attr[(attr[0], all_form)] = sym
            else:    # entity 只有一种形式
                self.sym_of_attr[attr] = sym  # 符号

        else:  # 有符号就返回符号
            sym = self.sym_of_attr[attr]

        return sym

    def set_value_of_sym(self, sym, value, premise, theorem):  # 设置符号的值
        if self.value_of_sym[sym] is None:
            self.value_of_sym[sym] = value
            self.define_equation(sym - value, eType.value, premise, theorem)
        return False

    def angle_repsentation_alignment(self):    # 使角的角度表示符号一致
        for angle in self.conditions.items[cType.angle]:
            if (aType.MA.name, angle) in self.sym_of_attr.keys():  # 有符号了就不用再赋予了
                continue

            a_points = angle[0]
            o_point = angle[1]
            b_points = angle[2]
            sym = self.get_sym_of_attr((aType.MA.name, angle))

            coll_a = None  # 与AO共线的collinear
            coll_b = None  # 与OB共线的collinear
            for coll in self.conditions.items[cType.collinear]:
                if a_points in coll and o_point in coll:
                    coll_a = coll
                if o_point in coll and b_points in coll:
                    coll_b = coll

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
                    self.sym_of_attr[(aType.MA.name, a_point + o_point + b_point)] = sym


class Problem(ProblemLogic):

    def __init__(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer):
        super().__init__()
        self.problem_index = problem_index
        self.fl = FormalLanguage(construction_fls, text_fls, image_fls, target_fls)
        self.theorem_seqs = theorem_seqs
        self.answer = answer
        self.solve_time_list = []

    """------------解方程相关------------"""

    @func_set_timeout(5)  # 限时5s
    def solve_equations(self, target_sym=None, target_equation=None):  # 求解方程

        if target_sym is None:  # 只涉及basic、value、theorem
            if self.equation_solved:  # basic、theorem没有更新，不用重复求解
                return

            sym_set = []    # 所有值未知的符号
            for equation in list(self.basic_equations.values()) + list(self.theorem_equations.values()):
                sym_set += equation.free_symbols
            sym_set = list(set(sym_set))    # 快速去重

            update = True
            while update:    # 循环求解变量值
                update = False
                self.simplify_equations()  # solve前先化简方程
                for sym in sym_set:  # 遍历所有的值未知的符号值，并形成求解该符号的最小方程组
                    if self.value_of_sym[sym] is None:  # 符号值未知，尝试求解
                        min_equations, premise = self.get_minimum_equations(set(), sym, False)
                        solved_result = solve(min_equations)  # 求解min_equations
                        if len(solved_result) > 0:  # 有解
                            if isinstance(solved_result, list):  # 解不唯一，选第一个(涉及三角函数时可能有多个解)
                                solved_result = solved_result[0]
                            if sym in solved_result.keys() and isinstance(solved_result[sym], Float):  # sym有实数解
                                self.set_value_of_sym(sym, solved_result[sym], premise, -3)
                                update = True

            self.theorem_equations = {}  # 清空 theorem_equations
            self.equation_solved = True  # 更新方程求解状态
        else:  # 求解target
            equations, premise = self.get_minimum_equations({target_sym}, target_equation, True)  # 使用value + basic
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

                if len(self.basic_equations[key].free_symbols) == 1:  # 化简后只剩一个符号，自然求得符号
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

    def get_minimum_equations(self, target_sym, target_equation, solve_target):  # 找到与求解目标方程相关的最小(basic、value)方程组
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
                        if solve_target:    # 当求解目标时，需要加入equation:sym-value；只求解basic+theorem的时候不用
                            min_equations.append(equation)  # 加入解题方程组
                        premise.append(self.conditions.get_index(equation, cType.equation))  # 方程序号作为前提
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
        print("\033[32mbasic_equations:\033[0m")
        for equation in self.basic_equations.keys():
            print(equation, end=":   ")
            print(self.basic_equations[equation])
        print("\033[32mtheorem_equations:\033[0m")
        for equation in self.theorem_equations.keys():
            print(equation, end=":   ")
            print(self.theorem_equations[equation])
        print()

    """------------辅助功能------------"""

    def new_problem(self, problem_index, construction_fls, text_fls, image_fls, target_fls, theorem_seqs, answer): 
        self.problem_index = problem_index
        self.fl = FormalLanguage(construction_fls, text_fls, image_fls, target_fls)
        self.theorem_seqs = theorem_seqs
        self.answer = answer
        self.solve_time_list = []

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

    def anti_generate_fl_using_logic(self):
        pass

    def show(self):
        # Formal Language
        print("\033[36mproblem_index:\033[0m", end=" ")
        print(self.problem_index)
        print("\033[36mconstruction_fls:\033[0m")
        for construction_fl in self.fl.construction_fls:  # 解析 formal language
            print(construction_fl)
        print("\033[36mtext_fls:\033[0m")
        for text_fl in self.fl.text_fls:  # 解析 formal language
            print(text_fl)
        print("\033[36mimage_fls:\033[0m")
        for image_fl in self.fl.image_fls:  # 解析 formal language
            print(image_fl)

        print("\033[36mtarget_fls:\033[0m")
        for target_fl in self.fl.target_fls:  # 解析 formal language
            print(target_fl)

        print("\033[36mreasoning_fls:\033[0m")
        for reasoning_fl in self.fl.reasoning_fls:  # 解析 formal language
            print(reasoning_fl, end="   (step:")
            print(self.fl.reasoning_fls_step[reasoning_fl], end=", theorem:")
            print(self.fl.reasoning_fls_theorem[reasoning_fl], end=")\n")

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
        for i in range(0, self.target_count):
            print("\033[34m{}:\033[0m {}".format(self.target_type[i].name, str(self.target[i])), end="  ")
            print("\033[34mcorrect answer:\033[0m {}".format(self.answer[i]), end="  ")
            if self.target_solved[i] == "solved":
                print("\033[32m{}\033[0m".format(self.target_solved[i]))
            else:
                print("\033[31m{}\033[0m".format(self.target_solved[i]))

        # 求解时间
        for solve_time in self.solve_time_list:
            print(solve_time)

        print()

    def simpel_show(self):
        print("\033[36mproblem_index:\033[0m", end=" ")
        print(self.problem_index)
        for i in range(0, self.target_count):
            print("\033[34m{}:\033[0m {}".format(self.target_type[i].name, str(self.target[i])), end="  ")
            print("\033[34mcorrect answer:\033[0m {}".format(self.answer[i]), end="  ")
            if self.target_solved[i] == "solved":
                print("\033[32m{}\033[0m".format(self.target_solved[i]))
            else:
                print("\033[31m{}\033[0m".format(self.target_solved[i]))

    def show_fl(self):
        pass
