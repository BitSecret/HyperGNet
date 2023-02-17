import warnings
from definition.object import Condition, Construction, Relation, Equation
from itertools import combinations
from sympy import symbols
from aux_tools.parse import EqParser


class Problem:
    def __init__(self, predicate_GDL, problem_CDL):
        """
        initialize a problem.
        :param predicate_GDL: parsed predicate_GDL.
        :param problem_CDL: parsed problem_CDL
        """
        Condition.id = 0  # init step and id
        Condition.step = 0

        self.loaded = False  # indicate whether problem is completely loaded

        self.problem_CDL = problem_CDL  # parsed problem msg, it will be further decomposed
        self.predicate_GDL = predicate_GDL  # problem predicate definition

        self.theorems_applied = []  # applied theorem list
        self.get_predicate_by_id = {}
        self.get_id_by_step = {}
        self.gathered = False

        self.conditions = {}  # init conditions
        for predicate in self.predicate_GDL["Construction"]:
            self.conditions[predicate] = Construction(predicate)
        for predicate in self.predicate_GDL["Entity"]:
            self.conditions[predicate] = Relation(predicate)
        for predicate in self.predicate_GDL["Relation"]:
            self.conditions[predicate] = Relation(predicate)
        self.conditions["Equation"] = Equation("Equation", self.predicate_GDL["Attribution"])

        for predicate, item in problem_CDL["parsed_cdl"]["construction_cdl"]:  # conditions of construction
            if predicate == "Collinear":
                self.add(predicate, tuple(item), (-1,), "prerequisite")
        for predicate, item in problem_CDL["parsed_cdl"]["construction_cdl"]:
            if predicate == "Shape":
                self.add(predicate, tuple(item), (-1,), "prerequisite")

        self.construction_init()  # start construction

        for predicate, item in problem_CDL["parsed_cdl"]["text_and_image_cdl"]:  # conditions of text_and_image
            if predicate == "Equal":
                self.add("Equation", EqParser.get_equation_from_tree(self, item), (-1,), "prerequisite")
            else:
                self.add(predicate, tuple(item), (-1,), "prerequisite")

        self.goal = {"type": problem_CDL["parsed_cdl"]["goal"]["type"]}  # set goal
        if self.goal["type"] == "value":
            self.goal["item"] = EqParser.get_expr_from_tree(self, problem_CDL["parsed_cdl"]["goal"]["item"][1][0])
            self.goal["answer"] = EqParser.get_expr_from_tree(self, problem_CDL["parsed_cdl"]["goal"]["answer"])
        elif self.goal["type"] == "equal":
            self.goal["item"] = EqParser.get_equation_from_tree(self, problem_CDL["parsed_cdl"]["goal"]["item"][1])
            self.goal["answer"] = 0
        else:  # relation type
            self.goal["item"] = problem_CDL["parsed_cdl"]["goal"]["item"]
            self.goal["answer"] = tuple(problem_CDL["parsed_cdl"]["goal"]["answer"])
        self.goal["solved"] = False
        self.goal["solved_answer"] = None
        self.goal["premise"] = None
        self.goal["theorem"] = None
        self.goal["solving_msg"] = []

        self.loaded = True

    def construction_init(self):
        """
        1.Iterative build all shape.
        Shape(BC*A), Shape(A*CD)  ==>  Shape(ABCD)
        2.Make the symbols of angles the same.
        Measure(Angle(ABC)), Measure(Angle(ABD))  ==>  m_abc,  if Collinear(BCD)
        """
        update = True  # 1.Iterative build all shape
        traversed = []
        while update:
            update = False
            for shape1 in list(self.conditions["Shape"].get_id_by_item):
                for shape2 in list(self.conditions["Shape"].get_id_by_item):
                    if (shape1, shape2) in traversed:  # skip traversed
                        continue
                    traversed.append((shape1, shape2))

                    if not (shape1[len(shape1) - 1] == shape2[0] and  # At least two points are the same
                            shape1[len(shape1) - 2] == shape2[1]):
                        continue

                    same_length = 2  # Number of identical points
                    while same_length < len(shape1) and same_length < len(shape2):
                        if shape1[len(shape1) - same_length - 1] == shape2[same_length]:
                            same_length += 1
                        else:
                            break

                    new_shape = list(shape1[0:len(shape1) - same_length + 1])  # points in shape1, the first same point
                    new_shape += list(shape2[same_length:len(shape2)])  # points in shape2
                    new_shape.append(shape1[len(shape1) - 1])  # the second same point

                    if 2 < len(new_shape) == len(set(new_shape)):  # make sure new_shape is Shape and no ring
                        premise = (self.conditions["Shape"].get_id_by_item[shape1],
                                   self.conditions["Shape"].get_id_by_item[shape2])
                        update = self.add("Shape", tuple(new_shape), premise, "extended") or update

        collinear = []  # 2.Make the symbols of angles the same
        for predicate, item in self.problem_CDL["parsed_cdl"]["construction_cdl"]:
            if predicate == "Collinear":
                collinear.append(tuple(item))
        angles = list(self.conditions["Angle"].get_id_by_item)
        for angle in angles:
            if (angle, "Measure") in self.conditions["Equation"].sym_of_attr:
                continue
            sym = self.get_sym_of_attr(angle, "Measure")

            a, v, b = angle
            a_points = []  # Points collinear with a and on the same side with a
            b_points = []
            for coll in collinear:
                if v in coll and a in coll:
                    if coll.index(v) < coll.index(a):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            a_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            a_points.append(coll[i])
                            i += 1
                    break
            if len(a_points) == 0:
                a_points.append(a)
            for coll in collinear:
                if v in coll and b in coll:
                    if coll.index(v) < coll.index(b):  # .....V...P..
                        i = coll.index(v) + 1
                        while i < len(coll):
                            b_points.append(coll[i])
                            i += 1
                    else:  # ...P.....V...
                        i = 0
                        while i < coll.index(v):
                            b_points.append(coll[i])
                            i += 1
                    break
            if len(b_points) == 0:
                b_points.append(b)

            if len(a_points) == 1 and len(b_points) == 1:  # 角只有一种表示
                continue

            same_angles = []
            for a_point in a_points:
                for b_point in b_points:
                    same_angles.append((a_point, v, b_point))  # 相同的角设置一样的符号

            for same_angle in same_angles:
                self.conditions["Equation"].sym_of_attr[(same_angle, "Measure")] = sym
            self.conditions["Equation"].attr_of_sym[sym] = [same_angles, "Measure"]

    def add(self, predicate, item, premise, theorem):
        """
        Add item to condition of specific predicate category.
        Also consider condition expansion and equation construction.
        :param predicate: Construction, Entity, Relation or Equation.
        :param item: <tuple> or equation.
        :param premise: tuple of <int>, premise of item.
        :param theorem: <str>, theorem of item.
        :return: True or False
        """
        if not self.item_is_valid(predicate, item):   # validity check
            return False  # return when invalid

        self.gathered = False

        if predicate == "Equation":  # Equation
            added, _ = self.conditions["Equation"].add(item, premise, theorem)
            return added
        elif predicate in self.predicate_GDL["Entity"]:  # Entity
            added, _id = self.conditions[predicate].add(item, premise, theorem)
            if added:
                for para_list in self.predicate_GDL["Entity"][predicate]["multi"]:  # multi
                    para = []
                    for i in para_list:
                        para.append(item[i])
                    self.conditions[predicate].add(tuple(para), (_id,), "extended")

                for extended_predicate, para_list in self.predicate_GDL["Entity"][predicate]["extend"]:  # extended
                    para = []
                    for i in para_list:
                        para.append(item[i])
                    self.add(extended_predicate, tuple(para), (_id,), "extended")
                return True
        elif predicate in self.predicate_GDL["Relation"]:  # Relation
            added, _id = self.conditions[predicate].add(item, premise, theorem)
            if added:
                for para_list in self.predicate_GDL["Relation"][predicate]["multi"]:  # multi
                    para = []
                    for i in para_list:
                        para.append(item[i])
                    self.conditions[predicate].add(tuple(para), (_id,), "extended")

                for extended_predicate, para_list in self.predicate_GDL["Relation"][predicate]["extend"]:  # extended
                    para = []
                    for i in para_list:
                        para.append(item[i])
                    self.add(extended_predicate, tuple(para), (_id,), "extended")
                return True
        elif predicate == "Shape":  # Construction predicate: Shape
            added, _id = self.conditions["Shape"].add(item, premise, theorem)
            if added:  # if added successful
                non_vertex_points = []
                for i in range(0, len(item)):
                    point1 = item[i]  # sliding window in the length of 3
                    point2 = item[(i + 1) % len(item)]
                    j = 0
                    while True:
                        point3 = item[(i + 2 + j) % len(item)]
                        if (point1, point2, point3) not in self.conditions["Collinear"].get_id_by_item:
                            break
                        j += 1
                    if j > 0:
                        non_vertex_points.append("".join([item[(i + 1 + k) % len(item)] for k in range(j)]))
                new_non_vertex_points = []
                for i in non_vertex_points:
                    can_add = True
                    for j in non_vertex_points:
                        if i != j and i in j:
                            can_add = False
                            break
                    if can_add:
                        new_non_vertex_points.append(i)
                non_vertex_points = new_non_vertex_points

                polygon_premise = [_id]
                for non_vertex_point in non_vertex_points:
                    for collinear in self.conditions["Collinear"].get_id_by_item:
                        if non_vertex_point in collinear:
                            polygon_premise.append(self.conditions["Collinear"].get_id_by_item[collinear])
                            break

                extended_shape = []  # shape after remove collinear points
                non_vertex_points = list((set("".join(non_vertex_points))))
                for i in range(len(non_vertex_points) + 1):
                    for removed_points in combinations(non_vertex_points, i):
                        new_item = list(item)
                        for removed_point in removed_points:
                            new_item.pop(new_item.index(removed_point))
                        extended_shape.append(tuple(new_item))
                for item in extended_shape:
                    l = len(item)
                    for bias in range(l):
                        self.conditions["Shape"].add(
                            tuple([item[(i + bias) % l] for i in range(l)]), (_id,), "extended"
                        )
                        extended_angle = [item[0 + bias], item[(1 + bias) % l], item[(2 + bias) % l]]  # extend Angle
                        self.add("Angle", tuple(extended_angle), (_id,), "extended")

                self.add("Polygon", extended_shape[-1], tuple(polygon_premise), "extended")  # shape no collinear points
                if len(extended_shape) > 1:
                    eq = self.get_sym_of_attr(extended_shape[0], "Area") - \
                         self.get_sym_of_attr(extended_shape[-1], "Area")
                    self.conditions["Equation"].add(eq, tuple(polygon_premise), "extended")
                return True
        elif predicate == "Polygon":
            added, _id = self.conditions["Polygon"].add(item, tuple(premise), theorem)
            if added:  # if added successful
                l = len(item)
                for bias in range(l):
                    self.conditions["Polygon"].add(tuple([item[(i + bias) % l] for i in range(l)]), (_id,), "extended")
                if l == 3:    # default execute theorems
                    self.add("Triangle", item, (_id,), "definition_of_triangle")
                return True
        elif predicate == "Collinear":  # Construction predicate: Collinear
            added, _id = self.conditions["Collinear"].add(item, premise, theorem)
            if added:
                for l in range(3, len(item) + 1):  # extend collinear
                    for extended_item in combinations(item, l):
                        self.conditions["Collinear"].add(extended_item, (_id,), "extended")
                        self.conditions["Collinear"].add(extended_item[::-1], (_id,), "extended")
                        if len(extended_item) == 3:
                            self.conditions["Angle"].add(extended_item, (_id,), "extended")
                            self.conditions["Angle"].add(extended_item[::-1], (_id,), "extended")
                for i in range(len(item) - 1):  # extend line
                    for j in range(i + 1, len(item)):
                        self.add("Line", (item[i], item[j]), (_id,), "extended")
                return True
        return False

    """------------Format Control for <entity relation>------------"""

    def item_is_valid(self, predicate, item):
        """
        Validity check for the format of logic conditions.
        Length Validity check: LV check, throw <Exception>.
        Format Validity check: FV check, throw <Warning>.
        Entity Existence check: EE check, throw <Warning>.
        """
        if predicate not in self.conditions:
            raise Exception(
                "<PredicateNotDefined> Predicate '{}': not defined in current predicate GDL.".format(
                    predicate
                )
            )

        if predicate == "Equation":
            return True, None

        if predicate in self.predicate_GDL["Construction"]:  # FV check
            if len(item) == len(set(item)):
                return True, None
            if not self.loaded:
                warnings.warn("FV check not passed: [{}, {}]".format(predicate, item))
            return False

        if predicate in self.predicate_GDL["Entity"]:
            item_GDL = self.predicate_GDL["Entity"][predicate]
            if len(item) != len(item_GDL["vars"]):  # FV check
                raise Exception(
                    "<ParameterLengthError> Predicate '{}' excepted length: {}. Got: {}".format(
                        predicate, len(item_GDL["vars"]), item
                    )
                )
            if len(item) == len(set(item)):
                return True, None
            if not self.loaded:
                warnings.warn("FV check not passed: [{}, {}]".format(predicate, item))
            return False

        item_GDL = self.predicate_GDL["Relation"][predicate]
        if len(item) != len(item_GDL["vars"]):  # FV check
            raise Exception(
                "<ParameterLengthError> Predicate '{}' excepted length: {}. Got: {}".format(
                    predicate, len(item_GDL["vars"]), item
                )
            )

        for name, para in self.predicate_GDL["Relation"][predicate]["para"]:  # EE check
            if tuple([item[i] for i in para]) not in self.conditions[name].get_id_by_item:
                if not self.loaded:
                    warnings.warn("EE check not passed: [{}, {}]".format(predicate, item))
                return False

        if "format" in item_GDL:
            letters = []
            item_vars = list(item)
            for i in range(len(item_vars)):
                if item_vars[i] not in letters:
                    letters.append(item_vars[i])
                item_vars[i] = letters.index(item_vars[i])
            if item_vars in item_GDL["format"]:
                return True
            if not self.loaded:
                warnings.warn("FV check not passed: [{}, {}]".format(predicate, item))
            return False
        else:
            for mutex in item_GDL["mutex"]:
                if isinstance(mutex[0], list):
                    first = "".join([item[i] for i in mutex[0]])
                    second = "".join([item[i] for i in mutex[1]])
                    if first == second:
                        if not self.loaded:
                            warnings.warn("FV check not passed: [{}, {}]".format(predicate, item))
                        return False
                else:
                    points = [item[i] for i in mutex]
                    if len(points) != len(set(points)):
                        if not self.loaded:
                            warnings.warn("FV check not passed: [{}, {}]".format(predicate, item))
                        return False
            return True

    """-----------Format Control for <algebraic relation>-----------"""

    def get_sym_of_attr(self, item, attr):
        """
        Get symbolic representation of item's attribution.
        :param item: tuple, such as ('A', 'B')
        :param attr: attr's name, such as Length
        :return: sym
        """
        if not self.attr_is_valid(item, attr):   # validity check
            return None

        if (item, attr) not in self.conditions["Equation"].sym_of_attr:  # No symbolic representation, initialize one.
            if self.predicate_GDL["Attribution"][attr]["negative"] == "True":  # Judge whether sym can be negative.
                sym = symbols(self.predicate_GDL["Attribution"][attr]["sym"] + "_" + "".join(item).lower())
            else:
                sym = symbols(self.predicate_GDL["Attribution"][attr]["sym"] + "_" + "".join(item).lower(),
                              positive=True)

            self.conditions["Equation"].value_of_sym[sym] = None  # init symbol's value
            self.conditions["Equation"].sym_of_attr[(item, attr)] = sym  # add sym

            extend_items = [item]
            if isinstance(self.predicate_GDL["Attribution"][attr]["multi"], str):
                l = len(item)
                for bias in range(1, l):
                    extended_item = [item[(i + bias) % l] for i in range(l)]  # extend item
                    extend_items.append(tuple(extended_item))
                    self.conditions["Equation"].sym_of_attr[(tuple(extended_item), attr)] = sym  # multi representation
            else:
                for multi in self.predicate_GDL["Attribution"][attr]["multi"]:
                    extended_item = [item[i] for i in multi]  # extend item
                    extend_items.append(tuple(extended_item))
                    self.conditions["Equation"].sym_of_attr[(tuple(extended_item), attr)] = sym  # multi representation
            self.conditions["Equation"].attr_of_sym[sym] = [extend_items, attr]  # add attr
            return sym

        return self.conditions["Equation"].sym_of_attr[(item, attr)]

    def attr_is_valid(self, item, attr):
        """
        Validity check for format of algebra conditions.
        Length Validity check: LV check, throw <Exception>.
        Format Validity check: FV check, throw <Warning>.
        Entity Existence check: EE check, throw <Warning>.
        """
        if attr == "Free":
            return True

        if isinstance(self.predicate_GDL["Attribution"][attr]["multi"], str):  # EE check
            if item in self.conditions[self.predicate_GDL["Attribution"][attr]["para"]].get_id_by_item:  # EE check
                return True
            if not self.loaded:
                warnings.warn("EE check not passed: [{}, {}]".format(item, attr))
            return False
        else:
            excepted_length = len(self.predicate_GDL["Attribution"][attr]["vars"])
            if len(item) != excepted_length:  # FV check
                raise Exception(
                    "<ParameterLengthError> Attribute '{}' excepted length: {}. Got: {}".format(
                        attr, excepted_length, item
                    )
                )
            for predicate, para in self.predicate_GDL["Attribution"][attr]["para"]:
                if tuple([item[p] for p in para]) not in self.conditions[predicate].get_id_by_item:  # EE check
                    if not self.loaded:
                        warnings.warn("EE check not passed: [{}, {}]".format(item, attr))
                    return False
            return True

    def set_value_of_sym(self, sym, value, premise, theorem):
        """
        Set value of sym.
        Add equation to record the premise and theorem of solving the symbol's value at the same time.
        :param sym: <symbol>
        :param value: <float>
        :param premise: tuple of <int>, premise of getting value.
        :param theorem: <str>, theorem of getting value.
        """
        if self.conditions["Equation"].value_of_sym[sym] is None:
            self.conditions["Equation"].value_of_sym[sym] = value
            added, _id = self.conditions["Equation"].add(sym - value, premise, theorem)
            return added
        return False

    """-----------------------Auxiliary function----------------------"""

    def gather_conditions_msg(self):
        """Gather all conditions msg for problem showing, solution tree generating, etc..."""
        if self.gathered:
            return
        self.get_predicate_by_id = {}  # init
        self.get_id_by_step = {}
        for predicate in self.conditions:
            for _id in self.conditions[predicate].get_item_by_id:
                self.get_predicate_by_id[_id] = predicate
            for step, _id in self.conditions[predicate].step_msg:
                if step not in self.get_id_by_step:
                    self.get_id_by_step[step] = []
                self.get_id_by_step[step].append(_id)
        self.gathered = True

    def applied(self, theorem_name):
        """Execute when theorem successful applied. Save theorem name and update step."""
        self.theorems_applied.append(theorem_name)
        Condition.step += 1
