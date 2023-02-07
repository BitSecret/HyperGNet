from definition.object import Condition, Construction, Relation, Equation
from definition.exception import RuntimeException
from itertools import combinations


class Problem:
    def __init__(self, predicate_GDL, problem_CDL):
        """
        initialize a problem.
        :param predicate_GDL: parsed predicate_GDL.
        :param problem_CDL: parsed problem_CDL
        """
        Condition.id = 0  # init step and id
        Condition.step = 0

        self.problem_CDL = problem_CDL  # parsed problem msg. It will be further decomposed.
        self.predicate_GDL = predicate_GDL  # problem predicate definition

        self.theorems_applied = []  # applied theorem list
        self.get_predicate_by_id = {}
        self.get_id_by_step = {}
        self.gathered = False

        self.conditions = {}    # init conditions
        for predicate in self.predicate_GDL["Construction"]:
            self.conditions[predicate] = Construction(predicate)
        for predicate in self.predicate_GDL["Entity"]:
            self.conditions[predicate] = Relation(predicate)
        for predicate in self.predicate_GDL["Relation"]:
            self.conditions[predicate] = Relation(predicate)
        self.conditions["Equation"] = Equation("Equation", self.predicate_GDL["Attribution"])

        for predicate, item in problem_CDL["parsed_cdl"]["construction_cdl"]:  # conditions of construction
            self.add(predicate, tuple(item), (-1,), "prerequisite")

        self.construction_init()  # start construction

        for predicate, item in problem_CDL["parsed_cdl"]["text_and_image_cdl"]:  # conditions of text_and_image
            if predicate == "Equal":
                self.add("Equation", self.conditions["Equation"].get_equation_from_tree(item), (-1,), "prerequisite")
            else:
                self.add(predicate, tuple(item), (-1,), "prerequisite")

        self.goal = {  # set goal
            "solved": False,
            "solved_answer": None,
            "premise": None,
            "theorem": None,
            "solving_msg": [],
            "type": problem_CDL["parsed_cdl"]["goal"]["type"]
        }
        if self.goal["type"] == "value":
            self.goal["item"] = self.conditions["Equation"].get_expr_from_tree(
                problem_CDL["parsed_cdl"]["goal"]["item"][1][0]
            )
            self.goal["answer"] = self.conditions["Equation"].get_expr_from_tree(
                problem_CDL["parsed_cdl"]["goal"]["answer"]
            )
        elif self.goal["type"] == "equal":
            self.goal["item"] = self.conditions["Equation"].get_equation_from_tree(
                problem_CDL["parsed_cdl"]["goal"]["item"][1]
            )
            self.goal["answer"] = 0
        else:  # relation type
            self.goal["item"] = problem_CDL["parsed_cdl"]["goal"]["item"]
            self.goal["answer"] = tuple(problem_CDL["parsed_cdl"]["goal"]["answer"])

    def construction_init(self):
        """
        1.Iterative build all shape.
        Shape(BC*A), Shape(A*CD)  ==>  Shape(ABCD)
        2.Make the symbols of angles the same.
        Measure(Angle(ABC)), Measure(Angle(ABD))  ==>  m_abc,  if Collinear(BCD)
        """
        update = True    # build all shape
        traversed = []
        while update:
            update = False
            for shape1 in list(self.conditions["Shape"].get_id_by_item):
                for shape2 in list(self.conditions["Shape"].get_id_by_item):
                    if (shape1, shape2) in traversed:    # skip traversed
                        continue
                    traversed.append((shape1, shape2))

                    if not (shape1[len(shape1) - 1] == shape2[0] and    # At least two points are the same
                            shape1[len(shape1) - 2] == shape2[1]):
                        continue

                    same_length = 2    # Number of identical points
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

        collinear = []    # let same angle has same symbol
        for predicate, item in self.problem_CDL["parsed_cdl"]["construction_cdl"]:
            if predicate == "Collinear":
                collinear.append(tuple(item))
        self.conditions["Equation"].angle_alignment(list(self.conditions["Angle"].get_id_by_item), collinear)

    def gather_conditions_msg(self):
        """Gather all conditions msg for problem showing, solution tree generating, etc..."""
        if self.gathered:
            return
        self.get_predicate_by_id = {}    # init
        self.get_id_by_step = {}
        for predicate in self.conditions:
            for _id in self.conditions[predicate].get_item_by_id:
                self.get_predicate_by_id[_id] = predicate
            for step, _id in self.conditions[predicate].step_msg:
                if step not in self.get_id_by_step:
                    self.get_id_by_step[step] = []
                self.get_id_by_step[step].append(_id)
        self.gathered = True

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
        if not self._is_valid(predicate, item):  # return when invalid
            return False

        self.gathered = False

        if predicate == "Equation":  # Equation
            added, _id = self.conditions["Equation"].add(item, premise, theorem)
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
                self.add("Polygon", item, (_id,), "extended")
                l = len(item)
                for bias in range(l):
                    extended_shape = []  # extend Shape
                    for i in range(l):
                        extended_shape.append(item[(i + bias) % l])
                    self.conditions["Shape"].add(tuple(extended_shape), (_id,), "extended")
                    extended_angle = [item[0 + bias], item[(1 + bias) % l], item[(2 + bias) % l]]  # extend Angle
                    self.add("Angle", tuple(extended_angle), (_id,), "extended")
                return True
        elif predicate == "Polygon":
            item, i = list(item), 0
            premise = list(premise)
            while len(item) > 2 and i < len(item):  # Check whether collinear points exist
                point1 = item[i]  # sliding window in the length of 3
                point2 = item[(i + 1) % len(item)]
                point3 = item[(i + 2) % len(item)]

                if (point1, point2, point3) in self.conditions["Collinear"].get_id_by_item:  # Delete when collinear
                    item.pop(item.index(point2))
                    premise.append(self.conditions["Collinear"].get_id_by_item[(point1, point2, point3)])
                else:  # Move backward sliding window when not collinear
                    i += 1
            item = tuple(item)
            added, _id = self.conditions["Polygon"].add(item, tuple(premise), theorem)
            if added:  # if added successful
                l = len(item)
                for bias in range(l):
                    extended_shape = []  # extend Polygon
                    for i in range(l):
                        extended_shape.append(item[(i + bias) % l])
                    self.conditions["Polygon"].add(tuple(extended_shape), (_id,), "extended")
                return True
        elif predicate == "Collinear":  # Construction predicate: Collinear
            added, _id = self.conditions["Collinear"].add(item, premise, theorem)
            if added:
                for l in range(3, len(item) + 1):  # extend collinear
                    for extended_item in combinations(item, l):
                        self.conditions["Collinear"].add(extended_item, (_id,), "extended")
                for i in range(len(item) - 1):  # extend line
                    for j in range(i + 1, len(item)):
                        self.add("Line", (item[i], item[j]), (_id,), "extended")
                return True
        return False

    def _is_valid(self, predicate, item):    # 还没写完
        """Validity check for format of condition item."""
        if predicate not in self.conditions:
            raise RuntimeException("PredicateNotDefined",
                                   "Predicate '{}': not defined in current problem.".format(predicate))
        if predicate in self.predicate_GDL["Entity"]:
            return True
        elif predicate in self.predicate_GDL["Relation"]:
            return True
        else:
            return True

    def applied(self, theorem_name):
        """Execute when theorem successful applied. Save theorem name and update step."""
        self.theorems_applied.append(theorem_name)
        Condition.step += 1
