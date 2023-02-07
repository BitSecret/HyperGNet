import copy
import string
from pyparsing import oneOf, Combine, Word, nums, alphas, OneOrMore
from sympy import sin, cos, tan, pi
from definition.exception import RuntimeException

float_idt = Combine(OneOrMore(Word(nums)) + "." + OneOrMore(Word(nums)))
int_idt = OneOrMore(Word(nums))
alpha_idt = Word(alphas)
operator_idt = oneOf("+ - * / ^ { } @ # $ ~")
idt_expr = OneOrMore(float_idt | int_idt | alpha_idt | operator_idt)  # parse expr of str form to symbolic
operator = ["+", "-", "*", "/", "^", "{", "}", "@", "#", "$", "~"]
stack_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                  "{": 0, "}": None,
                  "@": 4, "#": 4, "$": 4, "~": 0}
outside_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                    "{": 5, "}": 0,
                    "@": 4, "#": 4, "$": 4, "~": 0}


class FLParser:
    @staticmethod
    def parse_predicate(predicate_GDL):
        """parse predicate_GDL to executable form."""
        predicate_GDL = predicate_GDL["Predicates"]
        parsed_GDL = {
            "Construction": ["Shape", "Polygon", "Collinear"],  # preset Construction,
            "Entity": {  # preset Entity
                "Point": {
                    "vars": [0], "format": [[0]], "multi": [], "extend": []
                },
                "Line": {
                    "vars": [0, 1], "format": [[0, 1]], "multi": [[1, 0]], "extend": [["Point", [0]], ["Point", [1]]]
                },
                "Angle": {
                    "vars": [0, 1, 2], "format": [[0, 1, 2]], "multi": [],
                    "extend": [["Line", [0, 1]], ["Line", [1, 2]]]
                }
            },
            "Relation": {},
            "Attribution": {  # preset Attribution
                "Free": {
                    "sym": "f",
                    "para": [],
                    "multi": [],
                    "negative": "True"
                },
                "Length": {
                    "sym": "l",
                    "para": [["Line", [0, 1]]],
                    "multi": [[0, 1], [1, 0]],  # [1, 0]  或 “normal”
                    "negative": "False"
                },
                "Measure": {
                    "sym": "m",
                    "para": [["Angle", [0, 1, 2]]],
                    "multi": [[0, 1, 2]],
                    "negative": "True"
                },
                "Area": {
                    "sym": "a",
                    "para": "Shape",
                    "multi": "normal",
                    "negative": "False"
                },
                "Perimeter": {
                    "sym": "p",
                    "para": "Shape",
                    "multi": "normal",
                    "negative": "False"
                }
            }
        }

        entity = predicate_GDL["Entity"]  # parse entity
        for key in entity:
            name, para, _ = FLParser._parse_one_predicate(entity[key]["name"])
            parsed_GDL["Entity"][name] = {  # parameter of predicate
                "vars": [i for i in range(len(para))]
            }
            if "format" in entity[key]:  # format control
                parsed_format = []
                for i in range(len(entity[key]["format"])):
                    parsed_format.append([])
                    checked = []
                    for item in entity[key]["format"][i]:
                        if item not in checked:
                            checked.append(item)
                        parsed_format[i].append(checked.index(item))
                parsed_GDL["Entity"][name]["format"] = parsed_format
            else:
                parsed_mutex = []

                for i in range(len(entity[key]["mutex"])):
                    parsed_mutex.append([])
                    if isinstance(entity[key]["mutex"][i], str):
                        for item in entity[key]["mutex"][i]:
                            parsed_mutex[i].append(para.index(item))
                    else:
                        parsed_mutex[i].append([])
                        parsed_mutex[i].append([])
                        for item in entity[key]["mutex"][i][0]:
                            parsed_mutex[i][0].append(para.index(item))
                        for item in entity[key]["mutex"][i][1]:
                            parsed_mutex[i][1].append(para.index(item))
                parsed_GDL["Entity"][name]["mutex"] = parsed_mutex
            parsed_GDL["Entity"][name]["multi"] = []  # multi
            for multi in entity[key]["multi"]:
                _, extend_para, _ = FLParser._parse_one_predicate(multi)
                parsed_GDL["Entity"][name]["multi"].append([para.index(i) for i in extend_para])
            parsed_GDL["Entity"][name]["extend"] = []  # extend
            for extend in entity[key]["extend"]:
                extend_name, extend_para, _ = FLParser._parse_one_predicate(extend)
                parsed_GDL["Entity"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

        relation = predicate_GDL["Relation"]  # parse relation
        for key in relation:
            name, para, para_len = FLParser._parse_one_predicate(relation[key]["name"])
            parsed_GDL["Relation"][name] = {  # parameter of predicate
                "vars": [i for i in range(len(para))],
                "para_structure": para_len
            }
            if "format" in relation[key]:  # format control
                parsed_format = []
                for i in range(len(relation[key]["format"])):
                    parsed_format.append([])
                    checked = []
                    for item in relation[key]["format"][i]:
                        if item not in checked:
                            checked.append(item)
                        parsed_format[i].append(checked.index(item))
                parsed_GDL["Relation"][name]["format"] = parsed_format
            else:
                parsed_mutex = []
                for i in range(len(relation[key]["mutex"])):
                    parsed_mutex.append([])
                    if isinstance(relation[key]["mutex"][i], str):
                        for item in relation[key]["mutex"][i]:
                            parsed_mutex[i].append(para.index(item))
                    else:
                        parsed_mutex[i].append([])
                        parsed_mutex[i].append([])
                        for item in relation[key]["mutex"][i][0]:
                            parsed_mutex[i][0].append(para.index(item))
                        for item in relation[key]["mutex"][i][1]:
                            parsed_mutex[i][1].append(para.index(item))
                parsed_GDL["Relation"][name]["mutex"] = parsed_mutex
            parsed_GDL["Relation"][name]["multi"] = []  # multi
            for multi in relation[key]["multi"]:
                _, extend_para, _ = FLParser._parse_one_predicate(multi)
                parsed_GDL["Relation"][name]["multi"].append([para.index(i) for i in extend_para])
            parsed_GDL["Relation"][name]["extend"] = []  # extend
            for extend in relation[key]["extend"]:
                extend_name, extend_para, _ = FLParser._parse_one_predicate(extend)
                parsed_GDL["Relation"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

        attribution = predicate_GDL["Attribution"]  # parse attribution
        for key in attribution:
            name = attribution[key]["name"]
            parsed_GDL["Attribution"][name] = {  # sym
                "sym": attribution[key]["sym"]
            }
            parsed_GDL["Attribution"][name]["para"] = []  # para
            for predicate in attribution[key]["para"]:
                para_name, para, _ = FLParser._parse_one_predicate(predicate)
                parsed_GDL["Attribution"][name]["para"].append(
                    [para_name, [attribution[key]["multi"][0].index(item) for item in para]]
                )
            parsed_GDL["Attribution"][name]["multi"] = [  # multi
                [attribution[key]["multi"][0].index(item) for item in attribution[key]["multi"][i]]
                for i in range(len(attribution[key]["multi"]))
            ]
            parsed_GDL["Attribution"][name]["negative"] = attribution[key]["negative"]  # negative

        return parsed_GDL

    @staticmethod
    def parse_theorem(theorem_GDL):
        """parse theorem_GDL to executable form."""
        theorem_GDL = theorem_GDL["Theorems"]
        parsed_GDL = {}

        for key in theorem_GDL:
            theorem_name = theorem_GDL[key]["name"]
            parsed_GDL[theorem_name] = {}
            for branch in theorem_GDL[key]["description"]:
                parsed_GDL[theorem_name][branch] = {}

                letters = []  # vars

                parsed_premise = FLParser._parse_premise(  # premise
                    [theorem_GDL[key]["description"][branch]["premise"]]
                )
                for i in range(len(parsed_premise)):
                    for j in range(len(parsed_premise[i])):
                        if "Equal" in parsed_premise[i][j]:
                            parsed_premise[i][j] = FLParser._parse_equal_predicate(parsed_premise[i][j])
                            parsed_premise[i][j] = FLParser._replace_letter_with_vars(parsed_premise[i][j], letters)
                        else:
                            predicate, para, _ = FLParser._parse_one_predicate(parsed_premise[i][j])
                            for k in range(len(para)):
                                if para[k] not in letters:
                                    letters.append(para[k])
                                para[k] = letters.index(para[k])

                            parsed_premise[i][j] = [predicate, para]

                parsed_conclusion = []  # conclusion
                for item in theorem_GDL[key]["description"][branch]["conclusion"]:
                    if "Equal" in item:
                        parsed_conclusion.append(
                            FLParser._replace_letter_with_vars(FLParser._parse_equal_predicate(item), letters)
                        )
                    else:
                        predicate, para, _ = FLParser._parse_one_predicate(item)
                        for k in range(len(para)):
                            para[k] = letters.index(para[k])
                        parsed_conclusion.append([predicate, para])

                parsed_GDL[theorem_name][branch]["vars"] = [i for i in range(len(letters))]
                parsed_GDL[theorem_name][branch]["premise"] = parsed_premise
                parsed_GDL[theorem_name][branch]["conclusion"] = parsed_conclusion
        return parsed_GDL

    @staticmethod
    def parse_problem(problem_CDL):
        """parse problem_CDL to executable form."""
        parsed_CDL = {
            "id": problem_CDL["problem_id"],
            "cdl": {
                "construction_cdl": problem_CDL["construction_cdl"],
                "text_cdl": problem_CDL["text_cdl"],
                "image_cdl": problem_CDL["image_cdl"],
                "goal_cdl": problem_CDL["goal_cdl"]
            },
            "parsed_cdl": {
                "construction_cdl": [],
                "text_and_image_cdl": [],
                "goal": {},
            }
        }
        for fl in problem_CDL["construction_cdl"]:
            predicate, para, _ = FLParser._parse_one_predicate(fl)
            parsed_CDL["parsed_cdl"]["construction_cdl"].append([predicate, para])

        for fl in problem_CDL["text_cdl"] + problem_CDL["image_cdl"]:
            if fl.startswith("Equal"):
                parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append(FLParser._parse_equal_predicate(fl))
            else:
                predicate, para, _ = FLParser._parse_one_predicate(fl)
                parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append([predicate, para])

        if problem_CDL["goal_cdl"].startswith("Value"):
            parsed_CDL["parsed_cdl"]["goal"]["type"] = "value"
            parsed_CDL["parsed_cdl"]["goal"]["item"] = FLParser._parse_equal_predicate(problem_CDL["goal_cdl"])
            parsed_CDL["parsed_cdl"]["goal"]["answer"] = problem_CDL["problem_answer"]
        elif problem_CDL["goal_cdl"].startswith("Equal"):
            parsed_CDL["parsed_cdl"]["goal"]["type"] = "equal"
            parsed_CDL["parsed_cdl"]["goal"]["item"] = FLParser._parse_equal_predicate(problem_CDL["goal_cdl"])
            parsed_CDL["parsed_cdl"]["goal"]["answer"] = 0
        else:
            parsed_CDL["parsed_cdl"]["goal"]["type"] = "relation"
            predicate, para, _ = FLParser._parse_one_predicate(problem_CDL["goal_cdl"])
            parsed_CDL["parsed_cdl"]["goal"]["item"] = predicate
            parsed_CDL["parsed_cdl"]["goal"]["answer"] = para

        return parsed_CDL

    @staticmethod
    def _parse_one_predicate(s):
        """
        parse s to get predicate, para, and structural msg.
        >> parse_one('Predicate(ABC)')
        ('Predicate', ['A', 'B', 'C'], [3])
        >> parse_one('Predicate(ABC, DE)')
        ('Predicate', ['A', 'B', 'C', 'D', 'E'], [3, 2])
        """
        predicate_name, para = s.split("(")
        para = para.split(")")[0]
        if "," not in para:
            return predicate_name, list(para), [len(para)]
        para_len = []
        para = para.split(",")
        for item in para:
            para_len.append(len(item))
        return predicate_name, list("".join(para)), para_len

    @staticmethod
    def _parse_premise(premise_GDL):
        """
        Convert geometric logic statements into disjunctive normal forms.
        A&(B|C) ==> A&B|A&C ==> [[A, B], [A, C]]
        >> _parse_premise('IsoscelesTriangle(ABC)&Collinear(BMC)&(IsAltitude(AM,ABC)|Median(AM,ABC)|Bisector(AM,CAB))')
        [['IsoscelesTriangle(ABC)', Collinear(BMC)', IsAltitude(AM,ABC)'],
        ['IsoscelesTriangle(ABC)', Collinear(BMC)', Median(AM,ABC)'],
        ['IsoscelesTriangle(ABC)', Collinear(BMC)', Bisector(AM,CAB)']]
        """
        update = True
        while update:
            expanded = []
            update = False
            for item in premise_GDL:
                if "|" not in item:
                    expanded.append(item)
                else:
                    update = True
                    head = item[0:item.index("&(") + 1]
                    body = []
                    tail = ""

                    count = 1
                    i = item.index("&(") + 2
                    j = i
                    while count > 0:
                        if item[j] == "(":
                            count += 1
                        elif item[j] == ")":
                            count -= 1
                        elif item[j] == "|" and count == 1:
                            body.append(item[i:j])
                            i = j + 1
                        j += 1
                    body.append(item[i:j - 1])
                    if j < len(item):
                        tail = item[j:len(item)]

                    for b in body:
                        expanded.append(head + b + tail)
            premise_GDL = expanded
        for i in range(len(premise_GDL)):
            premise_GDL[i] = premise_GDL[i].split("&")
        return premise_GDL

    @staticmethod
    def _parse_equal_predicate(s):
        """
        Parse s to a Tree.
        >> parse_equal('Equal(Length(AB),Length(CD))')
        ['Equal', [[Length, ['A', 'B']], [Length, ['C', 'D']]]]
        """
        i = 0
        j = 0
        stack = []
        while j < len(s):
            if s[j] == "(":
                stack.append(s[i: j])
                stack.append(s[j])
                i = j + 1
            elif s[j] == ",":
                if i < j:
                    stack.append(s[i: j])
                    i = j + 1
                else:
                    i = i + 1
            elif s[j] == ")":
                if i < j:
                    stack.append(s[i: j])
                    i = j + 1
                else:
                    i = i + 1
                item = []
                while stack[-1] != "(":
                    item.append(stack.pop())
                stack.pop()  # 弹出 "("
                stack.append([stack.pop(), item[::-1]])
            j = j + 1
        return FLParser._listing(stack.pop())

    @staticmethod
    def _listing(s_tree):
        """
        Recursive trans s_tree's para to para list.
        >> listing(['Add', [['Length', ['AB']], ['Length', ['CD']]]])
        ['Add', [['Length', ['A', 'B']], ['Length', ['C', 'D']]]]
        """
        if not isinstance(s_tree, list):
            return s_tree

        is_para = True  # Judge whether the para list is reached.
        for para in s_tree:
            if isinstance(para, list):
                is_para = False
                break
            for p in list(para):
                if p not in string.ascii_uppercase:
                    is_para = False
                    break
            if not is_para:
                break

        if is_para:
            return list("".join(s_tree))
        else:
            return [FLParser._listing(para) for para in s_tree]

    @staticmethod
    def _replace_letter_with_vars(s_tree, s_var):
        """
        Recursive trans s_tree's para to vars.
        >> replace_letter_with_vars(['Add', [['Length', ['A', 'B']], ['Length', ['C', 'D']]]], ['A', 'B', 'C', 'D'])
        ['Add', [['Length', ['0', '1']], ['Length', ['2', '3']]]]
        """
        if not isinstance(s_tree, list):
            return s_tree

        is_para = True  # Judge whether the para list is reached.
        for para in s_tree:
            if isinstance(para, list):
                is_para = False
                break
            for p in list(para):
                if p not in string.ascii_uppercase:
                    is_para = False
                    break
            if not is_para:
                break

        if is_para:
            return [s_var.index(para) for para in s_tree]
        else:
            return [FLParser._replace_letter_with_vars(para, s_var) for para in s_tree]


class EqParser:

    @staticmethod
    def get_expr_from_tree(problem, tree, replaced=False, letters=None):
        """
        Recursively trans expr_tree to symbolic algebraic expression.
        :param problem: class <Problem>.
        :param tree: An expression in the form of a list tree.
        :param replaced: Optional. Set True when tree's item is expressed by vars.
        :param letters: Optional. Letters that will replace vars.
        >> get_expr_from_tree(problem, ['Length', ['T', 'R']])
        l_tr
        >> get_expr_from_tree(problem, ['Add', [['Length', ['Z', 'X']], '2*x-14']])
        2.0*f_x + l_zx - 14.0
        >> get_expr_from_tree(problem, ['Sin', [['Measure', ['0', '1', '2']]]], True, ['X', 'Y', 'Z'])
        sin(pi*m_zxy/180)
        """
        if not isinstance(tree, list):  # expr
            return EqParser.parse_expr(problem, tree)

        if tree[0] in problem.predicate_GDL["Attribution"]:  # attr
            if not replaced:
                return problem.get_sym_of_attr(tuple(tree[1]), tree[0])
            else:
                replaced_item = [letters[i] for i in tree[1]]
                return problem.get_sym_of_attr(tuple(replaced_item), tree[0])

        if tree[0] in ["Add", "Mul"]:  # operate
            expr_list = []
            for item in tree[1]:
                expr = EqParser.get_expr_from_tree(problem, item, replaced, letters)
                if expr is None:
                    return None
                expr_list.append(expr)
            if tree[0] == "Add":
                result = 0
                for expr in expr_list:
                    result += expr
            else:
                result = 1
                for expr in expr_list:
                    result *= expr
            return result
        elif tree[0] in ["Sub", "Div", "Pow"]:
            expr_left = EqParser.get_expr_from_tree(problem, tree[1][0], replaced, letters)
            if expr_left is None:
                return None
            expr_right = EqParser.get_expr_from_tree(problem, tree[1][1], replaced, letters)
            if expr_right is None:
                return None
            if tree[0] == "Sub":
                return expr_left - expr_right
            elif tree[0] == "Div":
                return expr_left / expr_right
            else:
                return expr_left ** expr_right
        elif tree[0] in ["Sin", "Cos", "Tan"]:
            expr = EqParser.get_expr_from_tree(problem, tree[1][0], replaced, letters)
            if expr is None:
                return None
            if tree[0] == "Sin":
                return sin(expr * pi / 180)
            elif tree[0] == "Cos":
                return cos(expr * pi / 180)
            else:
                return tan(expr * pi / 180)
        else:
            raise RuntimeException("OperatorNotDefined",
                                   "No operation {}, please check your expression.".format(tree[0]))

    @staticmethod
    def get_equation_from_tree(problem, tree, replaced=False, letters=None):
        """Called by function <get_expr_from_tree>."""
        left_expr = EqParser.get_expr_from_tree(problem, tree[0], replaced, letters)
        if left_expr is None:
            return None
        right_expr = EqParser.get_expr_from_tree(problem, tree[1], replaced, letters)
        if right_expr is None:
            return None
        return left_expr - right_expr

    @staticmethod
    def parse_expr(problem, expr):
        """Parse the expression in <str> form into <symbolic> form"""
        expr_list = idt_expr.parseString(expr + "~").asList()
        expr_stack = []
        operator_stack = ["~"]  # 栈底元素

        i = 0
        while i < len(expr_list):
            unit = expr_list[i]
            if unit in operator:  # 运算符
                if stack_priority[operator_stack[-1]] < outside_priority[unit]:
                    operator_stack.append(unit)
                    i = i + 1
                else:
                    operator_unit = operator_stack.pop()
                    if operator_unit == "+":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 + expr_2)
                    elif operator_unit == "-":
                        expr_2 = expr_stack.pop()
                        expr_1 = 0 if len(expr_stack) == 0 else expr_stack.pop()
                        expr_stack.append(expr_1 - expr_2)
                    elif operator_unit == "*":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 * expr_2)
                    elif operator_unit == "/":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 / expr_2)
                    elif operator_unit == "^":
                        expr_2 = expr_stack.pop()
                        expr_1 = expr_stack.pop()
                        expr_stack.append(expr_1 ** expr_2)
                    elif operator_unit == "{":  # 只有unit为"}"，才能到达这个判断
                        i = i + 1
                    elif operator_unit == "@":  # sin
                        expr_1 = expr_stack.pop()
                        expr_stack.append(sin(expr_1))
                    elif operator_unit == "#":  # cos
                        expr_1 = expr_stack.pop()
                        expr_stack.append(cos(expr_1))
                    elif operator_unit == "$":  # tan
                        expr_1 = expr_stack.pop()
                        expr_stack.append(tan(expr_1))
                    elif operator_unit == "~":  # 只有unit为"~"，才能到达这个判断，表示表达式处理完成
                        break
            else:  # 实数或符号
                unit = problem.get_sym_of_attr((unit,), "Free") if unit.isalpha() else float(unit)
                expr_stack.append(unit)
                i = i + 1

        return expr_stack.pop()


def anti_parse_logic_to_cdl(problem, de_redundant=False):
    """
    Anti parse conditions of logic form to CDL.
    Refer to function <anti_parse_one_by_id>.
    """
    problem.gather_conditions_msg()  # gather conditions msg before generate CDL.

    anti_parsed_cdl = {}
    for step in range(len(problem.get_id_by_step)):
        anti_parsed_cdl[step] = []
        for _id in problem.get_id_by_step[step]:
            anti_parsed_cdl[step].append(anti_parse_one_by_id(problem, _id))

    if de_redundant:
        for step in anti_parsed_cdl:
            new_anti_parsed = []
            i = 0
            while i < len(anti_parsed_cdl[step]):
                predicate = anti_parsed_cdl[step][i].split("(")[0]
                if predicate in ["Shape", "Collinear", "Point", "Line", "Angle"]:  # skip
                    i += 1
                    continue
                new_anti_parsed.append(anti_parsed_cdl[step][i])
                if predicate in problem.predicate_GDL["Entity"]:
                    i += len(problem.predicate_GDL["Entity"][predicate]["multi"])
                elif predicate in problem.predicate_GDL["Relation"]:
                    i += len(problem.predicate_GDL["Relation"][predicate]["multi"])
                i += 1
            anti_parsed_cdl[step] = new_anti_parsed

    return anti_parsed_cdl


def anti_parse_one_by_id(problem, _id):
    """
    Anti parse conditions of logic form to CDL.
    ['Shape', ['A', 'B', 'C']]           ==>   'Shape(ABC)'
    ['Parallel', ['A', 'B', 'C', 'D']]   ==>   'Parallel(AB,CD)'
    """
    predicate = problem.get_predicate_by_id[_id]
    condition = problem.conditions[predicate]
    if predicate in list(problem.predicate_GDL["Construction"]) + list(problem.predicate_GDL["Entity"]):
        return predicate + "(" + "".join(condition.get_item_by_id[_id]) + ")"
    elif predicate in problem.predicate_GDL["Relation"]:
        item = []
        i = 0
        for l in problem.predicate_GDL["Relation"][predicate]["para_split"]:
            item.append("")
            for _ in range(l):
                item[-1] += condition.get_item_by_id[_id][i]
                i += 1
        return predicate + "(" + ",".join(item) + ")"
    else:  # equation
        equation = condition.get_item_by_id[_id]
        if len(equation.free_symbols) > 1:
            equation_str = str(condition.get_item_by_id[_id])
            equation_str = equation_str.replace(" ", "")
            return "Equation" + "(" + equation_str + ")"
        else:
            item, predicate = condition.attr_of_sym[list(equation.free_symbols)[0]]
            return predicate + "(" + "".join(item) + ")"


def replace_free_vars_with_letters(free_vars, points, mutex_points):
    """
    Replace free vars with points with mutex_points constrain.
    >> replace_free_vars_with_letters(['A', 'B', 2], ['A', 'B', 'C'], [[0, 1, 2]])
    >> [['A', 'B', 'C']]
    >> replace_free_vars_with_letters(['A', 'B', 2, 3], ['A', 'B', 'C', 'D'], [[0, 1, 2, 3]])
    >> [['A', 'B', 'C', 'D'], ['A', 'B', 'D', 'C']]
    """
    all_para_combination = [free_vars]
    while True:
        len_before_update = len(all_para_combination)
        for i in range(len(all_para_combination)):  # list all possible values for free vars
            for j in range(len(all_para_combination[i])):
                if isinstance(all_para_combination[i][j], int):
                    for point in points:
                        all_para_combination.append(copy.copy(all_para_combination[i]))
                        all_para_combination[-1][j] = point[0]
                    all_para_combination.pop(i)  # delete being replaced para
                    break
        if len_before_update == len(all_para_combination):
            break

    checked_para_combination = []  # filter out combinations that meet the mutex constraint
    for para in all_para_combination:
        checked = True
        for mutex_point in mutex_points:
            if not isinstance(mutex_point[0], list):
                check_points = [para[j] for j in mutex_point]
                if len(set(check_points)) < len(mutex_point):
                    checked = False
                    break
            else:
                check_shapes = ["".join([para[j] for j in mutex_point[i]]) for i in range(len(mutex_point))]
                if len(set(check_shapes)) < len(mutex_point):
                    checked = False
                    break
        if checked:
            checked_para_combination.append(para)

    return checked_para_combination


def build_vars_from_algebraic_relation(t_vars, equal_tree, attr):
    """
    Build vars from algebraic relation.
    >> build_vars_from_algebraic_relation(
           [0, 1, 2, 3],
           ['Equal', [['Length', [0, 1]],['Length', [2, 3]]]],
           [('A', 'B'), 'Length']
       )
    [[0, 1, 'A', 'B'], ['A', 'B', 2, 3]]
    """
    results = []
    _get_all_attr(equal_tree, attr[1], results)
    for i in range(len(results)):
        results[i] = [attr[0][results[i][1].index(v)] if v in results[i][1] else v for v in t_vars]

    return results


def _get_all_attr(equal_tree, target_attr, results):
    """
    Return all attrs same as target attr. Called by function <build_vars_from_algebraic_relation>.
    >> _get_all_attr(
           ['Equal', [['Length', [0, 1]],['Length', [2, 3]]]],
           'Length',
           []
       )
    [['Length', [0, 1]], ['Length', [2, 3]]]
    """
    if not isinstance(equal_tree, list):
        return

    if equal_tree[0] in ["Equal", "Add", "Sub", "Mul", "Div", "Pow"]:
        for item in equal_tree[1]:
            _get_all_attr(item, target_attr, results)
    elif equal_tree[0] in ["Sin", "Cos", "Tan"]:
        _get_all_attr(equal_tree[1][0], target_attr, results)
    elif equal_tree[0] == target_attr:
        results.append(equal_tree)
