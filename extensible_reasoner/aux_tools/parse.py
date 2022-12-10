import copy
from pyparsing import oneOf, Combine, Word, nums, alphas, OneOrMore

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


def parse_predicate(predicate_GDL):
    """parse predicate_GDL to parsed form."""
    predicate_GDL = predicate_GDL["Predicates"]
    parsed_GDL = {}

    entity = predicate_GDL["Entity"]  # parse entity
    parsed_GDL["Construction"] = {  # preset Construction
        "Shape": {
            "vars": [], "multi": [], "extend": []
        },
        "Polygon": {
            "vars": [], "multi": [], "extend": []
        },
        "Collinear": {
            "vars": [], "multi": [], "extend": []
        }
    }
    parsed_GDL["Entity"] = {  # preset Entity
        "Point": {
            "vars": [0], "multi": [], "extend": []
        },
        "Line": {
            "vars": [0, 1], "multi": [["Line", [1, 0]]], "extend": [["Point", [0]], ["Point", [1]]]
        },
        "Angle": {
            "vars": [0, 1, 2], "multi": [], "extend": [["Line", [0, 1]], ["Line", [1, 2]]]
        }
    }
    for key in entity:
        name, para = _parse_one_predicate(entity[key]["format"])
        parsed_GDL["Entity"][name] = {
            "vars": [i for i in range(len(para))],
            "multi": [],
            "extend": []
        }
        for multi in entity[key]["multi"]:
            extend_name, extend_para = _parse_one_predicate(multi)
            parsed_GDL["Entity"][name]["multi"].append([extend_name, [para.index(i) for i in extend_para]])
        for extend in entity[key]["extend"]:
            extend_name, extend_para = _parse_one_predicate(extend)
            parsed_GDL["Entity"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

    relation = predicate_GDL["Relation"]  # parse relation
    parsed_GDL["Relation"] = {}
    for key in relation:
        name, para, para_len = _parse_one_predicate_attached_len(relation[key]["format"])
        parsed_GDL["Relation"][name] = {
            "vars": [i for i in range(len(para))],
            "para_split": para_len,
            "multi": [],
            "extend": []
        }
        for multi in relation[key]["multi"]:
            extend_name, extend_para = _parse_one_predicate(multi)
            parsed_GDL["Relation"][name]["multi"].append([extend_name, [para.index(i) for i in extend_para]])
        for extend in relation[key]["extend"]:
            extend_name, extend_para = _parse_one_predicate(extend)
            parsed_GDL["Relation"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

    attribution = predicate_GDL["Attribution"]  # parse attribution
    parsed_GDL["Attribution"] = {  # preset Attribution
        "Free": {
            "sym": "f",
            "attr_multi": "False",
            "negative": "True"
        },
        "Length": {
            "sym": "l",
            "attr_multi": "True",
            "negative": "False"
        },
        "Measure": {
            "sym": "m",
            "attr_multi": "False",
            "negative": "True"
        },
        "Area": {
            "sym": "a",
            "attr_multi": "True",
            "negative": "False"
        },
        "Perimeter": {
            "sym": "p",
            "attr_multi": "True",
            "negative": "False"
        },
    }
    for key in attribution:
        name = attribution[key]["format"]
        parsed_GDL["Attribution"][name] = {
            "sym": attribution[key]["sym"],
            "attr_multi": attribution[key]["attr_multi"],
            "negative": attribution[key]["negative"]
        }
    return parsed_GDL


def parse_theorem(theorem_GDL):
    """parse theorem_GDL to parsed form."""
    theorem_GDL = theorem_GDL["Theorems"]
    parsed_GDL = {}

    for key in theorem_GDL:
        theorem_name = theorem_GDL[key]["name"]
        parsed_GDL[theorem_name] = {
            "vars": [],
            "mutex_points": [],
            "premise": {"entity_relation": [],
                        "algebraic_relation": []},
            "conclusion": {"entity_relation": [],
                           "algebraic_relation": []}
        }

        for s in theorem_GDL[key]["premise"]["entity_relation"]:  # replace entity_relation's letter with vars
            name, para = _parse_one_predicate(s)
            for i in range(len(para)):
                if para[i] not in parsed_GDL[theorem_name]["vars"]:
                    parsed_GDL[theorem_name]["vars"].append(para[i])
                para[i] = parsed_GDL[theorem_name]["vars"].index(para[i])
            parsed_GDL[theorem_name]["premise"]["entity_relation"].append([name, para])

        for s in theorem_GDL[key]["premise"]["algebraic_relation"]:  # replace algebraic_relation's letter with vars
            parsed_GDL[theorem_name]["premise"]["algebraic_relation"].append(
                _replace_letter_with_vars(_parse_equal_predicate(s), parsed_GDL[theorem_name]["vars"])
            )

        for mutex_point in theorem_GDL[key]["mutex_points"]:  # replace mutex_point's letter with vars
            if not isinstance(mutex_point[0], list):
                parsed_GDL[theorem_name]["mutex_points"].append(
                    [parsed_GDL[theorem_name]["vars"].index(mutex_point[i])
                     for i in range(len(mutex_point))]
                )
            else:
                parsed_GDL[theorem_name]["mutex_points"].append(
                    [[parsed_GDL[theorem_name]["vars"].index(mutex_point[i][j])
                      for j in range((len(mutex_point[i])))]
                     for i in range(len(mutex_point))]
                )

        for s in theorem_GDL[key]["conclusion"]["entity_relation"]:  # replace entity_relation's letter with vars
            name, para = _parse_one_predicate(s)
            parsed_GDL[theorem_name]["conclusion"]["entity_relation"].append(
                [name, [parsed_GDL[theorem_name]["vars"].index(para[i]) for i in range(len(para))]]
            )

        for s in theorem_GDL[key]["conclusion"]["algebraic_relation"]:  # replace algebraic_relation's letter with vars
            parsed_GDL[theorem_name]["conclusion"]["algebraic_relation"].append(
                _replace_letter_with_vars(_parse_equal_predicate(s), parsed_GDL[theorem_name]["vars"])
            )

        parsed_GDL[theorem_name]["vars"] = [i for i in range(len(parsed_GDL[theorem_name]["vars"]))]  # build vars

    return parsed_GDL


def parse_problem(problem_CDL):
    """parse problem_CDL to parsed form."""
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
        predicate, para = _parse_one_predicate(fl)
        parsed_CDL["parsed_cdl"]["construction_cdl"].append([predicate, para])

    for fl in problem_CDL["text_cdl"] + problem_CDL["image_cdl"]:
        if fl.startswith("Equal"):
            parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append(_parse_equal_predicate(fl))
        else:
            predicate, para = _parse_one_predicate(fl)
            parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append([predicate, para])

    if problem_CDL["goal_cdl"].startswith("Value"):
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "value"
        parsed_CDL["parsed_cdl"]["goal"]["item"] = _parse_equal_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = problem_CDL["problem_answer"]
    elif problem_CDL["goal_cdl"].startswith("Equal"):
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "equal"
        parsed_CDL["parsed_cdl"]["goal"]["item"] = _parse_equal_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = 0
    else:
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "relation"
        predicate, para = _parse_one_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["item"] = predicate
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = para

    # for key in parsed_CDL:
    #     print(key)
    #     print(parsed_CDL[key])
    #     print()

    return parsed_CDL


def _parse_one_predicate(s):
    """
    parse s to get predicate and para list.
    >> parse_one('Predicate(ABC)')
    ('Predicate', ['A', 'B', 'C'])
    >> parse_one('Predicate(ABC, DE)')
    ('Predicate', ['A', 'B', 'C', 'D', 'E'])
    """
    predicate_name, para = s.split("(")
    para = para.split(")")[0]
    if "," not in para:
        return predicate_name, list(para)
    para = para.split(",")
    return predicate_name, list("".join(para))


def _parse_one_predicate_attached_len(s):
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
    return _listing(stack.pop())


def _listing(s_tree):
    """
    Recursive trans s_tree's para to para list.
    >> listing(['Add', [['Length', ['AB']], ['Length', ['CD']]]])
    ['Add', [['Length', ['A', 'B']], ['Length', ['C', 'D']]]]
    """
    if not isinstance(s_tree, list):
        return s_tree

    is_para = True  # Judge whether the bottom layer is reached.
    for para in s_tree:
        if isinstance(para, list):
            is_para = False
            break

    if is_para:
        return list("".join(s_tree))
    else:
        return [_listing(para) for para in s_tree]


def _replace_letter_with_vars(s_tree, s_var):
    """
    Recursive trans s_tree's para to vars.
    >> replace_letter_with_vars(['Add', [['Length', ['A', 'B']], ['Length', ['C', 'D']]]], ['A', 'B', 'C', 'D'])
    ['Add', [['Length', ['0', '1']], ['Length', ['2', '3']]]]
    """
    if not isinstance(s_tree, list):
        return s_tree

    is_para = True  # Judge whether the bottom layer is reached.
    for para in s_tree:
        if isinstance(para, list):
            is_para = False
            break

    if is_para:
        return [s_var.index(para) for para in s_tree]
    else:
        return [_replace_letter_with_vars(para, s_var) for para in s_tree]


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
    ['Shape' ['A', 'B', 'C']]           ==>   'Shape(ABC)'
    ['Parallel' ['A', 'B', 'C', 'D']]   ==>   'Parallel(AB,CD)'
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
        _get_all_attr(equal_tree[1][0], target_attr, results)
        _get_all_attr(equal_tree[1][1], target_attr, results)
    elif equal_tree[0] in ["Sin", "Cos", "Tan"]:
        _get_all_attr(equal_tree[1][0], target_attr, results)
    elif equal_tree[0] == target_attr:
        results.append(equal_tree)
