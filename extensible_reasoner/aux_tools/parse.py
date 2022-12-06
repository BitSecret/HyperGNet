# 解析操作
# 1.解析定义语言(谓词啦、定理啦)
# 2.解析描述语言（）
# 3.反向生成FL
from pyparsing import oneOf, Combine, Word, nums, alphas, OneOrMore
from sympy import sin, cos, tan, pi
from definition.exception import RuntimeException
float_idt = Combine(OneOrMore(Word(nums)) + "." + OneOrMore(Word(nums)))  # 浮点数
int_idt = OneOrMore(Word(nums))  # 整数
alpha_idt = Word(alphas)  # 符号
operator_idt = oneOf("+ - * / ^ { } @ # $ ~")  # 运算符
idt_expr = OneOrMore(float_idt | int_idt | alpha_idt | operator_idt)
operator = ["+", "-", "*", "/", "^", "{", "}", "@", "#", "$", "~"]
stack_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                  "{": 0, "}": None,
                  "@": 4, "#": 4, "$": 4, "~": 0}
outside_priority = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3,
                    "{": 5, "}": 0,
                    "@": 4, "#": 4, "$": 4, "~": 0}


def parse_predicate(predicate_GDL):
    """parse predicate_GDL to parsed form."""
    parsed_GDL = {}

    entity = predicate_GDL["Entity"]  # parse entity
    parsed_GDL["Entity"] = {}
    for key in entity:
        name, para = parse_one_predicate(entity[key]["format"])
        parsed_GDL["Entity"][name] = {
            "vars": [i for i in range(len(para))],
            "extend": []
        }
        for extend in entity[key]["extend"]:
            extend_name, extend_para = parse_one_predicate(extend)
            parsed_GDL["Entity"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

    relation = predicate_GDL["Relation"]  # parse relation
    parsed_GDL["Relation"] = {}
    for key in relation:
        name, para, para_len = parse_one_predicate_attached_len(relation[key]["format"])
        parsed_GDL["Relation"][name] = {
            "vars": [i for i in range(len(para))],
            "para_split": para_len,
            "extend": []
        }
        for extend in relation[key]["extend"]:
            extend_name, extend_para = parse_one_predicate(extend)
            parsed_GDL["Relation"][name]["extend"].append([extend_name, [para.index(i) for i in extend_para]])

    attribution = predicate_GDL["Attribution"]  # parse attribution
    parsed_GDL["Attribution"] = {}
    for key in attribution:
        name = attribution[key]["format"]
        parsed_GDL["Attribution"][name] = {
            "sym": attribution[key]["sym"],
            "multi_rep": attribution[key]["multi_rep"],
            "negative": attribution[key]["negative"]
        }

    return parsed_GDL


def parse_theorem(theorem_GDL):
    """parse theorem_GDL to parsed form."""
    parsed_GDL = {}

    for key in theorem_GDL:
        theorem_name = theorem_GDL[key]["name"]
        parsed_GDL[theorem_name] = {
            "vars": [],
            "premise": {"entity_relation": [],
                        "algebraic_relation": []},
            "conclusion": {"entity_relation": [],
                           "algebraic_relation": []}
        }

        for s in theorem_GDL[key]["premise"]["entity_relation"]:
            name, para = parse_one_predicate(s)
            for i in range(len(para)):
                if para[i] not in parsed_GDL[theorem_name]["vars"]:
                    parsed_GDL[theorem_name]["vars"].append(para[i])
                para[i] = parsed_GDL[theorem_name]["vars"].index(para[i])
            parsed_GDL[theorem_name]["premise"]["entity_relation"].append([name, para])

        for s in theorem_GDL[key]["premise"]["algebraic_relation"]:
            s_tree = parse_equal_predicate(s)
            s_tree = replace_letter_with_vars(s_tree, parsed_GDL[theorem_name]["vars"])
            parsed_GDL[theorem_name]["premise"]["algebraic_relation"].append(s_tree)

        for s in theorem_GDL[key]["conclusion"]["entity_relation"]:
            name, para = parse_one_predicate(s)
            for i in range(len(para)):
                para[i] = parsed_GDL[theorem_name]["vars"].index(para[i])
            parsed_GDL[theorem_name]["conclusion"]["entity_relation"].append([name, para])

        for s in theorem_GDL[key]["conclusion"]["algebraic_relation"]:
            s_tree = parse_equal_predicate(s)
            s_tree = replace_letter_with_vars(s_tree, parsed_GDL[theorem_name]["vars"])
            parsed_GDL[theorem_name]["conclusion"]["algebraic_relation"].append(s_tree)

        parsed_GDL[theorem_name]["vars"] = [i for i in range(len(parsed_GDL[theorem_name]["vars"]))]

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
        predicate, para = parse_one_predicate(fl)
        parsed_CDL["parsed_cdl"]["construction_cdl"].append([predicate, para])

    for fl in problem_CDL["text_cdl"] + problem_CDL["image_cdl"]:
        if fl.startswith("Equal"):
            parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append(parse_equal_predicate(fl))
        else:
            predicate, para = parse_one_predicate(fl)
            parsed_CDL["parsed_cdl"]["text_and_image_cdl"].append([predicate, para])

    if problem_CDL["goal_cdl"].startswith("Value"):
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "value"
        parsed_CDL["parsed_cdl"]["goal"]["item"] = parse_equal_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = problem_CDL["problem_answer"]
    elif problem_CDL["goal_cdl"].startswith("Equal"):
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "equal"
        parsed_CDL["parsed_cdl"]["goal"]["item"] = parse_equal_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = 0
    else:
        parsed_CDL["parsed_cdl"]["goal"]["type"] = "relation"
        predicate, para = parse_one_predicate(problem_CDL["goal_cdl"])
        parsed_CDL["parsed_cdl"]["goal"]["item"] = predicate
        parsed_CDL["parsed_cdl"]["goal"]["answer"] = para

    return parsed_CDL


def parse_one_predicate(s):
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


def parse_one_predicate_attached_len(s):
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


def parse_equal_predicate(s):
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
    return listing(stack.pop())


def listing(s_tree):
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
        return [listing(para) for para in s_tree]


def replace_letter_with_vars(s_tree, s_var):
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
        return [replace_letter_with_vars(para, s_var) for para in s_tree]


def get_expr_from_tree(tree, equation, replaced=False, letters=None):
    """
    Recursively trans expr_tree to symbolic algebraic expression.
    :param tree: An expression in the form of a list tree.
    :param equation: Used to get symbolic representation for attr or free sym.
    :param replaced: Optional. Set True when tree's item is expressed by vars.
    :param letters: Optional. Letters that will replace vars.
    >> get_expr_from_tree_data(['Length', ['T', 'R']], equation)
    l_tr
    >> get_expr_from_tree_data(['Add', [['Length', ['Z', 'X']], '2*x-14']], equation)
    2.0*f_x + l_zx - 14.0
    >> get_expr_from_tree_data(['Sin', [['Measure', ['Z', 'X', 'Y']]]], equation)
    sin(pi*m_zxy/180)
    """
    if not isinstance(tree, list):  # expr
        return parse_expr(tree, equation)

    if tree[0] in equation.attr_GDL:    # attr
        if not replaced:
            return equation.get_sym_of_attr(tree[0], tuple(tree[1]))
        else:
            replaced_item = [letters[i] for i in tree[1]]
            return equation.get_sym_of_attr(tree[0], tuple(replaced_item))

    if tree[0] == "Add":    # operate
        return get_expr_from_tree(tree[1][0], equation) + get_expr_from_tree(tree[1][1], equation)
    elif tree[0] == "Sub":
        return get_expr_from_tree(tree[1][0], equation) - get_expr_from_tree(tree[1][1], equation)
    elif tree[0] == "Mul":
        return get_expr_from_tree(tree[1][0], equation) * get_expr_from_tree(tree[1][1], equation)
    elif tree[0] == "Div":
        return get_expr_from_tree(tree[1][0], equation) / get_expr_from_tree(tree[1][1], equation)
    elif tree[0] == "Pow":
        return get_expr_from_tree(tree[1][0], equation) ** get_expr_from_tree(tree[1][1], equation)
    elif tree[0] == "Sin":
        return sin(get_expr_from_tree(tree[1][0], equation) * pi / 180)
    elif tree[0] == "Cos":
        return cos(get_expr_from_tree(tree[1][0], equation) * pi / 180)
    elif tree[0] == "Tan":
        return tan(get_expr_from_tree(tree[1][0], equation) * pi / 180)
    else:
        raise RuntimeException("OperatorNotDefined", "No operation {}, please check your expression.".format(tree[0]))


def parse_expr(expr, equation):
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
            unit = equation.get_sym_of_attr("Free", (unit,)) if unit.isalpha() else float(unit)
            expr_stack.append(unit)
            i = i + 1

    return expr_stack.pop()


def get_equation_from_equal_tree_para(e_tree, equation, replaced=False, letters=None):
    """
    Refer to func <get_expr_from_tree> for details.
    >> get_expr_from_tree_data([['Length', ['T', 'R']], ['Add', [['Length', ['Z', 'X']], '2*x-14']]], equation)
    l_tr - 2.0*f_x - l_zx + 14.0
    """
    left_expr = get_expr_from_tree(e_tree[0], equation, replaced, letters)
    right_expr = get_expr_from_tree(e_tree[1], equation, replaced, letters)
    return left_expr - right_expr


def anti_parse_logic_to_cdl(problem):
    """Anti parse conditions of logic form to CDL"""
    anti_parsed_cdl = {}
    for step in range(len(problem.get_id_by_step)):
        anti_parsed_cdl[step] = []
        for _id in problem.get_id_by_step[step]:
            predicate = problem.get_predicate_by_id[_id]
            condition = problem.conditions[predicate]
            if predicate in ["Shape", "Collinear", "Point", "Line", "Angle"] + list(problem.predicate_GDL["Entity"]):
                anti_parsed_cdl[step].append([predicate, "".join(condition.get_item_by_id[_id])])
            elif predicate in problem.predicate_GDL["Relation"]:
                item = []
                i = 0
                for l in problem.predicate_GDL["Relation"][predicate]["para_split"]:
                    item.append("")
                    for _ in range(l):
                        item[-1] += condition.get_item_by_id[_id][i]
                        i += 1
                anti_parsed_cdl[step].append([predicate, item])
            else:    # equation
                equation = condition.get_item_by_id[_id]
                if len(equation.free_symbols) > 1:
                    anti_parsed_cdl[step].append(["Equation", str(condition.get_item_by_id[_id])])
                else:
                    item, predicate = condition.attr_of_sym[list(equation.free_symbols)[0]]
                    anti_parsed_cdl[step].append(["predicate", "".join(item)])
    return anti_parsed_cdl
