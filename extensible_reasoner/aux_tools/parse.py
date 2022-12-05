# 解析操作
# 1.解析定义语言(谓词啦、定理啦)
# 2.解析描述语言（）
# 3.反向生成FL
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
        "fls": {
            "construction_fls": problem_CDL["construction_fls"],
            "text_fls": problem_CDL["text_fls"],
            "image_fls": problem_CDL["image_fls"],
            "target_fls": problem_CDL["target_fls"]
        },
        "parsed_fls": {
            "construction_fls": [],
            "text_and_image_fls": [],
            "target": {},
        }
    }
    for fl in problem_CDL["construction_fls"]:
        predicate, para = parse_one_predicate(fl)
        parsed_CDL["parsed_fls"]["construction_fls"].append([predicate, para])

    for fl in problem_CDL["text_fls"] + problem_CDL["image_fls"]:
        if fl.startswith("Equal"):
            s = parse_equal_predicate(fl)
            s[0] = "Equation"
            parsed_CDL["parsed_fls"]["text_and_image_fls"].append(s)
        else:
            predicate, para = parse_one_predicate(fl)
            parsed_CDL["parsed_fls"]["text_and_image_fls"].append([predicate, para])

    if problem_CDL["target_fls"].startswith("Value"):
        parsed_CDL["parsed_fls"]["target"]["type"] = "value"
        parsed_CDL["parsed_fls"]["target"]["item"] = parse_equal_predicate(problem_CDL["target_fls"])
        parsed_CDL["parsed_fls"]["target"]["answer"] = problem_CDL["problem_answer"]
    elif problem_CDL["target_fls"].startswith("Equal"):
        parsed_CDL["parsed_fls"]["target"]["type"] = "equal"
        parsed_CDL["parsed_fls"]["target"]["item"] = parse_equal_predicate(problem_CDL["target_fls"])
        parsed_CDL["parsed_fls"]["target"]["answer"] = parse_equal_predicate(problem_CDL["target_fls"])
    else:
        parsed_CDL["parsed_fls"]["target"]["type"] = "relation"
        predicate, para = parse_one_predicate(problem_CDL["target_fls"])
        parsed_CDL["parsed_fls"]["target"]["item"] = [predicate, para]
        parsed_CDL["parsed_fls"]["target"]["answer"] = [predicate, para]

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
