import json
from aux_tools.parse import anti_parse_logic_to_cdl
from sympy import Float


def load_json(filename):
    return json.load(open(filename, "r", encoding="utf-8"))


def save_json(data, filename):
    json.dump(data, open(filename, "w", encoding="utf-8"))


def show(problem):
    """-----------Conditional Declaration Statement-----------"""
    print("\033[36mproblem_index:\033[0m", end=" ")
    print(problem.id)
    print("\033[36mconstruction_cdl:\033[0m")
    for construction_fl in problem.fl["construction_cdl"]:
        print(construction_fl)
    print("\033[36mtext_cdl:\033[0m")
    for text_fl in problem.fl["text_cdl"]:
        print(text_fl)
    print("\033[36mimage_cdl:\033[0m")
    for image_fl in problem.fl["image_cdl"]:
        print(image_fl)
    print("\033[36mgoal_cdl:\033[0m")
    print(problem.fl["goal_cdl"])
    print()

    """-----------Process of Problem Solving-----------"""
    print("\033[36mtheorem_applied:\033[0m")
    for i in range(len(problem.theorems_applied)):
        print("{} {}".format(i, problem.theorems_applied[i]))
    print("\033[36mreasoning_cdl:\033[0m")
    problem.gather_conditions_msg()
    anti_parsed_logic = anti_parse_logic_to_cdl(problem)
    for step in anti_parsed_logic:
        print("step: {}".format(step))
        for cdl in anti_parsed_logic[step]:
            print(cdl)
    print()

    used_id = []
    if problem.goal["solved"]:
        used_id = list(set(problem.goal["premise"]))
        while True:
            len_used_id = len(used_id)
            for _id in used_id:
                if _id >= 0:
                    used_id += list(problem.conditions[problem.get_predicate_by_id[_id]].premises[_id])
                    used_id = list(set(used_id))
            if len_used_id == len(used_id):
                break

    """-----------Logic Form-----------"""
    print("\033[33mRelation:\033[0m")
    predicates = ["Point", "Line", "Angle", "Shape", "Collinear"]
    predicates += list(problem.predicate_GDL["Entity"].keys())
    predicates += list(problem.predicate_GDL["Relation"].keys())
    for predicate in predicates:
        condition = problem.conditions[predicate]
        if len(condition.get_item_by_id) > 0:
            print(predicate + ":")
            for _id in condition.get_item_by_id:
                if _id not in used_id:
                    print("{0:^6}{1:^15}{2:^25}{3:^6}".format(
                        _id,
                        ",".join(condition.get_item_by_id[_id]),
                        str(condition.premises[_id]),
                        condition.theorems[_id])
                    )
                else:
                    print("\033[35m{0:^6}{1:^15}{2:^25}{3:^6}\033[0m".format(
                        _id,
                        ",".join(condition.get_item_by_id[_id]),
                        str(condition.premises[_id]),
                        condition.theorems[_id])
                    )

    print("\033[33mEntity attribution\'s Symbol and Value:\033[0m")
    equation = problem.conditions["Equation"]
    for attr in equation.sym_of_attr.keys():
        sym = equation.sym_of_attr[attr]
        if isinstance(equation.value_of_sym[sym], Float):
            print("{0:^10}{1:^10}{2:^15.3f}".format(str(attr[1]), str(sym), equation.value_of_sym[sym]))
        else:
            print("{0:^10}{1:^10}{2:^15}".format(str(attr[1]), str(sym), str(equation.value_of_sym[sym])))

    print("\033[33mEquations:\033[0m")
    if len(equation.get_item_by_id) > 0:
        for _id in equation.get_item_by_id:
            if _id not in used_id:
                print_str = "{0:^6}{1:^70}{2:^25}{3:>6}"
            else:
                print_str = "\033[35m{0:^6}{1:^70}{2:^25}{3:>6}\033[0m"
            if len(equation.premises[_id]) > 5:
                print(print_str.format(_id,
                                       str(equation.get_item_by_id[_id]),
                                       str(equation.premises[_id][0:5]) + "...",
                                       equation.theorems[_id]))
            else:
                print(print_str.format(_id,
                                       str(equation.get_item_by_id[_id]),
                                       str(equation.premises[_id]),
                                       equation.theorems[_id]))
    print()

    # goal
    print("\033[34mSolving Goal:\033[0m")
    print("type: {}".format(str(problem.goal["type"])))
    print("goal: {}".format(str(problem.goal["item"])))
    print("correct answer: {}".format(str(problem.goal["answer"])))
    if problem.goal["solved"]:
        print("solved: \033[32m{}\033[0m".format(str(problem.goal["solved"])))
    else:
        print("solved: \033[31m{}\033[0m".format(str(problem.goal["solved"])))
    if problem.goal["solved_answer"] is not None:
        print("solved_answer: {}".format(str(problem.goal["solved_answer"])))
        print("premise: {}".format(str(problem.goal["premise"])))
        print("theorem: {}".format(str(problem.goal["theorem"])))
    print()

    print("\033[34mTime consumption:\033[0m")
    for s in problem.goal["solving_msg"]:
        print(s)

