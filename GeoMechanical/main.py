import json
import time
import traceback
import func_timeout
from solver import Solver
test_g3k_tri_file = "F:/PythonProject/geo3k_trans_data/data_tri.json"
test_problem_file = "F:/PythonProject/PGPS/GeoMechanical/test_data/problem.json"
test_define_file = "F:/PythonProject/PGPS/GeoMechanical/test_data/test_define.json"


def save_data(data_json):
    with open(test_g3k_tri_file, "w") as f:
        json.dump(data_json, f)


def test_g3k_tri():
    solver = Solver()
    while True:
        problem_index = input("input problem id:")
        try:
            problems = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))[problem_index]
            solver.new_problem(problems["problem_id"], problems["formal_languages"], problems["theorem_seqs"])
            solver.solve()
            answer = solver._parse_expr(problems["problem_answer"])
        except func_timeout.exceptions.FunctionTimedOut:  # 求解超时
            print("求解超时")
            solver.problem.show()
            print()
        except Exception as e:  # 其他错误
            print(traceback.format_exc())
            print()
        else:
            solver.problem.show()
            print("\033[32mcorrect answer: \033[0m{}".format(answer))
            print()


def theorem_test():
    solver = Solver()
    fl = ["Perpendicular(Point(S),Line(TS),Line(RS))"]

    solver.new_problem(0, fl, [])    # 定理10测试
    solver.solve()
    solver.problem.show()


def test_problem():
    solver = Solver()
    problem_index = "0"
    problems = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))[problem_index]
    solver.new_problem(problems["problem_id"], problems["construction_fls"], problems["text_fls"],
                       problems["image_fls"], problems["theorem_seqs"])
    solver.solve()
    # print("\033[32mbasic_equations:\033[0m")
    # for i in solver.problem.basic_equations:
    #     print(i)
    # solver.problem.simplify_basic_equations()
    # print("\033[32mtheorem_equations:\033[0m")
    # for i in solver.problem.theorem_equations:
    #     print(i)
    # print("\033[32mvalue_equations:\033[0m")
    # for i in solver.problem.value_equations:
    #     print(i)

    solver.problem.show()


if __name__ == "__main__":
    # test_g3k_tri()
    # theorem_test()
    test_problem()
