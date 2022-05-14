import json
from solver import Solver
test_g3k_tri_file = "F:/PythonProject/geo3k_trans_data/data_tri.json"
test_problem_file = "F:/PythonProject/PGPS/GeoMechanical/test_data/problem.json"
test_define_file = "F:/PythonProject/PGPS/GeoMechanical/test_data/test_define.json"


def save_data(data_json):
    with open(test_g3k_tri_file, "w") as f:
        json.dump(data_json, f)


if __name__ == "__main__":
    solver = Solver()
    problem_index = "8"
    problems = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))[problem_index]
    solver.new_problem(problems["problem_id"], problems["construction_fls"], problems["text_fls"],
                       problems["image_fls"], problems["theorem_seqs"], problems["problem_answer"])
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
