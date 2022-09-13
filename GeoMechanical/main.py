import json
from solver import Solver
from func_timeout import FunctionTimedOut
import traceback
geo3k_trans_data = "../data/geo3k_trans_data/trans.json"
template_data = "../data/generated_data/temp.json"
generated_data = "../data/generated_data/gen.json"


def mode_0(solver):
    while True:
        try:
            problem_index = input("problem id:")
            if problem_index == "-1":
                break
            problem = json.load(open(geo3k_trans_data, "r", encoding="utf-8"))[problem_index]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.show()
        except Exception:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")


def mode_1(solver):
    count = int(input("problem max count:"))
    problems = json.load(open(geo3k_trans_data, "r", encoding="utf-8"))

    i = 0
    while i <= count:
        try:
            problem = problems[str(i)]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.simpel_show()
        except Exception:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")

        i += 1


def mode_2(solver):
    while True:
        try:
            problem_index = input("problem id:")
            if problem_index == "-1":
                break
            problem = json.load(open(template_data, "r", encoding="utf-8"))[problem_index]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.show()
        except Exception as e:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")


def mode_3(solver):
    problems = json.load(open(generated_data, "r", encoding="utf-8"))
    for key in problems.keys():
        problem = problems[key]
        solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                           problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                           problem["problem_answer"])
        solver.solve()
        solver.problem.simpel_show()


def main():
    mode = 0
    solver = Solver()

    if mode == 0:    # trans
        mode_0(solver)
    elif mode == 1:    # trans-auto
        mode_1(solver)
    elif mode == 2:    # gen-temp
        mode_2(solver)
    elif mode == 3:    # gen
        mode_3(solver)


if __name__ == "__main__":
    main()
