import json
from solver import Solver
from func_timeout import FunctionTimedOut
import traceback
import time
geo3k_trans_data = "../data/geo3k_trans_data/trans.json"
template_data = "../data/generated_data/temp.json"
generated_data = "../data/generated_data/gen.json"
theorem_test_data = "../data/theorem_test_data/theo.json"
solution_data = "./solution_data/"


# geo3k_trans_data, 一次运行1个题目，并输出解题过程
def geo3k_normal_mode():
    solver = Solver()
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
            solver.problem.save(solution_data + "g3k_normal/")    # 保存求解树和每一步解题
        except Exception:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")


# geo3k_trans_data, 一次运行n个题目，并输出求解结果
def geo3k_step_mode():
    solver = Solver()
    count = int(input("problem max count:"))
    problems = json.load(open(geo3k_trans_data, "r", encoding="utf-8"))
    time_start = time.time()
    theorems = []

    i = 0
    while i <= count:
        try:
            problem = problems[str(i)]
            theorems += problem["theorem_seqs"]
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

    print()
    print("Used theorem list:", end=" ")
    print(list(set(theorems)))
    time_cons = time.time() - time_start
    print("{} problems. time consuming: {:.6f}s. avg: {} s/problem".format(count, time_cons, time_cons / count))


# geo3k_trans_data, 一次运行n个题目，并输出求解结果
def geo3k_embedding_mode():
    solver = Solver()
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
            solver.problem.save(solution_data + "g3k_normal/")
        except Exception:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")

        i += 1


# template_data, 一次运行1个题目，并输出解题过程
def template_normal_mode():
    solver = Solver()
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


# generated_data, 一次运行所有题目，并输出求解结果
def generated_step_mode():
    solver = Solver()
    problems = json.load(open(generated_data, "r", encoding="utf-8"))
    for key in problems.keys():
        problem = problems[key]
        solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                           problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                           problem["problem_answer"])
        solver.solve()
        solver.problem.simpel_show()


# theorem_test_data, 一次运行1个题目，并输出解题过程
def theorem_normal_mode():
    solver = Solver()
    while True:
        try:
            problem_index = input("problem id:")
            if problem_index == "-1":
                break
            problem = json.load(open(theorem_test_data, "r", encoding="utf-8"))[problem_index]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.show()
        except Exception:  # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:  # 超时报错
            print("求解方程组超时！")


if __name__ == "__main__":
    geo3k_embedding_mode()
