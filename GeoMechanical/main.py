import json
from solver import Solver
from func_timeout import FunctionTimedOut
import traceback
geo3k_trans_data = "../data/geo3k_trans_data/trans.json"
template_data = "../data/generated_data/temp.json"
generated_data = "../data/generated_data/gen.json"


if __name__ == "__main__":
    mode = 2    # 0 trans   1 temp    2 gen
    solver = Solver()
    if mode == 0:
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
            except Exception as e:    # 一般报错
                traceback.print_exc()
            except FunctionTimedOut as e:    # 超时报错
                print("求解方程组超时！")
    elif mode == 1:
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
            except Exception as e:    # 一般报错
                traceback.print_exc()
            except FunctionTimedOut as e:    # 超时报错
                print("求解方程组超时！")
    else:
        problems = json.load(open(generated_data, "r", encoding="utf-8"))
        for key in problems.keys():
            problem = problems[key]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.simpel_show()

