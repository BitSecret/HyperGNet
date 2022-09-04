import json
from solver import Solver
from func_timeout import FunctionTimedOut
import traceback
test_g3k_tri_file = "F:/geo3k_trans_data/data_tri.json"
test_problem_file = "F:/PGPS/GeoMechanical/test_data/problem.json"
test_define_file = "F:/PGPS/GeoMechanical/test_data/test_define.json"


if __name__ == "__main__":
    solver = Solver()
    while True:
        try:
            problem_index = input("problem id:")
            if problem_index == "-1":
                break
            problem = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))[problem_index]
            solver.new_problem(problem["problem_id"], problem["construction_fls"], problem["text_fls"],
                               problem["image_fls"], problem["target_fls"], problem["theorem_seqs"],
                               problem["problem_answer"])
            solver.solve()
            solver.problem.show()
            print()
        except Exception as e:    # 一般报错
            traceback.print_exc()
        except FunctionTimedOut as e:    # 超时报错
            print("求解方程组超时！")
