import json
import time
import traceback
import func_timeout
from solver import Solver
test_g3k_tri_file = "F:/PythonProject/geo3k_trans_data/data_tri.json"


def save_data(data_json):
    with open(test_g3k_tri_file, "w") as f:
        json.dump(data_json, f)


def test_g3k_tri():
    problem_index = "2"
    problems = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))[problem_index]
    solver = Solver(problems["problem_id"], problems["formal_languages"], problems["theorem_seqs"])
    solver.solve()
    solver.problem.show()


def test_g3k_simple():
    problems = json.load(open(test_g3k_tri_file, "r", encoding="utf-8"))
    i = 0
    time_start = time.time()
    for key in problems.keys():
        if problems[key]["completeness"] == "True":
            try:
                solver = Solver(problems[key]["problem_id"],
                                problems[key]["formal_languages"],
                                [5, 3, 1])
                solver.solve()
            except func_timeout.exceptions.FunctionTimedOut:    # 求解超时
                pass
            except Exception:    # 其他错误
                pass
            else:
                if solver.problem.target_solved[0] == "solved":
                    problems[key]["theorem_seqs"] = [5, 3, 1]
                    solver.problem.show()
                    i = i + 1
    print("总时间花费: {:.6f}".format(time.time() - time_start))
    print("解题成功数: {}".format(i))
    save_data(problems)


if __name__ == "__main__":
    test_g3k_tri()
