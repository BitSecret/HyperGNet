import json
from solver import Solver


def main():
    # test_define()
    test_problem()


def load_data(problems_path):  # 读取json数据并解析成列表
    problem_data = json.load(open(problems_path, "r"))  # 文本 Formal Language
    return list(problem_data.values())


def test_define():
    problems_path = "./test_data/test_define.json"
    data = load_data(problems_path)
    for i in range(0, len(data)):
        try:
            solver = Solver(data[i]["problem_index"], data[i]["formal_languages"], data[i]["theorem_seqs"])
            # solver.solve()
            solver.problem.show_problem()
        except Exception as e:
            print(e)
        print()
        a = input("输入1继续执行：")


def test_problem():
    problems_path = "./test_data/test_problem.json"
    data = load_data(problems_path)[5]
    solver = Solver(data["problem_index"], data["formal_languages"], data["theorem_seqs"])
    solver.solve()
    solver.problem.show_problem()


if __name__ == "__main__":
    main()
