from solver import Solver
import json


def main():
    problems_path = "./test_data/problems_data.json"
    data = load_data(problems_path)[0]
    solver = Solver(data["problem_index"], data["formal_languages"], data["theorem_seqs"])
    solver.solve()


def load_data(problems_path):    # 读取json数据并解析成列表
    problem_data = json.load(open(problems_path, "r"))  # 文本 Formal Language
    return list(problem_data.values())


if __name__ == "__main__":
    main()
