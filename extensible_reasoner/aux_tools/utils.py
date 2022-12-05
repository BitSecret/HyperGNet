import json
# 一些实用的工具，如
# 1.数据处理
# 2.生成定理超图
# 3.可视化解题过程等...


def load_json(filename):
    return json.load(open(filename, "r", encoding="utf-8"))


def save_json(data, filename):
    json.dump(data, open(filename, "w", encoding="utf-8"))


def show_problem(problem):
    pass
