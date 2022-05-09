import json
import re
our_path = "F:/PythonProject/geo3k_trans_data/"


def load_data():
    return json.load(open(our_path + "data_tri.json", "r", encoding="utf-8"))


def save_data(data_json):
    with open(our_path + "data_tri.json", "w") as f:
        json.dump(data_json, f)


def alignment_fl(data_json):
    for key in data_json.keys():
        data_one = data_json[key]

        data_one["completeness"] = "True"
        for i in range(len(data_one["formal_language"])):
            fl = data_one["formal_language"][i]
            if fl.startswith("Perpendicular(") and "Point" not in fl:  # 处理垂直
                fl = fl.replace("Perpendicular(", "Perpendicular(Point($),")
                if fl[28] == fl[37]:
                    fl = fl.replace("$", fl[28])
                elif fl[28] == fl[38]:
                    fl = fl.replace("$", fl[28])
                elif fl[29] == fl[37]:
                    fl = fl.replace("$", fl[29])
                elif fl[29] == fl[38]:
                    fl = fl.replace("$", fl[29])

            match_result = re.search(r"\d+[a-z]", fl)  # 处理乘号缺失问题
            if match_result is not None:
                matched = match_result.group()
                fl = re.sub(r"\d+[a-z]", matched[0:-1] + "*" + matched[-1], fl, count=1)

            match_result = re.search(r"\d+{", fl)  # 处理乘号缺失问题
            if match_result is not None:
                matched = match_result.group()
                fl = re.sub(r"\d+{", matched[0:-1] + "*" + matched[-1], fl, count=1)

            if re.search(r"Angle\([A-Z]\)", fl) is not None:    # 角的表示，不合法的标出来
                data_one["completeness"] = "False"
            if re.search(r"Degree\(angle\d+\)", fl) is not None:
                data_one["completeness"] = "False"
            data_one["formal_language"][i] = fl

        data_json[key] = data_one
    return data_json


def alignment_answer(data_json):
    for key in data_json.keys():
        data_one = data_json[key]
        problem_answer = data_one["problem_answer"]

        match_result = re.search(r"\\sqrt\d+", problem_answer)  # 处理 answer sqrt
        if match_result is not None:
            matched = re.search(r"\d+", match_result.group()).group()
            problem_answer = re.sub(r"\\sqrt{\d+}", "{" + "{}^0.5".format(matched) + "}", problem_answer, count=1)

        match_result = re.search(r"\d+{", problem_answer)  # 处理answer乘号缺失问题
        if match_result is not None:
            matched = match_result.group()
            problem_answer = re.sub(r"\d+{", matched[0:-1] + "*" + matched[-1], problem_answer, count=1)

        if problem_answer.startswith("{") and problem_answer.endswith("}"):
            problem_answer = problem_answer[1:-1]

        data_one["problem_answer"] = problem_answer
        data_json[key] = data_one
    return data_json


def show_fl(data_json):
    for key in data_json.keys():
        for fl in data_json[key]["formal_language"]:
            print(fl)
        print()


def show_answer(data_json):
    for key in data_json.keys():
        print(data_json[key]["problem_answer"])

    # for key in data_json.keys():
    #     if "\\" in data_json[key]["problem_answer"]:
    #         print(key)
    #         print(data_json[key]["problem_answer"])


def check_answer(data_json):
    for key in data_json.keys():
        if "\\" in data_json[key]["problem_answer"]:
            print(key)
            print(data_json[key]["problem_answer"])
            data_json[key]["problem_answer"] = input(":")
    return data_json


def main():
    data_json = load_data()
    # data_json = alignment_fl(data_json)
    # data_json = alignment_answer(data_json)

    # show_fl(data_json)
    show_answer(data_json)

    # save_data(data_json)


if __name__ == '__main__':
    main()

