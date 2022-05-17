import json
import re
import time
our_path = "F:/PythonProject/geo3k_trans_data/"


def load_data():
    return json.load(open(our_path + "data_tri.json", "r", encoding="utf-8"))


def save_data(data_json):
    with open(our_path + "data_new_{}.json".format(int(time.time())), "w", encoding="utf-8") as f:
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
    data_old = load_data()
    data_new = {}
    for i in range(0, 1213):
        if i <= 20:
            data_unit = {
                "problem_id": data_old[str(i)]["problem_id"],
                "annotation": data_old[str(i)]["annotation"],
                "source": data_old[str(i)]["source"],
                "problem_type": data_old[str(i)]["problem_type"],
                "problem_text_cn": data_old[str(i)]["problem_text_cn"],
                "problem_text_en": data_old[str(i)]["problem_text_en"],
                "problem_img": data_old[str(i)]["problem_img"],
                "problem_answer": data_old[str(i)]["problem_answer"],
                "construction_fls": data_old[str(i)]["construction_fls"],
                "text_fls": data_old[str(i)]["text_fls"],
                "image_fls": data_old[str(i)]["image_fls"],
                "theorem_seqs": data_old[str(i)]["theorem_seqs"],
                "completeness": data_old[str(i)]["completeness"]
            }
        else:
            data_unit = {
                "problem_id": data_old[str(i)]["problem_id"],
                "annotation": data_old[str(i)]["annotation"],
                "source": data_old[str(i)]["source"],
                "problem_type": data_old[str(i)]["problem_type"],
                "problem_text_cn": data_old[str(i)]["problem_text_cn"],
                "problem_text_en": data_old[str(i)]["problem_text_en"],
                "problem_img": data_old[str(i)]["problem_img"],
                "problem_answer": data_old[str(i)]["problem_answer"],
                "construction_fls": data_old[str(i)]["construction_fls"],
                "text_fls":  data_old[str(i)]["text_fls"],
                "image_fls": data_old[str(i)]["image_fls"],
                "theorem_seqs": data_old[str(i)]["theorem_seqs"],
                "completeness": data_old[str(i)]["completeness"]
            }

        data_new[str(i)] = data_unit
    print(data_new)
    save_data(data_new)


if __name__ == '__main__':
    main()

