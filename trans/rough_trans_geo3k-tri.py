import json
import os
import re
import shutil
geo3k_path = "F:/PythonProject/InterGPS/data/geometry3k/train/"
our_path = "F:/PythonProject/geo3k_trans_data/"


def get_fl(file_path, problem_type):    # 得到形式化语言
    data_filename = file_path + "data.json"
    logic_filename = file_path + "logic_form.json"
    if json.load(open(data_filename, "r", encoding="utf-8"))["problem_type_graph"][0] != problem_type:
        return None

    problem_logic = json.load(open(logic_filename, "r", encoding="utf-8"))
    problem_fl = problem_logic["diagram_logic_form"] + problem_logic["dissolved_text_logic_form"]
    return problem_fl


def get_problem(file_path):
    problem = json.load(open(file_path + "data.json", "r", encoding="utf-8"))
    text = problem["problem_text"]
    answer = problem["compact_choices"][ord(problem["answer"]) - ord("A")]
    return text, answer


def alignment_fl(fls):    # 将Geometry3k数据转化为我们的表示形式
    for i in range(len(fls)):
        fls[i] = fls[i].replace(" ", "")
        fls[i] = fls[i].replace("Of", "")
        fls[i] = fls[i].replace("Is", "")
        fls[i] = fls[i].replace("Equals", "Equal")
        fls[i] = fls[i].replace("Measure", "Degree")
        fls[i] = fls[i].replace("IntersectAt", "Intersect")
        fls[i] = fls[i].replace("PointLiesOnLine", "PointOn")
        fls[i] = fls[i].replace("PointLiesOnCircle", "PointOn")

        match_result = re.search(r"PointOn\([A-Z],", fls[i])  # Point
        if match_result is not None:
            fls[i] = re.sub(r"PointOn\([A-Z],", "PointOn(Point({}),".format(match_result.group()[8]), fls[i], count=1)

        match_result = re.search(r"\\sqrt{\d+}", fls[i])    # 识别转化sqrt
        if match_result is not None:
            matched = re.search(r"\d+", match_result.group()).group()
            fls[i] = re.sub(r"\\sqrt{\d+}", "{" + "{}^0.5".format(matched) + "}", fls[i], count=1)

        match_result = re.search(r"\\frac{\d+}{\d+}", fls[i])    # 识别转化frac
        if match_result is not None:
            matched = re.findall(r"\d+", match_result.group())
            fls[i] = re.sub(r"\\frac{\d+}{\d+}", "{}/{}".format(matched[0], matched[1]), fls[i], count=1)

        match_result = re.search(r"[A-Z],[A-Z],[A-Z],[A-Z]", fls[i])  # 去掉,
        if match_result is not None:
            matched = re.findall(r"[A-Z]", match_result.group())
            fls[i] = re.sub(r"[A-Z],[A-Z],[A-Z],[A-Z]",
                            "{}{}{}{}".format(matched[0], matched[1], matched[2], matched[3]), fls[i], count=1)

        match_result = re.search(r"[A-Z],[A-Z],[A-Z]", fls[i])  # 去掉,
        if match_result is not None:
            matched = re.findall(r"[A-Z]", match_result.group())
            fls[i] = re.sub(r"[A-Z],[A-Z],[A-Z]", "{}{}{}".format(matched[0], matched[1], matched[2]), fls[i], count=1)

        match_result = re.search(r"[A-Z],[A-Z]", fls[i])  # 去掉,
        if match_result is not None:
            matched = re.findall(r"[A-Z]", match_result.group())
            fls[i] = re.sub(r"[A-Z],[A-Z]", "{}{}".format(matched[0], matched[1]), fls[i], count=1)

        # fls[i] = fls[i].replace("", "")
        # fls[i] = fls[i].replace("", "")
    return fls


def alignment_answer(answer):
    match_result = re.search(r"\\sqrt{\d+}", answer)  # 识别转化sqrt
    if match_result is not None:
        matched = re.search(r"\d+", match_result.group()).group()
        answer = re.sub(r"\\sqrt{\d+}", "{" + "{}^0.5".format(matched) + "}", answer, count=1)

    match_result = re.search(r"\\frac{\d+}{\d+}", answer)  # 识别转化frac
    if match_result is not None:
        matched = re.findall(r"\d+", match_result.group())
        answer = re.sub(r"\\frac{\d+}{\d+}", "{}/{}".format(matched[0], matched[1]), answer, count=1)

    return answer


def main():
    data_json = {}
    count = 0
    problem_type = "Triangle"
    problem_dirs = os.listdir(geo3k_path)
    for i in range(3001):
        if str(i) not in problem_dirs:
            continue

        problem_dir = geo3k_path + str(i) + "/"    # 问题文件夹
        fls = get_fl(problem_dir, problem_type)    # 获得指定类型的问题

        if fls is not None:
            problem_text, problem_answer = get_problem(problem_dir + "/")
            data_json_unit = {
                "problem_id": count,
                "annotation": "xiaokaizhang_2022-04-28",
                "source": "Geometry3k-{}".format(i),
                "problem_type": [problem_type],
                "problem_text": problem_text,
                "problem_img": "{}_g3k-{}.png".format(count, i),
                "problem_answer": alignment_answer(problem_answer),
                "formal_language": alignment_fl(alignment_fl(fls)),    # 转化为 ours 形式
                "theorem_seqs": []
            }
            shutil.copyfile(problem_dir + "img_diagram.png", our_path + "{}_g3k-{}.png".format(count, i))
            data_json[str(count)] = data_json_unit
            count = count + 1

    with open(our_path + "data_tri.json", "w") as f:
        json.dump(data_json, f)


if __name__ == '__main__':
    main()

