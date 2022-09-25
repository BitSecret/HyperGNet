import json
import time
count = 89
theorem_test_data = "../data/theorem_test_data/theo.json"


def save_data(filename, data_json):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_json, f)


def main():
    data_new = {}    # 新数据保存
    for i in range(count):
        data_unit = {
            "problem_id": i + 1,
            "annotation": "theorem_2022-09-22",
            "source": "theorem_{}".format(i + 1),
            "problem_level": 1,
            "problem_text_cn": "",
            "problem_text_en": "",
            "problem_img": "",
            "problem_answer": [],
            "construction_fls": [],
            "text_fls": [],
            "image_fls": [],
            "target_fls": [],
            "theorem_seqs": [i + 1]
        }

        data_new[str(i + 1)] = data_unit

    save_data(theorem_test_data, data_new)
    print("{} problems processed and saved.".format(len(data_new)))


if __name__ == '__main__':
    main()
