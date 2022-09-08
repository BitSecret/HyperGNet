import json
import time
trans_data_path = "../data/geo3k_trans_data/"
gen_data_path = "../data/generated_data/"
trans_filename = "data_trans.json"
gen_filename = "data_gen.json"
count = 1213


def load_data(filename):
    return json.load(open(filename, "r", encoding="utf-8"))


def save_data(path, data_json):
    with open(path + "data_new_{}.json".format(int(time.time())), "w", encoding="utf-8") as f:
        json.dump(data_json, f)


def main():
    data_old = load_data(trans_data_path + trans_filename)    # 载入数据

    data_new = {}    # 新数据保存
    for i in range(0, count):
        data_unit = {
            "problem_id": data_old[str(i)]["problem_id"],
            "annotation": data_old[str(i)]["annotation"],
            "source": data_old[str(i)]["source"],
            "problem_level": data_old[str(i)]["problem_level"],
            "problem_text_cn": data_old[str(i)]["problem_text_cn"],
            "problem_text_en": data_old[str(i)]["problem_text_en"],
            "problem_img": data_old[str(i)]["problem_img"],
            "problem_answer": data_old[str(i)]["problem_answer"],
            "construction_fls": data_old[str(i)]["construction_fls"],
            "text_fls": data_old[str(i)]["text_fls"],
            "image_fls": data_old[str(i)]["image_fls"],
            "target_fls": data_old[str(i)]["target_fls"],
            "theorem_seqs": data_old[str(i)]["theorem_seqs"],
            "completeness": data_old[str(i)]["completeness"]
        }

        data_new[str(i)] = data_unit

    save_data(trans_data_path, data_new)
    print("OK")


if __name__ == '__main__':
    main()
