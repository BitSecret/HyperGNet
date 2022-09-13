import json
import time
import xlrd
trans_data_path = "../data/geo3k_trans_data/"
gen_data_path = "../data/generated_data/"
trans_filename = "trans.json"
gen_filename = "gen.json"
count = 1213


def load_data(filename):
    return json.load(open(filename, "r", encoding="utf-8"))


def save_data(path, data_json):
    with open(path + "data_new_{}.json".format(int(time.time())), "w", encoding="utf-8") as f:
        json.dump(data_json, f)


def change_theorem_seqs_map(theorem_map, theorem_seqs):
    new_theorem_seqs = []
    for seq in theorem_seqs:
        if theorem_map[seq] != -1:
            new_theorem_seqs.append(theorem_map[seq])

    return new_theorem_seqs


def load_theorem_seqs_map():
    file_theorem_map = "../GeoMechanical/doc/formal_language.xlsx"
    theorem_map = {}
    table = xlrd.open_workbook(file_theorem_map).sheets()[1]
    theorem_count = table.nrows  # 获取该sheet中的有效行数
    for i in range(1, theorem_count):
        if int(table.cell_value(i, 0)) == -1:
            print("{}: Theorem.{},".format(int(table.cell_value(i, 1)), table.cell_value(i, 2)))
        else:
            theorem_map[int(table.cell_value(i, 0))] = int(table.cell_value(i, 1))
            if int(table.cell_value(i, 1)) != -1:
                print("{}: Theorem.{},".format(int(table.cell_value(i, 1)), table.cell_value(i, 2)))
    return theorem_map


def delete(fls):
    result = []
    for fl in fls:
        if not fl.startswith("Extended") and not fl.startswith("PointOn"):
            result.append(fl)
    return result


def main():
    theorem_map = load_theorem_seqs_map()
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
            "construction_fls": delete(data_old[str(i)]["construction_fls"]),
            "text_fls": data_old[str(i)]["text_fls"],
            "image_fls": delete(data_old[str(i)]["image_fls"]),
            "target_fls": data_old[str(i)]["target_fls"],
            "theorem_seqs": change_theorem_seqs_map(theorem_map, data_old[str(i)]["theorem_seqs"])
        }

        data_new[str(i)] = data_unit

    save_data(trans_data_path, data_new)
    print("{} problems processed and saved.".format(len(data_new)))


if __name__ == '__main__':
    main()
