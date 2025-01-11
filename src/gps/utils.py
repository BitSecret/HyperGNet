from formalgeo.data import DatasetLoader
from formalgeo.tools import load_json
import os
import pickle
import psutil
import torch
import formalgeo
import argparse
import platform


def test_env():
    print("formalgeo.__version__: {}".format(formalgeo.__version__))

    print("torch.__version__: {}".format(torch.__version__))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

    device_count = torch.cuda.device_count()
    print("torch.cuda.device_count(): {}".format(device_count))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    for i in range(device_count):
        print("Device {}: {}".format(i, torch.cuda.get_device_name(i)))


def get_config():
    """Load configuration."""
    config = load_json("../../data/config.json")
    if platform.system() == 'Linux':
        config["data"]["datasets_path"] = config["data"]["datasets_path_linux"]
    config["multiprocess"] = int(psutil.cpu_count() * config["multiprocess"])

    return config


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def download_dataset():
    """Build dir and download dataset."""
    config = get_config()

    if not os.path.exists(config["data"]["datasets_path"]):
        os.makedirs(config["data"]["datasets_path"])
        formalgeo.data.download_dataset(dataset_name=config["data"]["dataset_name"],
                                        datasets_path=config["data"]["datasets_path"])


def get_level_count(problem_level, log, level):
    total_level_count = [0 for _ in range(level + 1)]  # [total, l1, l2, ...]
    solved_level_count = [0 for _ in range(level + 1)]

    for pid in log["total"]:
        total_level_count[0] += 1
        total_level_count[problem_level[pid]] += 1
        if str(pid) in log["solved"]:
            solved_level_count[0] += 1
            solved_level_count[problem_level[pid]] += 1

    return total_level_count, solved_level_count


def show_contrast_results(level=6, span=2):
    log_files = {
        "Forward Search": "../../data/outputs/log_pssr_fw-rs.json",
        "Backward Search": "../../data/outputs/log_pssr_bw-bfs.json",

        "FGeo with T5-small": "../../data/outputs/log_pssr_t5-small_bs20_timeout600.json",
        "FGeo with BART-base": "../../data/outputs/log_pssr_bart-base_bs20_timeout600.json",

        # "GPT-4o": "../../data/outputs/log_pssr_gpt4o.json",
        "DeepSeek-v3": "../../data/outputs/log_pssr_deepseek.json",

        "Inter-GPS": "../../data/outputs/log_pssr_intergps.json",
        "NGS": "../../data/outputs/log_pssr_ngs_bs10_timeout600.json",
        "DualGeoSolver": "../../data/outputs/log_pssr_dualgeosolver_bs10_timeout600.json",
        "FGeo-TP": [80.86, 96.43, 85.44, 76.12, 62.26, 48.88, 29.55],  # FGeo-TP-Results.csv
        "FGeo-DRL": "../../data/outputs/log_pssr_fgeodrl.json",
        "FGeo-HyperGNet": "../../data/outputs/log_pac_TTT_bs5_gb_tm600.json",
        "FGeo-HyperGNet on Geometry3K": "../../data/outputs/log_pssr_geometry3k.json",
        "FGeo-HyperGNet on GeoQA": "../../data/outputs/log_pssr_geoqa.json"
    }
    methods_max_len = max([len(m) for m in log_files])
    problem_level = {}  # map problem_id to level
    level_map = {}  # map t_length to level (start from 0)
    for i in range(level):
        for j in range(span):
            level_map[i * span + j + 1] = i + 1
    config = get_config()
    dl = DatasetLoader(config["data"]["dataset_name"], config["data"]["datasets_path"])
    for pid in range(1, dl.info["problem_number"] + 1):
        t_length = len(dl.get_problem(pid)["theorem_seqs"])
        problem_level[pid] = level_map[t_length] if t_length <= level * span else level

    print("--------------------------------------------------------------------------------------")
    for method in log_files.keys():  # log
        print(method + "".join([" "] * (methods_max_len - len(method))), end="\t")
        results = log_files[method]
        if isinstance(results, list):
            if len(results) != 0:
                for result in results:
                    print(round(result, 2), end="\t")
        else:
            total_level_count, solved_level_count = get_level_count(problem_level, load_json(results), level)
            for i in range(level + 1):
                print(round(solved_level_count[i] / total_level_count[i] * 100, 2), end="\t")
        print()
        if method in ["Backward Search", "DeepSeek-v3", "FGeo with BART-base", "FGeo-HyperGNet"]:
            print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")


def read_last_line(file_path):
    with open(file_path, 'rb') as file:
        file.seek(0, 2)
        file_size = file.tell()
        buffer_size = 1024
        buffer = b""
        position = file_size

        while position > 0:
            step = min(buffer_size, position)
            position -= step
            file.seek(position)
            buffer = file.read(step) + buffer
            if b'\n' in buffer:
                break

        lines = buffer.splitlines()
        return lines[-1].decode('utf-8') if lines else None


def show_ablation_results():
    methods = ["FGeo-HyperGNet", "-w/o Pretrain", "-w/o SE", "-w/o Hypertree"]
    methods_max_len = max([len(m) for m in methods])
    marks = ["TTT", "FTT", "TFT", "TTF"]
    beams = [1, 3, 5]
    print("-----------------------------")
    for i in range(4):
        for j in range(3):
            if j == 1:
                print(methods[i] + "".join([" "] * (methods_max_len - len(methods[i]))), end="\t")
            else:
                print("".join([" "] * methods_max_len), end="\t")

            tpa = read_last_line(f"../../data/outputs/results_train_bst_model_test_{marks[i]}_bs{beams[j]}.txt")
            tpa = round(float(tpa.split("acc: ")[1].split(", ")[0]) * 100, 2)
            print(tpa, end="\t")
            pssr = load_json(f"../../data/outputs/log_pac_{marks[i]}_bs{beams[j]}_bs_tm60.json")
            pssr = round(len(pssr["solved"]) / len(pssr["total"]) * 100, 2)
            print(pssr, end="\t")
            print()
        print("-----------------------------")


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["test_env", "init_project", "show_contrast_results", "show_ablation_results", "kill"],
                        help="function that you want to run")
    parser.add_argument("--py_filename", type=str, required=False,
                        help="python filename that you want to kill")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


def clean_process(py_filename):
    for pid in psutil.pids():
        process = psutil.Process(pid)
        if process.name() == "python" and py_filename in process.cmdline():
            print(f"kill -9 {pid}")


if __name__ == '__main__':
    """
    python utils.py --func test_env
    python utils.py --func download_dataset
    python utils.py --func show_contrast_results
    python utils.py --func show_ablation_results
    python utils.py --func kill
    """
    print("Contrast Results:")
    show_contrast_results()
    print("\n\nAblation Results:")
    show_ablation_results()
    args = get_args()
    if args.func == "test_env":
        test_env()
    if args.func == "download_dataset":
        download_dataset()
    elif args.func == "show_contrast_results":
        show_contrast_results()
    elif args.func == "show_ablation_results":
        show_ablation_results()
    elif args.func == "kill":
        clean_process(args.py_filename)
