from formalgeo.data import download_dataset, DatasetLoader
from formalgeo.tools import load_json
import matplotlib.pyplot as plt
import os
import zipfile
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


def init_project():
    """Build dir and download dataset."""
    config = get_config()

    dirs = [config["data"]["datasets_path"], "../../data/checkpoints", "../../data/outputs", "../../data/training_data"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    if not os.path.exists(f"{config['data']['datasets_path']}/{config['data']['dataset_name']}.json"):
        download_dataset(dataset_name=config["data"]["dataset_name"], datasets_path=config["data"]["datasets_path"])


# def result_alignment(log, test_pids, timeout=600):
#     """
#     Alignment PAC results.
#     1.select the test set data.
#     2.set the max search time as timeout.
#     3.set the unhandled and error problems as unsolved and set it timeout as <timeout> * 0.5.
#     """
#     alignment_log = {"solved": {}, "unsolved": {}, "timeout": {}}
#     for pid in test_pids:  # 1.select the test set data.
#         pid = str(pid)
#         added = False
#         for key in alignment_log:
#             if pid in log[key]:
#                 if log[key][pid]["timing"] > timeout:  # 2.set the max search time as timeout.
#                     alignment_log["timeout"][pid] = log[key][pid]
#                     alignment_log["timeout"][pid]["timing"] = timeout
#                 else:
#                     alignment_log[key][pid] = log[key][pid]
#                 added = True
#                 break
#         if not added:  # 3.set the unhandled and error problems as unsolved and set it timeout as <timeout> * 0.5.
#             alignment_log["unsolved"][pid] = {"msg": "Unhandled problems.", "timing": timeout * 0.5, "step_size": 1}
#
#     return alignment_log
#
#


def show_contrast_results(level=6, span=2):
    log_files = {
        "HyperGNet-NB-600": "../../data/outputs/log_pac_TFTT_bs5_nb_tm600.json",
        "HyperGNet-GB-600": "../../data/outputs/log_pac_TFTT_bs5_gb_tm600.json"
    }
    problem_level = {}  # map problem_id to level
    problem_total = [0 for _ in range(level + 1)]  # total number of a level

    config = get_config()
    dl = DatasetLoader(config["data"]["dataset_name"], config["data"]["datasets_path"])
    test_problem_ids = load_json("../../data/outputs/problem_split.json")["test"]
    level_map = {}  # map t_length to level (start from 0)
    for i in range(level):
        for j in range(span):
            level_map[i * span + j + 1] = i + 1
    for pid in test_problem_ids:
        t_length = len(dl.get_problem(pid)["theorem_seqs"])
        problem_level[pid] = level_map[t_length] if t_length <= level * span else level
        problem_total[0] += 1
        problem_total[problem_level[pid]] += 1

    for model_name in log_files.keys():
        print(model_name, end="\t")
        model_results = [0 for _ in range(level + 1)]  # model solved problem
        for pid in load_json(log_files[model_name])["solved"]:
            model_results[0] += 1
            model_results[problem_level[int(pid)]] += 1
        for i in range(level + 1):
            print(round(model_results[i] / problem_total[i] * 100, 2), end="\t")
        print()


def show_ablation_results():
    pass


#     evaluation_data_path = os.path.normpath(os.path.join(Configuration.path_data, "log/experiments/{}"))
#     figure_save_path = os.path.normpath(os.path.join(Configuration.path_data, "log/{}"))
#     dl = DatasetLoader(Configuration.dataset_name, Configuration.path_datasets)
#     test_pids = dl.get_problem_split()["split"]["test"]
#
#     level_count = 6
#     level_map = {}
#     for pid in test_pids:
#         t_length = dl.get_problem(pid)["problem_level"]
#         if t_length <= 2:
#             level_map[pid] = 0
#         elif t_length <= 4:
#             level_map[pid] = 1
#         elif t_length <= 6:
#             level_map[pid] = 2
#         elif t_length <= 8:
#             level_map[pid] = 3
#         elif t_length <= 10:
#             level_map[pid] = 4
#         else:
#             level_map[pid] = 5
#     level_total = [0 for _ in range(level_count)]
#     for pid in test_pids:
#         level_total[level_map[pid]] += 1
#
#     contrast_log_files = {  # table 1
#         "Inter-GPS-600": result_alignment(
#             load_json(evaluation_data_path.format("inter_gps.json")),
#             test_pids, 600),
#         "HyperGNet-NB-30": result_alignment(
#             load_json(evaluation_data_path.format("pac_log_pretrain_beam_5.json")),
#             test_pids, 30),
#         "HyperGNet-GB-30": result_alignment(
#             load_json(evaluation_data_path.format("pac_log_pretrain_greedy_beam_5.json")),
#             test_pids, 30),
#         "HyperGNet-GB-600": result_alignment(
#             load_json(evaluation_data_path.format("pac_log_pretrain_greedy_beam_5.json")),
#             test_pids, 600)
#     }
#     contrast_filenames = list(contrast_log_files.keys())
#     dl = DatasetLoader(Configuration.dataset_name, Configuration.path_datasets)
#     test_pids = dl.get_problem_split()["split"]["test"]
#
#     table_data = [[0 for _ in range(level_count)] for _ in range(len(contrast_log_files))]
#     for i in range(len(contrast_filenames)):
#         for pid in test_pids:
#             if str(pid) in contrast_log_files[contrast_filenames[i]]["solved"]:
#                 table_data[i][level_map[pid]] += 1
#
#     print("Table 1:")
#     print("total" + "".join([f"\tl{i + 1}" for i in range(level_count)]))
#     for i in range(len(contrast_filenames)):
#         print(contrast_filenames[i], end="")
#         print("\t{:.2f}".format(len(contrast_log_files[contrast_filenames[i]]["solved"]) / len(test_pids) * 100),
#               end="")
#         for j in range(level_count):
#             print("\t{:.2f}".format(table_data[i][j] / level_total[j] * 100), end="")
#         print()
#     print()
#
#     step_wised_log_files = [  # table 3
#         load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_1.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_3.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_5.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_1.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_3.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_5.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_1.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_3.json")),
#         load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_5.json"))
#     ]
#     pac_log_files = [  # table 3
#         result_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_1.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_3.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_5.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_1.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_3.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_5.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_1.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_3.json")), test_pids),
#         result_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_5.json")), test_pids)
#     ]
#     print("Table 2:")
#     print("step_wised_acc\toverall_acc\tavg_time\tavg_step")
#     for i in range(len(step_wised_log_files)):
#         time_sum = 0
#         step_sum = 0
#         for key in pac_log_files[i]:
#             for pid in pac_log_files[i][key]:
#                 time_sum += pac_log_files[i][key][pid]["timing"]
#                 step_sum += pac_log_files[i][key][pid]["step_size"]
#         print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
#             step_wised_log_files[i]["acc"] * 100,
#             len(pac_log_files[i]["solved"]) / len(test_pids) * 100,
#             time_sum / len(test_pids),
#             step_sum / len(test_pids)
#         ))
#
#     timing = [[0 for _ in range(level_count)] for _ in range(len(contrast_filenames))]
#     step = [[0 for _ in range(level_count)] for _ in range(len(contrast_filenames))]
#
#     for i in range(len(contrast_filenames)):
#         for key in contrast_log_files[contrast_filenames[i]]:
#             for pid in contrast_log_files[contrast_filenames[i]][key]:
#                 timing[i][level_map[int(pid)]] += contrast_log_files[contrast_filenames[i]][key][pid]["timing"]
#                 step[i][level_map[int(pid)]] += contrast_log_files[contrast_filenames[i]][key][pid]["step_size"]
#
#     for i in range(len(contrast_filenames)):
#         for j in range(level_count):
#             timing[i][j] /= level_total[j]
#             step[i][j] /= level_total[j]
#
#     x = [i + 1 for i in range(level_count)]
#     fontsize = 24
#     axis_fontsize = 15
#     line_width = 3
#     plt.figure(figsize=(16, 8))  # figure 1
#
#     plt.subplot(131)
#     i = contrast_filenames.index("HyperGNet-NB-30")
#     y = [table_data[i][j] / level_total[j] * 100 for j in range(level_count)]
#     plt.plot(x, y, label="HyperGNet-NB", linewidth=line_width)
#     i = contrast_filenames.index("HyperGNet-GB-30")
#     y = [table_data[i][j] / level_total[j] * 100 for j in range(level_count)]
#     plt.plot(x, y, label="HyperGNet-GB", linewidth=line_width)
#     plt.xlabel("Problem Difficulty", fontsize=fontsize)
#     plt.ylabel("Acc (%)", fontsize=fontsize)
#     plt.legend(loc="upper right", fontsize=axis_fontsize)
#     plt.tick_params(axis='both', labelsize=axis_fontsize)
#
#     plt.subplot(132)
#     plt.plot(x, timing[contrast_filenames.index("HyperGNet-NB-30")], label="HyperGNet-NB", linewidth=line_width)
#     plt.plot(x, timing[contrast_filenames.index("HyperGNet-GB-30")], label="HyperGNet-GB", linewidth=line_width)
#     plt.xlabel("Problem Difficulty", fontsize=fontsize)
#     plt.ylabel("Avg Time (s)", fontsize=fontsize)
#     plt.legend(loc="upper left", fontsize=axis_fontsize)
#     plt.tick_params(axis='both', labelsize=axis_fontsize)
#
#     plt.subplot(133)
#     plt.plot(x, step[contrast_filenames.index("HyperGNet-NB-30")], label="HyperGNet-NB", linewidth=line_width)
#     plt.plot(x, step[contrast_filenames.index("HyperGNet-GB-30")], label="HyperGNet-GB", linewidth=line_width)
#     plt.xlabel("Problem Difficulty", fontsize=fontsize)
#     plt.ylabel("Avg Step", fontsize=fontsize)
#     plt.legend(loc="upper left", fontsize=axis_fontsize)
#     plt.tick_params(axis='both', labelsize=axis_fontsize)
#
#     plt.tight_layout()
#     plt.savefig(figure_save_path.format("acc_time_step.pdf"), format='pdf')
#     plt.show()


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
    python utils.py --func init_project
    python utils.py --func show_contrast_results
    python utils.py --func show_ablation_results
    python utils.py --func kill
    """
    args = get_args()
    if args.func == "test_env":
        test_env()
    if args.func == "init_project":
        init_project()
    elif args.func == "show_contrast_results":
        show_contrast_results()
    elif args.func == "show_ablation_results":
        show_ablation_results()
    elif args.func == "kill":
        clean_process(args.py_filename)
