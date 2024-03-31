from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle, edges_words
from pgps.symbolic_system import tokenize_theorem
from formalgeo.tools import load_json, save_json, safe_save_json, get_used_pid_and_theorem
from formalgeo.data import DatasetLoader
from formalgeo.solver import Interactor
from formalgeo.parse import inverse_parse_one_theorem
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration, get_linear_schedule_with_warmup
import psutil
from multiprocessing import Process, Queue
from func_timeout import func_timeout, FunctionTimedOut
import requests
import json
from tqdm import tqdm
import torch
import warnings
import time
import os

torch.manual_seed(config.torch_seed)
torch.cuda.manual_seed_all(config.cuda_seed)


def tokenize_one_problem(problem_CDL, tokenizer, max_len=512):
    inputs = (problem_CDL["construction_cdl"] + problem_CDL["text_cdl"] +
              problem_CDL["image_cdl"] + [problem_CDL["goal_cdl"]])
    inputs = tokenizer.encode(",".join(inputs))
    if len(inputs) > max_len:
        inputs = inputs[:max_len]

    outputs = [edges_words.index(tokenize_theorem(t)) for t in problem_CDL["theorem_seqs"]]

    return inputs, outputs


def make_train_data():
    train_data_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/train.pk"))
    val_data_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/val.pk"))
    test_data_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/test.pk"))
    if os.path.exists(train_data_path):
        return load_pickle(train_data_path), load_pickle(val_data_path)

    print("Make training data...")
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    problem_split = dl.get_problem_split()["split"]  # official partition

    train_data = []
    for pid in problem_split["train"]:
        problem_CDL = dl.get_problem(pid)
        train_data.append(tokenize_one_problem(problem_CDL, tokenizer))
    save_pickle(train_data, train_data_path)

    val_data = []
    for pid in problem_split["val"]:
        problem_CDL = dl.get_problem(pid)
        val_data.append(tokenize_one_problem(problem_CDL, tokenizer))
    save_pickle(val_data, val_data_path)

    test_data = {}
    for pid in problem_split["test"]:
        problem_CDL = dl.get_problem(pid)
        inputs, outputs = tokenize_one_problem(problem_CDL, tokenizer)
        test_data[pid] = inputs
    save_pickle(test_data, test_data_path)

    return train_data, val_data


def collate_fn(batch):  # padding
    inputs, outputs = zip(*batch)

    inputs = [torch.LongTensor(i) for i in inputs]
    outputs = [torch.LongTensor(i) for i in outputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
    return inputs, outputs


def test():
    best_model_save_path = os.path.normpath(os.path.join(config.path_data, "trained_model/inter_gps_model_best.pt"))
    test_data_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/test.pk"))
    model_config_path = os.path.normpath(os.path.join(config.path_data, "trained_model/model_config.json"))
    result_save_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/predicted.json"))

    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    test_data = load_pickle(test_data_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BartForConditionalGeneration(BartConfig.from_dict(load_json(model_config_path))).to(device)  # init model
    model.load_state_dict(torch.load(best_model_save_path, map_location=device))

    predicted_seqs = {}
    for pid in dl.get_problem_split()["split"]["test"]:
        inputs = torch.LongTensor(test_data[pid]).unsqueeze(0).to(device)
        outputs = model.generate(inputs, bos_token_id=0, eos_token_id=2,
                                 max_length=25, num_beams=10, num_return_sequences=5)
        predicted_seqs[pid] = []
        for i in range(len(outputs)):
            predicted_seqs[pid].append([t for t in outputs[i].tolist() if t not in [0, 1, 2]])

        print(f"{pid} ok.")

    save_json(predicted_seqs, result_save_path)


def train():
    model_config_url = 'https://huggingface.co/facebook/bart-base/raw/main/config.json'
    pretrained_model_url = 'https://acl2021-intergps.s3.us-west-1.amazonaws.com/tp_model_best.pt'
    model_config_path = os.path.normpath(os.path.join(config.path_data, "trained_model/model_config.json"))
    pretrained_model_path = os.path.normpath(os.path.join(config.path_data, "trained_model/tp_model_best.pt"))
    best_model_save_path = os.path.normpath(os.path.join(config.path_data, "trained_model/inter_gps_model_best.pt"))

    if not os.path.exists(pretrained_model_path):
        print("Download pretrained model (Inter-GPS)...")
        model_config = json.loads(requests.get(model_config_url).text)
        save_json(model_config, model_config_path)
        pretrained_model = requests.get(pretrained_model_url, stream=True)
        with open(pretrained_model_path, "wb") as file:
            for data in pretrained_model.iter_content(1024):  # block_size = 1024
                file.write(data)

    train_data, val_data = make_train_data()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
    model = BartForConditionalGeneration(BartConfig.from_dict(load_json(model_config_path))).to(device)  # init model
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))  # use inter-GPS pretrained para
    optimizer = torch.optim.AdamW(model.parameters(), 3e-5)  # learning rate = 3e-5
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, len(train_loader) * 20)  # lr adjustment

    best_loss = 1e6  # for early-stop
    MAX_EPOCH = 20
    for epoch in tqdm(range(MAX_EPOCH)):
        total_loss = 0
        model.train()
        for idx, (inputs, outputs) in enumerate(train_loader):  # every train batch
            optimizer.zero_grad()
            res = model(inputs.to(device), labels=outputs[:, 1:].contiguous().to(device),
                        decoder_input_ids=outputs[:, :-1].contiguous().to(device))
            loss = res.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print('\nepoch: ', epoch, " train_loss ", total_loss)

        total_loss = 0
        model.eval()  # evaluate
        for idx, (inputs, outputs) in enumerate(val_loader):
            res = model(inputs.to(device), labels=outputs[:, 1:].contiguous().to(device),
                        decoder_input_ids=outputs[:, :-1].contiguous().to(device))
            loss = res.loss
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), best_model_save_path)
        print('epoch: ', epoch, " val_loss ", total_loss)


def solve_one(problem_id, dl, predicted_theorem, step_size, debug=False):
    problem_CDL = dl.get_problem(problem_id)
    if debug:
        warnings.filterwarnings("ignore")
        print(f"problem_id: {problem_id}")
        print(f"ground_truth: {problem_CDL['theorem_seqs']}")
    theorem_seqs = predicted_theorem[str(problem_id)]
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)

    for i in range(len(theorem_seqs)):
        solver.load_problem(problem_CDL)
        seqs = theorem_seqs[i]
        if debug:
            print(f"round {i}: ", end="")
        for theorem_id in seqs:
            if debug:
                print(f"{edges_words[theorem_id]}, ", end="")
            t_name, t_branch = edges_words[theorem_id].rsplit('_', 1)
            solver.apply_theorem(t_name=t_name, t_branch=t_branch)
            step_size["count"] += 1

            solver.problem.check_goal()
            if solver.problem.goal.solved:
                _, seqs = get_used_pid_and_theorem(solver.problem)
                seqs = [inverse_parse_one_theorem(s, solver.parsed_theorem_GDL) for s in seqs]
                if debug:
                    print()
                return "solved", seqs
        if debug:
            print()

    return "unsolved", "Out of stacks."


def solve(dl, predicted_theorem, task_queue, reply_queue):
    warnings.filterwarnings("ignore")

    while not task_queue.empty():
        problem_id = task_queue.get()
        timing = time.time()
        step_size = {"count": 0}

        try:
            solved, msg = func_timeout(
                600, solve_one, args=(problem_id, dl, predicted_theorem, step_size))
            state = solved
            info = (msg, time.time() - timing, step_size["count"])
        except FunctionTimedOut:
            state = "timeout"
            info = (f"FunctionTimedOut({600})", time.time() - timing, step_size["count"])
        except BaseException as e:
            state = "error"
            info = (repr(e), time.time() - timing, step_size["count"])
        reply_queue.put((os.getpid(), problem_id, state, info))


def start_process(dl, predicted_theorem, task_queue, reply_queue, process_ids):
    """Remove non-existent pid and start new process"""
    for i in range(len(process_ids))[::-1]:
        if process_ids[i] not in psutil.pids():
            process_ids.pop(i)

    while not task_queue.empty() and config.process_count - len(process_ids) > 0:
        process = Process(target=solve, args=(dl, predicted_theorem, task_queue, reply_queue))
        process.start()
        process_ids.append(process.pid)


def evaluate_acc(predicted_theorem_path, log_save_path):
    if not os.path.exists(log_save_path):
        save_json({"solved": {}, "unsolved": {}, "timeout": {}, "error": {}}, log_save_path)
    predicted_theorem = load_json(predicted_theorem_path)
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    log = load_json(log_save_path)

    solved_pids = (list(log["solved"].keys()) + list(log["unsolved"].keys()) +
                   list(log["timeout"].keys()) + list(log["error"].keys()))
    task_queue = Queue()
    for problem_id in predicted_theorem:
        if str(problem_id) not in solved_pids:
            task_queue.put(int(problem_id))

    reply_queue = Queue()
    process_ids = []
    while True:
        start_process(dl, predicted_theorem, task_queue, reply_queue, process_ids)
        process_id, problem_id, state, info = reply_queue.get()
        log[state][problem_id] = {"msg": info[0], "timing": info[1], "step_size": info[2]}
        safe_save_json(log, log_save_path)
        print("{} {} {}".format(state, problem_id, info))


def main():
    # train()

    # test()

    predicted_theorem_path = os.path.normpath(os.path.join(config.path_data, "training_data/inter-gps/predicted.json"))
    log_save_path = os.path.normpath(os.path.join(config.path_data, "log/experiments/inter_gps.json"))

    # dl = DatasetLoader(config.dataset_name, config.path_datasets)
    # predicted_theorem = load_json(predicted_theorem_path)
    # step_size = {"count": 0}
    # state, msg = solve_one(12, dl, predicted_theorem, step_size, True)
    # print(state)
    # print(msg)

    evaluate_acc(predicted_theorem_path, log_save_path)


main()

