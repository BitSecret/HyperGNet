from gps.utils import get_config
from gps.model import make_model
from gps.data import get_problem_state, make_problem_state_onehot, make_train_val_test_split, graph_collate_fn
from gps.train import get_mark
from gps.utils import nodes_words, edges_words, theorem_words
from formalgeo.tools import load_json, safe_save_json, get_used_pid_and_theorem
from formalgeo.data import DatasetLoader
from formalgeo.problem import Problem
from formalgeo.solver import Interactor
from formalgeo.parse import inverse_parse_one_theorem
from multiprocessing import Process, Queue
from func_timeout import func_timeout, FunctionTimedOut
import torch.nn.functional as F
import torch
import warnings
import random
import time
import argparse
import os
import psutil

config = get_config()
random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed_all(config["random_seed"])

skip_theorems = ["sine_theorem", "cosine_theorem"]


def get_state(beam_stacks):
    state_info = []
    for problem, _ in beam_stacks:
        nodes, serialized_edges, edges_structure, goal = make_problem_state_onehot(*get_problem_state(problem))
        state_info.append([nodes, serialized_edges, edges_structure, goal, [0]])

    nodes, edges, structures, goals, _ = graph_collate_fn(state_info)

    return nodes, edges, structures, goals


def apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam):
    predicted_theorems = F.softmax(predicted_theorems, dim=1)  # convert to a probability distribution
    for i in range(len(beam_stacks)):  # consider previous prob
        predicted_theorems[i] = predicted_theorems[i] * beam_stacks[i][1]
    predicted_theorems = predicted_theorems.flatten()  # flatten for sorting

    sorted_theorems = [(idx, prob)
                       for idx, prob in sorted(enumerate(predicted_theorems), key=lambda x: x[1], reverse=True)]

    new_beam_stacks = []
    prob_sum = 0
    if greedy_beam:  # greedy_beam
        for idx, prob in sorted_theorems:
            if len(new_beam_stacks) == beam_size:  # max len of beam_stacks
                break
            problem = Problem()
            problem.load_problem_by_copy(beam_stacks[int(idx / len(theorem_words))][0])  # which stack
            t_name, t_branch = theorem_words[idx % len(theorem_words)].rsplit('_', 1)  # which theorem
            solver.problem = problem
            if t_name not in skip_theorems and solver.apply_theorem(t_name=t_name, t_branch=t_branch):  # if update
                new_beam_stacks.append([solver.problem, prob])
                prob_sum += prob
    else:
        for idx, prob in sorted_theorems[:beam_size]:
            problem = Problem()
            problem.load_problem_by_copy(beam_stacks[int(idx / len(theorem_words))][0])
            t_name, t_branch = theorem_words[idx % len(theorem_words)].rsplit('_', 1)
            solver.problem = problem
            if t_name not in skip_theorems and solver.apply_theorem(t_name=t_name, t_branch=t_branch):  # if update
                new_beam_stacks.append([solver.problem, prob])
                prob_sum += prob

    for i in range(len(new_beam_stacks)):  # probability normalization
        new_beam_stacks[i][1] = new_beam_stacks[i][1] / prob_sum

    return new_beam_stacks


def solve(problem_id, request_queue, reply_queue, beam_size, greedy_beam):
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)
    solver.load_problem(dl.get_problem(problem_id))

    beam_stacks = [[solver.problem, 1]]  # max_len(beam_stack) = beam_size
    while len(beam_stacks) > 0:
        for problem, _ in beam_stacks:
            problem.check_goal()
            if problem.goal.solved:
                _, seqs = get_used_pid_and_theorem(problem)
                seqs = [inverse_parse_one_theorem(s, solver.parsed_theorem_GDL) for s in seqs]
                return "solved", seqs

        request_queue.put((os.getpid(), problem_id, "predict", get_state(beam_stacks)))

        reply_problem_id, predicted_theorems = reply_queue.get()
        while reply_problem_id != problem_id:  # may get last problem reply when last problem timeout
            reply_problem_id, predicted_theorems = reply_queue.get()

        beam_stacks = apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam)

    return "unsolved", "Out of stacks."


def multiprocess_solve(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout):
    warnings.filterwarnings("ignore")
    while not task_queue.empty():
        problem_id = task_queue.get()
        request_queue.put((os.getpid(), problem_id, "record", "None"))

        timing = time.time()
        try:
            solved, msg = func_timeout(
                timeout=timeout,
                func=solve,
                args=(problem_id, request_queue, reply_queue, beam_size, greedy_beam)
            )
            info = (solved, msg, time.time() - timing)
        except FunctionTimedOut:
            info = ("timeout", f"FunctionTimedOut({timeout})", time.time() - timing)
        except BaseException as e:
            info = ("error", repr(e), time.time() - timing)

        request_queue.put((os.getpid(), problem_id, "write", info))


def main(use_pretrain, use_residual, use_structural_encoding, use_hypertree, beam_size, greedy_beam, timeout, device):
    """Run PAC cycle."""
    mark = get_mark(use_pretrain, use_residual, use_structural_encoding, use_hypertree, beam_size)
    bst_model_path = f"../../data/checkpoints/train_model_{mark}_bst.pth"
    mark += "_gb" if greedy_beam else "_nb"
    mark += f"_tm{timeout}"
    log_path = f"../../data/outputs/log_pac_{mark}.json"

    log = {"processed": [], "solved": {}, "unsolved": {}, "timeout": {}, "error": {}}
    if os.path.exists(log_path):
        log = load_json(log_path)

    task_queue = Queue()
    for problem_id in make_train_val_test_split()["test"]:
        if problem_id in log["processed"]:
            continue
        task_queue.put(problem_id)

    request_queue = Queue()
    reply_queues = {}  # map pid to process queue

    device = torch.device(device)
    model = make_model(use_residual, use_structural_encoding, use_hypertree)
    model.load_state_dict(torch.load(bst_model_path, map_location=torch.device("cpu"))).to(device)

    while True:
        removed = []
        for process_id in reply_queues:  # Remove non-existent pid
            if not psutil.pid_exists(process_id):
                removed.append(process_id)
        for process_id in removed:
            del reply_queues[process_id]

        while not task_queue.empty() and config["multiprocess"] - len(reply_queues) > 0:
            reply_queue = Queue()
            process = Process(target=multiprocess_solve,
                              args=(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout))
            process.start()
            reply_queues[process.pid] = reply_queue

        process_id, problem_id, request, info = request_queue.get()
        if request == "record":  # processed problem
            log["processed"].append(process_id)
            safe_save_json(log, log_path)
        elif request == "write":  # write log
            result, msg, timing = info
            log[result][str(problem_id)] = {"msg": msg, "timing": timing}
            safe_save_json(log, log_path)
            print(f"process_id={process_id}, problem_id={problem_id}, result={result}, timing={timing}")
        elif request == "predict":  # predict theorem
            nodes, edges, structures, goals = info
            reply_queue = reply_queues[process_id]

            predicted_theorems = model(nodes=nodes.to(device), edges=edges.to(device),
                                       structures=structures.to(device), goals=goals.to(device)).cpu()
            reply_queue.put((problem_id, predicted_theorems))


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")

    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")

    parser.add_argument("--use_pretrain", type=lambda x: x == "True", default=True,
                        help="Use pretrain.")
    parser.add_argument("--use_residual", type=lambda x: x == "True", default=False,
                        help="Use residual.")
    parser.add_argument("--use_structural_encoding", type=lambda x: x == "True", default=True,
                        help="Use structural encoding.")
    parser.add_argument("--use_hypertree", type=lambda x: x == "True", default=True,
                        help="Use hypertree.")

    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size when calculate acc.")
    parser.add_argument("--greedy_beam", type=lambda x: x == "True", default=False,
                        help="Use greedy beam search.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout for one problem.")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    kill subprocess:
    python utils.py --func kill --py_filename pac.py
    
    run pac:
    python pac.py
    """
    args = get_args()
    main(use_pretrain=args.use_pretrain, use_residual=args.use_residual,
         use_structural_encoding=args.use_structural_encoding, use_hypertree=args.use_hypertree,
         beam_size=args.beam_size, greedy_beam=args.greedy_beam, timeout=args.timeout, device=args.device)
