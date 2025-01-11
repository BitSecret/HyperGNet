from gps.utils import get_config
from gps.model import make_model
from gps.data import get_problem_state, make_problem_state_onehot, make_train_val_test_split, graph_collate_fn
from gps.train import get_train_mark
from gps.data import theorem_words, pid_alive
from formalgeo.tools import load_json, safe_save_json, get_used_pid_and_theorem
from formalgeo.data import DatasetLoader
from formalgeo.problem import Problem
from formalgeo.solver import Interactor
from formalgeo.parse import inverse_parse_one_theorem
from multiprocessing import Process, Queue
from func_timeout import func_timeout, FunctionTimedOut
import torch
import warnings
import random
import time
import argparse
import os

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
    predicted_theorems = predicted_theorems.softmax(dim=1)  # convert to a probability distribution
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

    for i in range(len(new_beam_stacks)):
        problem, prob = new_beam_stacks[i]
        prob = prob / prob_sum  # probability normalization
        new_beam_stacks[i] = [problem, prob]

    return new_beam_stacks


def solve(problem_id, request_queue, reply_queue, beam_size, greedy_beam):
    dl = DatasetLoader(config["data"]["dataset_name"], config["data"]["datasets_path"])
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
        reply_problem_id, predicted_theorems = None, None
        while reply_problem_id is None or reply_problem_id != problem_id:
            if not reply_queue.empty():
                reply_problem_id, predicted_theorems = reply_queue.get()

        beam_stacks = apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam)

    return "unsolved", "Out of stacks."


def multiprocess_solve(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout):
    warnings.filterwarnings("ignore")
    while not task_queue.empty():
        problem_id = task_queue.get()

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


def main(use_pretrain, use_structural_encoding, use_hypertree,
         beam_size, greedy_beam, timeout,
         device, solve_again):
    """Run PAC cycle."""
    mark = get_train_mark(use_pretrain, use_structural_encoding, use_hypertree)
    bst_model_path = f"../../data/checkpoints/train_model_bst_{mark}.pth"
    mark += f"_bs{beam_size}"
    mark += "_gb" if greedy_beam else "_bs"
    mark += f"_tm{timeout}"
    log_path = f"../../data/outputs/log_pac_{mark}.json"
    test_problem_ids = make_train_val_test_split()["test"]

    log = {"total": test_problem_ids, "solved": {}, "unsolved": {}, "timeout": {}, "error": {}}
    if os.path.exists(log_path):
        log = load_json(log_path)

    task_queue = Queue()
    problem_ids = []
    if solve_again:  # clear unsolved, timeout, and error, run again
        log["unsolved"] = {}
        log["timeout"] = {}
        log["error"] = {}

    for problem_id in test_problem_ids:
        if str(problem_id) in log["solved"]:
            continue
        if str(problem_id) in log["unsolved"]:
            continue
        if str(problem_id) in log["timeout"]:
            continue
        if str(problem_id) in log["error"]:
            continue
        problem_ids.append(problem_id)

    random.shuffle(problem_ids)
    for problem_id in problem_ids:
        task_queue.put(problem_id)
    request_queue = Queue()
    reply_queues = {}  # map pid to process queue

    model = make_model(use_structural_encoding, use_hypertree)
    model.load_state_dict(torch.load(bst_model_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    model = model.to(device)
    count = 0
    while True:
        removed = []
        for process_id in reply_queues:  # Remove non-existent pid
            if not pid_alive(process_id):
                removed.append(process_id)
        for process_id in removed:
            del reply_queues[process_id]

        while not task_queue.empty() and config["multiprocess"] - len(reply_queues) > 0:
            reply_queue = Queue()
            process = Process(target=multiprocess_solve,
                              args=(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout))
            process.start()
            reply_queues[process.pid] = reply_queue

        if not request_queue.empty():  # directly calling .get() will block
            process_id, problem_id, request, info = request_queue.get()
            if request == "write":  # write log
                count += 1
                result, msg, timing = info
                log[result][str(problem_id)] = {"msg": msg, "timing": timing}
                safe_save_json(log, log_path)
                print(f"({count}/{len(problem_ids)}) process_id={process_id}, problem_id={problem_id}, "
                      f"result={result}, timing={timing}")
            elif request == "predict":  # predict theorem
                nodes, edges, structures, goals = info
                reply_queue = reply_queues[process_id]
                with torch.no_grad():
                    predicted_theorems = model(nodes=nodes.to(device), edges=edges.to(device),
                                               structures=structures.to(device), goals=goals.to(device)).cpu()
                reply_queue.put((problem_id, predicted_theorems))


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")

    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")
    parser.add_argument("--solve_again", type=lambda x: x == "True", default=True,
                        help="Solve problem which not in log['solved'].")

    parser.add_argument("--use_pretrain", type=lambda x: x == "True", default=True,
                        help="Use pretrain.")
    parser.add_argument("--use_structural_encoding", type=lambda x: x == "True", default=True,
                        help="Use structural encoding.")
    parser.add_argument("--use_hypertree", type=lambda x: x == "True", default=True,
                        help="Use hypertree.")

    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size when calculate acc.")
    parser.add_argument("--greedy_beam", type=lambda x: x == "True", default=False,
                        help="Use greedy beam search.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout for one problem.")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    Run in background:
    nohup <python test.py> >/dev/null 2>&1 &
    Kill subprocess:
    python utils.py --func kill --py_filename pac.py
    
    Run pac:
    python pac.py
    python pac.py --use_pretrain False
    python pac.py --use_structural_encoding False
    python pac.py --use_hypertree False
    
    python pac.py --beam_size 3
    python pac.py --use_pretrain False --beam_size 3
    python pac.py --use_structural_encoding False --beam_size 3
    python pac.py --use_hypertree False --beam_size 3
    
    python pac.py --beam_size 1
    python pac.py --use_pretrain False --beam_size 1
    python pac.py --use_structural_encoding False --beam_size 1
    python pac.py --use_hypertree False --beam_size 1
    
    Best result:
    python pac.py --timeout 600
    python pac.py --greedy_beam True --timeout 600
    """
    args = get_args()
    main(use_pretrain=args.use_pretrain,
         use_structural_encoding=args.use_structural_encoding, use_hypertree=args.use_hypertree,
         beam_size=args.beam_size, greedy_beam=args.greedy_beam, timeout=args.timeout,
         device=args.device, solve_again=args.solve_again)
