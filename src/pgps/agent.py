from pgps.utils import Configuration as config
from pgps.model import make_predictor_model
from pgps.symbolic_system import tokenize_cdl, serialize_edge, get_hypertree
from pgps.utils import nodes_words, edges_words, theorem_words
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


def make_onehot(state_info):
    onehot_data = []
    for nodes, edges, edges_structural, goal in state_info:
        if len(nodes) > config.max_len:  # random truncation
            for idx in sorted(random.sample(range(1, len(nodes)), len(nodes) - config.max_len))[::-1]:
                nodes.pop(idx)
                edges.pop(idx)
                edges_structural.pop(idx)

        one_hot_nodes = []
        for node in nodes:
            if len(node) > config.max_len_nodes - 1:  # truncation
                node = node[:config.max_len_nodes - 1]
            node = [nodes_words.index(n) for n in node]
            node.insert(0, 1)  # <start>
            node.extend([0] * (config.max_len_nodes - len(node)))  # padding
            one_hot_nodes.append(node)

        one_hot_edges = []
        for edge in edges:
            if len(edge) > config.max_len_edges - 1:  # truncation
                edge = edge[:config.max_len_edges - 1]
            edge = [edges_words.index(n) for n in edge]
            edge.insert(0, 1)  # <start>
            edge.extend([0] * (config.max_len_edges - len(edge)))  # padding
            one_hot_edges.append(edge)

        for i in range(len(edges_structural)):
            if len(edges_structural[i]) > config.max_len_edges - 1:  # truncation
                edges_structural[i] = edges_structural[i][:config.max_len_edges - 1]
            edges_structural[i].insert(0, 0)  # position 0 in edges is <start>, so padding
            edges_structural[i].extend([0] * (config.max_len_edges - len(edges_structural[i])))  # padding

        if len(goal) > config.max_len_nodes - 1:  # truncation
            goal = goal[:config.max_len_nodes - 1]
        one_hot_goal = [nodes_words.index(g) for g in goal]
        one_hot_goal.insert(0, 1)  # <start>
        one_hot_goal.extend([0] * (config.max_len_nodes - len(one_hot_goal)))  # padding

        if len(one_hot_nodes) < config.max_len:
            insert_count = config.max_len - len(one_hot_nodes)
            one_hot_nodes.extend([[0] * config.max_len_nodes] * insert_count)
            one_hot_edges.extend([[0] * config.max_len_edges] * insert_count)
            edges_structural.extend([[0] * config.max_len_edges] * insert_count)

        onehot_data.append([one_hot_nodes, one_hot_edges, edges_structural, one_hot_goal])

    nodes = torch.tensor([item[0] for item in onehot_data])
    edges = torch.tensor([item[1] for item in onehot_data])
    edges_structural = torch.tensor([item[2] for item in onehot_data])
    goal = torch.tensor([item[3] for item in onehot_data])

    return nodes, edges, edges_structural, goal


def get_state(beam_stacks, goal):
    goal = tokenize_cdl(goal)

    state_info = []
    for problem, _ in beam_stacks:
        nodes, edges = get_hypertree(problem)
        nodes = [tokenize_cdl(node) for node in nodes]
        serialized_edges = []
        edges_structural = []
        for edge in edges:
            serialized_edge, token_index = serialize_edge(edge)
            serialized_edges.append(serialized_edge)
            edges_structural.append(token_index)
        state_info.append([nodes, serialized_edges, edges_structural, goal])

    return make_onehot(state_info)


def apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam):
    predicted_theorems = F.softmax(predicted_theorems, dim=1)    # convert to a probability distribution
    for i in range(len(beam_stacks)):    # consider previous prob
        predicted_theorems[i] = predicted_theorems[i] * beam_stacks[i][1]
    predicted_theorems = predicted_theorems.flatten()    # flatten for sorting

    sorted_theorems = [(idx, prob)
                       for idx, prob in sorted(enumerate(predicted_theorems), key=lambda x: x[1], reverse=True)]

    new_beam_stacks = []
    prob_sum = 0
    if greedy_beam:
        i = 0
        while len(new_beam_stacks) < beam_size and i < len(sorted_theorems):
            idx, prob = sorted_theorems[i]
            problem = Problem()
            problem.load_problem_by_copy(beam_stacks[int(idx / len(theorem_words))][0])
            t_name, t_branch = theorem_words[idx % len(theorem_words)].rsplit('_', 1)
            solver.problem = problem
            if solver.apply_theorem(t_name=t_name, t_branch=t_branch):  # if update
                new_beam_stacks.append([solver.problem, prob])
                prob_sum += prob
            i += 1
    else:
        for idx, prob in sorted_theorems[:beam_size]:
            problem = Problem()
            problem.load_problem_by_copy(beam_stacks[int(idx / len(theorem_words))][0])
            t_name, t_branch = theorem_words[idx % len(theorem_words)].rsplit('_', 1)
            solver.problem = problem
            if solver.apply_theorem(t_name=t_name, t_branch=t_branch):  # if update
                new_beam_stacks.append([solver.problem, prob])
                prob_sum += prob

    for i in range(len(new_beam_stacks)):    # probability normalization
        new_beam_stacks[i][1] = new_beam_stacks[i][1] / prob_sum

    return new_beam_stacks


def solve_one(problem_id, request_queue, reply_queue, beam_size, greedy_beam, step_size):
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)
    problem_CDL = dl.get_problem(problem_id)
    solver.load_problem(problem_CDL)
    if solver.problem.goal.type == "algebra":
        goal = solver.problem.goal.item - solver.problem.goal.answer
        goal = "Equation" + "(" + str(goal).replace(" ", "") + ")"
    else:
        goal = problem_CDL["goal_cdl"].split("(", 1)[1]
        goal = goal[0:len(goal) - 1]

    beam_stacks = [[solver.problem, 1]]  # max_count(beam_stack) = beam_size
    while len(beam_stacks) > 0:
        for problem, _ in beam_stacks:
            solver.problem = problem
            solver.problem.check_goal()
            if solver.problem.goal.solved:
                _, seqs = get_used_pid_and_theorem(solver.problem)
                seqs = [inverse_parse_one_theorem(s, solver.parsed_theorem_GDL) for s in seqs]
                return "solved", seqs

        state_info = get_state(beam_stacks, goal)
        request_queue.put((os.getpid(), problem_id, "predict", state_info))

        reply_problem_id, predicted_theorems = reply_queue.get()
        while reply_problem_id != problem_id:
            reply_problem_id, predicted_theorems = reply_queue.get()

        beam_stacks = apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam)

        step_size["epoch"] += 1

    return "unsolved", "Out of stacks."


def solve(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout):
    warnings.filterwarnings("ignore")
    if not task_queue.empty():
        problem_id = task_queue.get()
        timing = time.time()
        step_size = {"epoch": 0}

        try:
            solved, msg = func_timeout(
                timeout, solve_one, args=(problem_id, request_queue, reply_queue, beam_size, greedy_beam, step_size))
            info = (solved, msg, time.time() - timing, step_size["epoch"])
        except FunctionTimedOut:
            info = ("timeout", f"FunctionTimedOut({timeout})", time.time() - timing, step_size["epoch"])
        except BaseException as e:
            info = ("error", repr(e), time.time() - timing, step_size["epoch"])

        request_queue.put((os.getpid(), problem_id, "write", info))


def start_process(task_queue, request_queue, reply_queues, beam_size, greedy_beam, timeout, max_process):
    """clean process and start new process."""
    removed = []
    for process_id in reply_queues:
        if time.time() - reply_queues[process_id][1] > timeout * 1.2:
            removed.append(process_id)
    for process_id in removed:
        del reply_queues[process_id]

    while not task_queue.empty() and max_process - len(reply_queues) > 0:
        reply_queue = Queue()
        process = Process(target=solve, args=(task_queue, request_queue, reply_queue, beam_size, greedy_beam, timeout))
        process.start()
        reply_queues[process.pid] = (reply_queue, time.time())


def main(model_name, device, beam_size, greedy_beam, use_hypertree, timeout, max_process):
    model_path = os.path.normpath(os.path.join(config.path_data, f"trained_model/{model_name}"))
    log_path = os.path.normpath(os.path.join(config.path_data, f"log/pac_log.json"))

    log = {"solved": {}, "unsolved": {}, "timeout": {}, "error": {}}
    if os.path.exists(log_path):
        log = load_json(log_path)

    task_queue = Queue()
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    for problem_id in dl.get_problem_split()["split"]["test"]:
        if str(problem_id) in log["solved"]:
            continue
        if str(problem_id) in log["unsolved"]:
            continue
        if str(problem_id) in log["timeout"]:
            continue
        if str(problem_id) in log["error"]:
            continue
        task_queue.put(problem_id)

    request_queue = Queue()
    reply_queues = {}  # map pid to process queue

    device = torch.device(device)
    model = make_predictor_model(torch.load(model_path, map_location=torch.device("cpu"))["model"]).to(device)

    while True:
        start_process(task_queue, request_queue, reply_queues, beam_size, greedy_beam, timeout, max_process)
        process_id, problem_id, request, info = request_queue.get()
        if request == "write":  # write log
            del reply_queues[process_id]
            result, msg, timing, step_size = info
            log[result][str(problem_id)] = {"msg": msg, "timing": timing, "step_size": step_size}
            safe_save_json(log, log_path)
            print(f"{process_id}\t{problem_id}\t{result}\t{timing}\t{step_size}")
        else:  # predict theorem
            nodes, edges, edges_structural, goal = info
            reply_queue = reply_queues[process_id][0]
            nodes = nodes.to(device)
            goal = goal.to(device)
            with torch.no_grad():
                if use_hypertree:
                    edges = edges.to(device)
                    edges_structural = edges_structural.to(device)
                    predicted_theorems = model(nodes, edges, edges_structural, goal).cpu()
                else:
                    predicted_theorems = model(nodes, None, None, goal).cpu()
            reply_queue.put((problem_id, predicted_theorems))


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use PGPS!")

    parser.add_argument("--model_name", type=str, required=True,
                        help="The tested model name.")
    parser.add_argument("--device", type=str, required=False, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")
    parser.add_argument("--beam_size", type=int, required=False, default=5,
                        help="Beam size when calculate acc.")
    parser.add_argument("--greedy_beam", type=bool, required=False, default=False,
                        help="Weather use greedy beam search or not.")
    parser.add_argument("--use_hypertree", type=bool, required=False, default=False,
                        help="Weather use hypertree information or not.")
    parser.add_argument("--timeout", type=int, required=False, default=30,
                        help="Timeout for one problem.")
    parser.add_argument("--max_process", type=int, required=False, default=config.process_count,
                        help="Number of multi process.")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    run pac:
    python agent.py --model_name predictor_model_no_pretrain.pth --use_hypertree true
    python agent.py --model_name predictor_model_no_pretrain.pth --use_hypertree true --greedy_beam true
    
    kill subprocess:
    python utils.py --func kill --py_filename agent.py
    """
    args = get_args()
    main(args.model_name, args.device, args.beam_size, args.greedy_beam, args.use_hypertree, args.timeout, args.max_process)
