from formalgeo.tools import get_meta_hypertree, load_json, safe_save_json
from formalgeo.solver import Interactor
from formalgeo.parse import parse_one_theorem
from formalgeo.data import DatasetLoader
from utils import save_pickle
from utils import Configuration as config
from utils import symbol_words, nodes_words, nodes_words, theorem_words
import os
import re
import random
from func_timeout import func_timeout, FunctionTimedOut
import warnings
from multiprocessing import Process, Queue


# def make_onehot(training_data):
#     predictor_train = []  # list of (predicate, words, path, goal_predicate, goal_words, theorems)
#     path_pretrain = []
#
#     for nodes, path, goal, theorems in train_data:
#         predicate = [nodes_words.index(node[0]) for node in nodes]
#         words = [[nodes_words.index(w) for w in node[1:]] for node in nodes]
#         path = [[path_words.index(p) for p in path_item] for path_item in path]
#
#         goal_predicate = nodes_words.index(goal[0])
#         goal_words = [nodes_words.index(w) for w in goal[1:]]
#
#         theorems = [theorem_words.index(t) for t in theorems]
#
#         predictor_train.append((predicate, words, path, goal_predicate, goal_words, theorems))
#
#         for p in path:
#             if p not in path_pretrain:
#                 path_pretrain.append(p)
#     nodes_pretrain = predictor_train[-1][1]
#
#     return nodes_pretrain, path_pretrain, predictor_train
#
#
# def show_onehot(training_data):
#     """Show train data."""
#     if onehot:
#         nodes_pretrain, path_pretrain, predictor_train = train_data
#         print("nodes_pretrain:")
#         for data in nodes_pretrain:
#             print(data)
#         print("path_pretrain:")
#         for data in path_pretrain:
#             print(data)
#         print()
#         for predicate, words, path, goal_predicate, goal_words, theorems in predictor_train:
#             print("predictor_train - predicate:")
#             print(predicate)
#             print("predictor_train - words:")
#             for data in words:
#                 print(data)
#             print("predictor_train - path:")
#             for data in path:
#                 print(data)
#             print("predictor_train - goal:")
#             print("{}, {}".format(goal_predicate, goal_words))
#             print("predictor_train - theorems:")
#             print(theorems)
#             print()
#     else:
#         for nodes, path, goal, selected_theorems in train_data:
#             print("nodes:")
#             for data in nodes:
#                 print(data)
#             print("path:")
#             for data in path:
#                 print(data)
#             print("goal:")
#             print(goal)
#             print("theorems:")
#             print(selected_theorems)
#             print()


def get_hypertree(problem):
    """
    Build hypertree and return.
    :param problem: instance of <formalgeo.problem.Problem>.
    :return nodes: n*1, List of hyper nodes.
    :return path: n*1, Path from hyper node i to other nodes.
    """
    nodes, edges, _, tree = get_meta_hypertree(problem)

    for edge_id in edges:
        if "(" in edges[edge_id]:
            t_name, para = edges[edge_id].split("(")
            edges[edge_id] = "{}_{}".format(t_name, para[0])
    all_nodes = list(nodes.keys())
    path = [[["none"] for _ in all_nodes] for _ in all_nodes]
    for node_id in all_nodes:
        path[all_nodes.index(node_id)][all_nodes.index(node_id)] = ["self"]

    for premise, theorem in tree:  # init path
        conclusion = tree[(premise, theorem)]
        for head_node_id in premise:
            for tail_node_id in conclusion:
                path[all_nodes.index(head_node_id)][all_nodes.index(tail_node_id)] = [edges[theorem]]

    update = True
    while update:  # gen path
        update = False
        for i in range(len(path)):
            for j in range(len(path)):
                if path[i][j] == ["self"] or path[i][j] == ["none"]:
                    continue
                for k in range(len(path)):
                    if path[i][k] == ["none"] and path[j][k] != ["self"] and path[j][k] != ["none"]:
                        path[i][k] = path[i][j] + path[j][k]
                        update = True

    return list(nodes.values()), path


def tokenize_cdl(cdl):
    """
    Tokenize one cdl.
    >> tokenize_cdl('CongruentBetweenTriangle(RST,XYZ)')
    ['CongruentBetweenTriangle', 'R', 'S', 'T', ',', 'X', 'Y', 'Z']
    >> tokenize_cdl('Equation(ll_tr-x-21)')
    ['Equation', 'll_', 't', 'r', '-', 'x', '-', 'nums']
    """
    cdl = cdl[0:len(cdl) - 1].split("(", maxsplit=1)
    tokenized = [cdl[0]]
    if cdl[0] != "Equation":
        tokenized += list(cdl[1])
    else:
        for matched in re.findall(r"sin\(pi\*ma_\w+/180\)", cdl[1]):  # adjust trigonometric
            cdl[1] = cdl[1].replace(matched, "sin({})".format(matched[7:13]))
        for matched in re.findall(r"cos\(pi\*ma_\w+/180\)", cdl[1]):
            cdl[1] = cdl[1].replace(matched, "cos({})".format(matched[7:13]))
        for matched in re.findall(r"tan\(pi\*ma_\w+/180\)", cdl[1]):
            cdl[1] = cdl[1].replace(matched, "tan({})".format(matched[7:13]))

        for matched in re.findall(r"\d+\.*\d*", cdl[1]):  # replace real number with 'nums'
            cdl[1] = cdl[1].replace(matched, "nums", 1)

        while len(cdl[1]) > 0:  # tokenize
            length = len(cdl[1])
            for c in symbol_words:
                if cdl[1].startswith(c):
                    tokenized.append(cdl[1][0:len(c)])
                    cdl[1] = cdl[1][len(c):len(cdl[1])]
            if length == len(cdl[1]):
                tokenized.append(cdl[1][0])
                cdl[1] = cdl[1][1:len(cdl[1])]

    return tokenized


def serialize_edge(edge):
    """
    Serialize one edge and add delimiter.
    >> serialize_edge(['self', ['a', 'b'], 'none', ['c']])
    >> ['self', '<and>', 'a', 'b', '<and>', 'none', '<and>', 'c']
    """
    serialized = [p for p in edge[0]]

    for j in range(1, len(edge)):
        serialized.append("<and>")
        serialized += edge[j]

    serialized = [item for item in serialized if item != "none"]

    return serialized


def tokenize_theorem(theorem):
    """
    Tokenize one theorem.
    >> tokenize_theorem('congruent_triangle_property_angle_equal(1,RST,XYZ)')
    >> 'congruent_triangle_property_angle_equal_1'
    """
    t_name, t_branch, _ = parse_one_theorem(theorem)
    return "{}_{}".format(t_name, t_branch)


def show_training_data(training_data):
    for i in range(len(training_data)):
        print("nodes (step {}):".format(i + 1))
        for node in training_data[i][0]:
            print(node)
        print("edges (step {}):".format(i + 1))
        for node in training_data[i][1]:
            print(node)
        print("goal (step {}):".format(i + 1))
        print(training_data[i][2])
        print("theorems (step {}):".format(i + 1))
        print(training_data[i][3])
        print()


def generate(dl, problem_ids, reply_queue):
    problem_split = dl.get_problem_split()["split"]  # official partition
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)

    for problem_id in problem_ids:
        msg = []
        if problem_id in problem_split["train"]:
            msg.append("train")
        elif problem_id in problem_split["val"]:
            msg.append("val")
        else:
            msg.append("test")
        msg.append("unsolved")
        problem_CDL = dl.get_problem(problem_id)

        if len(problem_CDL["theorem_seqs"]) == 0:
            msg.append("error")
            msg.append("NO theorem seqs.")
            reply_queue.put((os.getpid(), problem_id, msg))
            continue

        try:
            training_data, solved = func_timeout(timeout=50, func=generate_one, args=(solver, problem_CDL))
        except FunctionTimedOut:
            msg.append("timeout")
            training_data, solved = func_timeout(timeout=50, func=generate_one, args=(solver, problem_CDL))
        except Exception as e:
            msg.append("error")
            msg.append(repr(e))
            reply_queue.put((os.getpid(), problem_id, msg))
            continue

        save_pickle(
            training_data,
            os.path.join(config.path_data, "training_data/{}/raw/{}.pk".format(msg[0], problem_id))
        )

        if solved:
            msg[1] = "solved"
        reply_queue.put((os.getpid(), problem_id, msg))


def generate_one(solver, problem_CDL, use_theorem_para=False):
    solver.load_problem(problem_CDL)
    training_data = []
    if solver.problem.goal.type == "algebra":
        goal = solver.problem.goal.item - solver.problem.goal.answer
        goal = "Equation" + "(" + str(goal).replace(" ", "") + ")"
    else:
        goal = problem_CDL["goal_cdl"].split("(", 1)[1]
        goal = goal[0:len(goal) - 1]
    goal = tokenize_cdl(goal)

    theorem_stack = [theorem for theorem in problem_CDL["theorem_seqs_dag"]["START"]]
    while len(theorem_stack) > 0:
        nodes, edges = get_hypertree(solver.problem)
        nodes = [tokenize_cdl(node) for node in nodes]
        edges = [serialize_edge(edge) for edge in edges]
        theorems = [tokenize_theorem(theorem) for theorem in theorem_stack]
        training_data.append([nodes, edges, goal, theorems])  # add one step

        theorem = random.choice(theorem_stack)  # apply theorem and update state
        theorem_stack.remove(theorem)
        if theorem in problem_CDL["theorem_seqs_dag"]:
            theorem_stack += problem_CDL["theorem_seqs_dag"][theorem]
        t_name, t_branch, t_para = parse_one_theorem(theorem)
        if use_theorem_para:
            solver.apply_theorem(t_name, t_branch, t_para)
        else:
            solver.apply_theorem(t_name, t_branch, None)
        solver.problem.check_goal()

    return training_data, solver.problem.goal.solved


def main(problem_id=None):
    warnings.filterwarnings("ignore")
    dl = DatasetLoader(config.dataset_name, config.path_datasets)

    if problem_id is not None:
        solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)
        problem_CDL = dl.get_problem(problem_id)
        try:
            training_data, solved = generate_one(solver, problem_CDL)
        except FunctionTimedOut:
            training_data, solved = generate_one(solver, problem_CDL, use_theorem_para=True)
        show_training_data(training_data)
        return

    log = {}  # log {pid: ['train', 'solved', 'timeout', 'error']}
    log_filename = os.path.join(config.path_data, "log/gen_training_data_log.json")
    if os.path.exists(log_filename):
        log = load_json(log_filename)

    problem_ids = []
    for problem_id in range(1, dl.info["problem_number"] + 1):
        if str(problem_id) not in log:
            problem_ids.append(problem_id)

    reply_queue = Queue()
    started_process = 0
    problems_per_process = int(len(problem_ids) / config.process_count)
    while config.process_count - started_process > 0:
        start_index = started_process * problems_per_process
        end_index = len(problem_ids) if config.process_count - started_process == 1 \
            else (started_process + 1) * problems_per_process
        process = Process(target=generate, args=(dl, problem_ids[start_index:end_index], reply_queue))
        process.start()
        started_process += 1

    while True:
        process_id, problem_id, msg = reply_queue.get()
        log[problem_id] = msg
        safe_save_json(log, log_filename)
        print("{}: {}".format(problem_id, msg))


if __name__ == '__main__':
    random.seed(config.random_seed)
    main()
