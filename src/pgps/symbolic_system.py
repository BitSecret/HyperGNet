from formalgeo.tools import get_meta_hypertree, load_json, safe_save_json
from formalgeo.solver import Interactor
from formalgeo.parse import parse_one_theorem
from formalgeo.data import DatasetLoader
from pgps.utils import load_pickle, save_pickle
from pgps.utils import Configuration as config
from pgps.utils import symbol_words, nodes_words, edges_words, theorem_words
import os
import re
import random
from func_timeout import func_timeout, FunctionTimedOut
import warnings
from multiprocessing import Process, Queue
import psutil
import matplotlib.pyplot as plt

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

"""--------------data generation--------------"""


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
    >> serialize_edge([['self'], ['a', 'b'], ['none'], ['c']])
    >> (['self', 'a', 'b', 'c'], [0, 1, 1, 3])
    """
    serialized = []
    token_index = []

    for j in range(len(edge)):
        if edge[j] == ["none"]:
            continue
        for token in edge[j]:
            serialized.append(token)
            token_index.append(j)

    return [serialized, token_index]


def tokenize_theorem(theorem):
    """
    Tokenize one theorem.
    >> tokenize_theorem('congruent_triangle_property_angle_equal(1,RST,XYZ)')
    >> 'congruent_triangle_property_angle_equal_1'
    """
    t_name, t_branch, _ = parse_one_theorem(theorem)
    return "{}_{}".format(t_name, t_branch)


def generate(dl, task_queue, reply_queue):
    warnings.filterwarnings("ignore")
    problem_split = dl.get_problem_split()["split"]  # official partition
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)

    while not task_queue.empty():
        problem_id = task_queue.get()
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
            training_data, solved = func_timeout(timeout=50, func=generate_one, args=(solver, problem_CDL, True))
        except BaseException as e:
            msg.append("error")
            msg.append(repr(e))
            reply_queue.put((os.getpid(), problem_id, msg))
            continue

        save_pickle(
            training_data,
            os.path.normpath(os.path.join(config.path_data, "training_data/{}/raw/{}.pk".format(msg[0], problem_id)))
        )

        if solved:
            msg[1] = "solved"
        reply_queue.put((os.getpid(), problem_id, msg))


def start_process(dl, task_queue, reply_queue, process_ids):
    """Remove non-existent pid and start new process"""
    for i in range(len(process_ids))[::-1]:
        if process_ids[i] not in psutil.pids():
            process_ids.pop(i)

    while not task_queue.empty() and config.process_count - len(process_ids) > 0:
        process = Process(target=generate, args=(dl, task_queue, reply_queue))
        process.start()
        process_ids.append(process.pid)


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
    log_filename = os.path.normpath(os.path.join(config.path_data, "log/gen_training_data_log.json"))
    if os.path.exists(log_filename):
        log = load_json(log_filename)

    task_queue = Queue()
    for problem_id in range(1, dl.info["problem_number"] + 1):
        if str(problem_id) not in log:
            task_queue.put(problem_id)

    reply_queue = Queue()
    process_ids = []
    while True:
        start_process(dl, task_queue, reply_queue, process_ids)
        process_id, problem_id, msg = reply_queue.get()
        log[problem_id] = msg
        safe_save_json(log, log_filename)
        print("{}: {}".format(problem_id, msg))


"""--------------data preprocess--------------"""


def show_training_data(pid_or_training_data):
    if isinstance(pid_or_training_data, int):
        pid = pid_or_training_data
        log_filename = os.path.normpath(os.path.join(config.path_data, "log/gen_training_data_log.json"))
        log = load_json(log_filename)
        if str(pid) not in log:
            print("{} not generated.".format(pid))
            return
        data_filename = os.path.normpath(
            os.path.join(config.path_data, "training_data/{}/raw/{}.pk".format(log[pid][0], pid)))
        if not os.path.exists(data_filename):
            print("{} {} not generated.".format(log[pid][0], pid))
            return
        training_data = load_pickle(data_filename)
    else:
        training_data = pid_or_training_data

    for i in range(len(training_data)):
        print("nodes (step {}):".format(i + 1))
        for node in training_data[i][0]:
            print(node)
        print("edges (step {}):".format(i + 1))
        for edge, token_index in training_data[i][1]:
            print("{}\t{}".format(edge, token_index))
        print("goal (step {}):".format(i + 1))
        print(training_data[i][2])
        print("theorems (step {}):".format(i + 1))
        print(training_data[i][3])
        print()


def check_log():
    log_filename = os.path.normpath(os.path.join(config.path_data, "log/gen_training_data_log.json"))
    log = load_json(log_filename)
    log = {int(k): log[k] for k in log}
    log = {k: log[k] for k in sorted(log)}
    safe_save_json(log, log_filename)  # save sorted log

    timeout = []
    unsolved = []
    error = []
    unhandled = []
    for pid in range(1, 6982):
        if pid not in log:
            unhandled.append(pid)
        elif "timeout" in log[pid]:
            timeout.append(pid)
        elif "error" in log[pid]:
            error.append(pid)
        elif "unsolved" in log[pid]:
            unsolved.append(pid)

    print("timeout ({}):".format(len(timeout)))
    for pid in timeout:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("unsolved ({}):".format(len(unsolved)))
    for pid in unsolved:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("error ({}):".format(len(error)))
    for pid in error:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("unhandled ({}):".format(len(unhandled)))
    print(unhandled)


def check_len():
    nodes_len_file = os.path.normpath(os.path.join(config.path_data, "log/words_len/nodes_len.pk"))
    edges_len_file = os.path.normpath(os.path.join(config.path_data, "log/words_len/edges_len.pk"))
    se_len_file = os.path.normpath(os.path.join(config.path_data, "log/words_len/se_len.pk"))

    if not os.path.exists(nodes_len_file):
        log_filename = os.path.normpath(os.path.join(config.path_data, "log/gen_training_data_log.json"))
        log = load_json(log_filename)
        nodes_len = {}
        edges_len = {}
        se_len = {}

        for pid in log:
            filename = os.path.normpath(
                os.path.join(config.path_data, "training_data/{}/raw/{}.pk".format(log[pid][0], pid)))
            if not os.path.exists(filename):
                continue
            training_data = load_pickle(filename)
            for nodes, edges, goal, theorems in training_data:
                for node in nodes:
                    if len(node) not in nodes_len:
                        nodes_len[len(node)] = 1
                    else:
                        nodes_len[len(node)] += 1
                for edge, index_count in edges:
                    if len(edge) not in edges_len:
                        edges_len[len(edge)] = 1
                    else:
                        edges_len[len(edge)] += 1
                    for i in index_count:
                        if i not in se_len:
                            se_len[i] = 1
                        else:
                            se_len[i] += 1
                if len(goal) not in nodes_len:
                    nodes_len[len(goal)] = 1
                else:
                    nodes_len[len(goal)] += 1
            print("{} ok.".format(pid))

        nodes_len = [(k, nodes_len[k]) for k in sorted(nodes_len)]
        edges_len = [(k, edges_len[k]) for k in sorted(edges_len)]
        se_len = [(k, se_len[k]) for k in sorted(se_len)]
        save_pickle(edges_len, nodes_len_file)
        save_pickle(nodes_len, edges_len_file)
        save_pickle(se_len, se_len_file)
    else:
        nodes_len = load_pickle(nodes_len_file)
        edges_len = load_pickle(edges_len_file)
        se_len = load_pickle(se_len_file)

    draw_pic(nodes_len, "nodes", (32, 8))
    draw_pic(edges_len, "edges", (64, 8))
    draw_pic(se_len, "se", (128, 8))


def draw_pic(data, title, fig_size):
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    y_integral = [y[0]]
    for i in range(1, len(y)):
        y_integral.append(y_integral[i - 1] + y[i])
    y_integral = [item / y_integral[-1] for item in y_integral]
    print("{}:".format(title))
    for i in range(len(x) - 1):
        if y_integral[i] <= 0.9:
            if y_integral[i + 1] >= 0.9:
                print("0.9: {}".format(x[i + 1]))
        elif y_integral[i] <= 0.95:
            if y_integral[i + 1] >= 0.95:
                print("0.95: {}".format(x[i + 1]))
        elif y_integral[i] <= 0.99:
            if y_integral[i + 1] >= 0.99:
                print("0.99: {}".format(x[i + 1]))
    print("1.0: {}".format(x[-1]))
    print()

    plt.figure(figsize=fig_size)
    plt.plot(x, y, marker='o')
    plt.title('{} (Density)'.format(title))
    plt.savefig(os.path.normpath(os.path.join(config.path_data, "log/words_len/{}_density.png".format(title))))
    plt.show()

    plt.figure(figsize=fig_size)
    for i in range(len(x) - 1):  # draw line
        plt.plot(x[i:i + 2], y_integral[i:i + 2], 'b-')
    for i in range(len(x)):  # draw point
        if y_integral[i] < 0.9:
            plt.plot(x[i], y_integral[i], 'o', color="green")
        elif y_integral[i] < 0.95:
            plt.plot(x[i], y_integral[i], 'o', color="yellow")
        elif y_integral[i] < 0.99:
            plt.plot(x[i], y_integral[i], 'o', color="pink")
        else:
            plt.plot(x[i], y_integral[i], 'o', color="red")
    plt.title('{} (Integral)'.format(title))
    plt.savefig(os.path.normpath(os.path.join(config.path_data, "log/words_len/{}_integral.png".format(title))))
    plt.show()


if __name__ == '__main__':
    random.seed(config.random_seed)
    # main()
    # check_log()
    check_len()
