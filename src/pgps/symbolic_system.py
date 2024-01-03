from formalgeo.tools import get_meta_hypertree, load_json, safe_save_json
from formalgeo.solver import Interactor
from formalgeo.parse import parse_one_theorem
from formalgeo.data import DatasetLoader
from utils import save_pickle
from utils import Configuration as config
from utils import symbol_words, nodes_words, path_words, theorem_words
import os
import re
import random
from func_timeout import func_timeout, FunctionTimedOut
import warnings


def tokenize_cdl(cdl):
    """
    Build hypertree and return.
    :param cdl: CDL, such as 'CongruentBetweenTriangle(RST,XYZ)' or 'Equation(ll_tr-x-21)'.
    :return tokenized: tokenized CDL, such as ['CongruentBetweenTriangle', 'R', 'S', 'T', ',', 'X', 'Y', 'Z'].
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


def tokenize(train_data):
    """Tokenize train_data."""
    tokenized_train_data = []
    for nodes, path, goal, theorems in train_data:
        tokenized_nodes = []
        for node in nodes:
            tokenized_nodes.append(tokenize_cdl(node))

        tokenized_path = []
        for i in range(len(path)):
            node_i_paths = []
            for p in path[i][0]:
                node_i_paths.append(p)

            for j in range(1, len(path[i])):
                node_i_paths.append("<and>")
                for p in path[i][j]:
                    node_i_paths.append(p)

            tokenized_path.append(node_i_paths)

        tokenized_goal = tokenize_cdl(goal)

        tokenized_theorems = ["{}_{}".format(t_name, t_branch) for t_name, t_branch in theorems]

        tokenized_train_data.append((tokenized_nodes, tokenized_path, tokenized_goal, tokenized_theorems))

    return tokenized_train_data


def make_onehot(train_data):
    predictor_train = []  # list of (predicate, words, path, goal_predicate, goal_words, theorems)
    path_pretrain = []

    for nodes, path, goal, theorems in train_data:
        predicate = [nodes_words.index(node[0]) for node in nodes]
        words = [[nodes_words.index(w) for w in node[1:]] for node in nodes]
        path = [[path_words.index(p) for p in path_item] for path_item in path]

        goal_predicate = nodes_words.index(goal[0])
        goal_words = [nodes_words.index(w) for w in goal[1:]]

        theorems = [theorem_words.index(t) for t in theorems]

        predictor_train.append((predicate, words, path, goal_predicate, goal_words, theorems))

        for p in path:
            if p not in path_pretrain:
                path_pretrain.append(p)
    nodes_pretrain = predictor_train[-1][1]

    return nodes_pretrain, path_pretrain, predictor_train


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


def show_train_data(train_data, onehot=False):
    """Show train data."""
    if onehot:
        nodes_pretrain, path_pretrain, predictor_train = train_data
        print("nodes_pretrain:")
        for data in nodes_pretrain:
            print(data)
        print("path_pretrain:")
        for data in path_pretrain:
            print(data)
        print()
        for predicate, words, path, goal_predicate, goal_words, theorems in predictor_train:
            print("predictor_train - predicate:")
            print(predicate)
            print("predictor_train - words:")
            for data in words:
                print(data)
            print("predictor_train - path:")
            for data in path:
                print(data)
            print("predictor_train - goal:")
            print("{}, {}".format(goal_predicate, goal_words))
            print("predictor_train - theorems:")
            print(theorems)
            print()
    else:
        for nodes, path, goal, selected_theorems in train_data:
            print("nodes:")
            for data in nodes:
                print(data)
            print("path:")
            for data in path:
                print(data)
            print("goal:")
            print(goal)
            print("theorems:")
            print(selected_theorems)
            print()


def get_one_problem(solver, problem_CDL, use_theorem_para=False):
    train_data = []
    solver.load_problem(problem_CDL)
    if solver.problem.goal.type == "algebra":
        goal = solver.problem.goal.item - solver.problem.goal.answer
        goal = "Equation" + "(" + str(goal).replace(" ", "") + ")"
    else:
        goal = problem_CDL["goal_cdl"].split("(", 1)[1]
        goal = goal[0:len(goal) - 1]

    theorem_stack = [theorem for theorem in problem_CDL["theorem_seqs_dag"]["START"]]
    while len(theorem_stack) > 0:
        nodes, path = get_hypertree(solver.problem)
        theorems = []
        for theorem in theorem_stack:
            t_name, t_branch, _ = parse_one_theorem(theorem)
            theorems.append((t_name, t_branch))
        train_data.append((nodes, path, goal, theorems))

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

    return train_data


def get_train_data():
    warnings.filterwarnings("ignore")
    dl = DatasetLoader(config.dataset_name, config.path_datasets)
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)
    problem_split = dl.get_problem_split()["split"]
    log = {
        "train": [], "val": [], "test": [], "error": []
    }
    log_filename = os.path.join(config.path_data, "log/gen_train_data_log.json")
    if os.path.exists(log_filename):
        log = load_json(log_filename)

    for split in problem_split:
        for pid in problem_split[split]:
            problem_CDL = dl.get_problem(pid)
            if len(problem_CDL["theorem_seqs"]) == 0:
                continue
            if pid in log[split]:
                continue

            # if pid != 161:
            #     continue

            print("{} {} start.".format(split, pid))

            try:
                train_data = func_timeout(30, get_one_problem, args=(solver, problem_CDL))
            except FunctionTimedOut:
                train_data = func_timeout(30, get_one_problem, args=(solver, problem_CDL, True))
            except Exception as e:
                log["error"].append([split, pid, repr(e)])
                safe_save_json(log, log_filename)
                print("{} {} error.".format(split, pid))
                continue

            # show_train_data(train_data)
            # tokenized_train_data = tokenize(train_data)
            # # show_train_data(tokenized_train_data)
            # nodes_pretrain, path_pretrain, predictor_train = make_onehot(tokenized_train_data)
            # # show_train_data((nodes_pretrain, path_pretrain, predictor_train), onehot=True)
            #
            # save_pickle(nodes_pretrain,
            #             os.path.join(config.path_data, "pretrain_data/nodes/{}/{}.pk".format(split, pid)))
            # save_pickle(path_pretrain,
            #             os.path.join(config.path_data, "pretrain_data/path/{}/{}.pk".format(split, pid)))
            # save_pickle(predictor_train,
            #             os.path.join(config.path_data, "train_data/{}/{}.pk".format(split, pid)))
            #
            # log[split].append(pid)
            # safe_save_json(log, log_filename)
            # print("{} {} ok.".format(split, pid))

            try:
                # show_train_data(train_data)
                tokenized_train_data = tokenize(train_data)
                # show_train_data(tokenized_train_data)
                nodes_pretrain, path_pretrain, predictor_train = make_onehot(tokenized_train_data)
                # show_train_data((nodes_pretrain, path_pretrain, predictor_train), onehot=True)

                save_pickle(nodes_pretrain,
                            os.path.join(config.path_data, "pretrain_data/nodes/{}/{}.pk".format(split, pid)))
                save_pickle(path_pretrain,
                            os.path.join(config.path_data, "pretrain_data/path/{}/{}.pk".format(split, pid)))
                save_pickle(predictor_train,
                            os.path.join(config.path_data, "train_data/{}/{}.pk".format(split, pid)))

                log[split].append(pid)
                safe_save_json(log, log_filename)
                print("{} {} ok.".format(split, pid))
            except Exception as e:
                log["error"].append([split, pid, repr(e)])
                safe_save_json(log, log_filename)
                print("{} {} error.".format(split, pid))


if __name__ == '__main__':
    random.seed(config.random_seed)
    get_train_data()
