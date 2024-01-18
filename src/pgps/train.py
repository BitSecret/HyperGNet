from formalgeo.tools import load_json, safe_save_json
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle
from pgps.model import make_predictor_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
from tqdm import tqdm
import argparse


class PGPSDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []
        for one_hot_nodes, one_hot_edges, edges_structural, one_hot_goal, theorems_index in raw_data:
            for node in one_hot_nodes:
                node.insert(0, 1)  # <start>
                node.extend([0] * (config.max_len_nodes - len(node)))  # padding

            for edge in one_hot_edges:
                edge.insert(0, 1)  # <start>
                edge.extend([0] * (config.max_len_edges - len(edge)))  # padding

            for item in edges_structural:
                item.insert(0, 0)  # position 0 in edges is <start>, so padding
                item.extend([0] * (config.max_len_edges - len(item)))  # padding

            one_hot_goal.insert(0, 1)  # <start>
            one_hot_goal.extend([0] * (config.max_len_nodes - len(one_hot_goal)))  # padding

            theorems = [0] * config.vocab_theorems
            for idx in theorems_index:
                theorems[idx] = 1

            if len(one_hot_nodes) < config.max_len:
                insert_count = config.max_len - len(one_hot_nodes)
                one_hot_nodes.extend([[0] * config.max_len_nodes] * insert_count)
                one_hot_edges.extend([[0] * config.max_len_edges] * insert_count)
                edges_structural.extend([[0] * config.max_len_edges] * insert_count)

            self.data.append((torch.tensor(one_hot_nodes),
                              torch.tensor(one_hot_edges),
                              torch.tensor(edges_structural),
                              torch.tensor(one_hot_goal),
                              torch.tensor(theorems)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3], self.data[idx][4]


def load_dataset(dataset_type):
    """
    Pretrain nodes model and edges model.
    :param dataset_type: 'train', 'val' or 'test'.
    """
    onehot_path = os.path.normpath(os.path.join(config.path_data, f"training_data/{dataset_type}/one-hot.pk"))
    dataset_path = os.path.normpath(os.path.join(config.path_data, f"training_data/{dataset_type}/dataset_pgps.pk"))
    if os.path.exists(dataset_path):
        dataset = load_pickle(dataset_path)
    else:
        dataset = PGPSDataset(load_pickle(onehot_path))
        save_pickle(dataset, dataset_path)
    return dataset


def train(device, nodes_model_name, edges_model_name, gs_model_name, freeze_pretrained, beam_size):
    """Train theorem prediction model."""
    log_path = os.path.normpath(os.path.join(config.path_data, "log/train_log.json"))
    loss_save_path = os.path.normpath(os.path.join(config.path_data, "log/train/{}_loss.pk"))
    text_save_path = os.path.normpath(os.path.join(config.path_data, "log/train/{}_eval.text"))
    pretrained_path = os.path.normpath(os.path.join(config.path_data, "trained_model/{}"))
    model_path = os.path.normpath(os.path.join(config.path_data, "trained_model/predictor_model.pth"))
    model_bk_path = os.path.normpath(os.path.join(config.path_data, "trained_model/predictor_model_bk.pth"))
    device = torch.device(device)

    print("Load data and make model (the first time may be slow)...")
    dataset_train = load_dataset("train")
    dataset_val = load_dataset("val")
    log = {
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "timing": 1}
        "eval": {},  # epoch: {"acc": 1, "timing": 1}
        "best_acc": 0,
        "best_epoch": 0,
    }
    if os.path.exists(log_path):
        log = load_json(log_path)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_eval = DataLoader(dataset=dataset_val, batch_size=config.batch_size * 2, shuffle=False)

    model_state = None
    optimizer_state = None
    nodes_model = None
    edges_model = None
    gs_model = None
    if log["next_epoch"] > 1:
        last_epoch_msg = torch.load(model_bk_path, map_location=torch.device("cpu"))
        model_state = last_epoch_msg["model"]
        optimizer_state = last_epoch_msg["optimizer"]
    else:
        nodes_model = torch.load(pretrained_path.format(nodes_model_name), map_location=torch.device("cpu"))["model"]
        edges_model = torch.load(pretrained_path.format(edges_model_name), map_location=torch.device("cpu"))["model"]
        gs_model = torch.load(pretrained_path.format(gs_model_name), map_location=torch.device("cpu"))["model"]
    model = make_predictor_model(model_state, nodes_model, edges_model, gs_model, freeze_pretrained).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Adam optimizer
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loss_func = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    for epoch in range(log["next_epoch"], config.epoch + 1):
        model.train()
        step_list = []
        loss_list = []
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (nodes, edges, edges_structural, goal, theorems_gt) in loop:
            nodes = nodes.to(device)
            edges = edges.to(device)
            edges_structural = edges_structural.to(device)
            goal = goal.to(device)
            theorems_gt = theorems_gt.to(device)
            theorems = model(nodes, edges, edges_structural, goal)  # model prediction results
            loss = loss_func(theorems.float(), theorems_gt.float())  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize param

            step_list.append((epoch - 1) * len(loop) + step)
            loss_list.append(float(loss))
            loop.set_description(f"Epoch [{epoch}/{config.epoch}] (Pretraining)")
            loop.set_postfix(loss=float(loss))

        save_pickle((step_list, loss_list), loss_save_path.format(epoch))
        log["train"][str(epoch)] = {"avg_loss": sum(loss_list) / len(loss_list), "timing": time.time() - timing}

        acc, timing = evaluate(model, data_loader_eval, device, beam_size, save_filename=text_save_path.format(epoch))
        log["eval"][str(epoch)] = {"beam_size": beam_size, "acc": acc, "timing": timing}
        if acc > log["best_acc"]:
            log["best_acc"] = acc
            log["best_epoch"] = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_bk_path)
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


def test(device, model_name, beam_size):
    """
    Test trained model.
    :param device: 'cpu' or 'cuda:gpu_id'.
    :param model_name: tested model name in data/trained_model.
    :param beam_size: beam size used for calculate acc.
    """
    print("Load data and make model (the first time may be slow)...")
    text_path = os.path.normpath(os.path.join(config.path_data, f"log/test/predictor_eval.text"))
    results_path = os.path.normpath(os.path.join(config.path_data, f"log/test/predictor_test_log.json"))
    device = torch.device(device)

    state_dict = torch.load(
        os.path.normpath(os.path.join(config.path_data, "trained_model", model_name)), map_location=torch.device("cpu")
    )["model"]
    model = make_predictor_model(state_dict).to(device)
    data_loader = DataLoader(dataset=load_dataset("test"), batch_size=config.batch_size * 2, shuffle=False)

    acc, timing = evaluate(model, data_loader, device, beam_size, text_path)

    safe_save_json({"acc": acc, "timing": timing}, results_path)
    print(f"Acc: {acc}. Details saved in {text_path}")


def evaluate(model, data_loader, device, beam_size, save_filename=None):
    """
    Evaluate model and save evaluation results.
    :param model: tested model, instance of <Predictor>.
    :param data_loader: test datasets data loader.
    :param device: device that model run, torch.device().
    :param beam_size: beam size used for calculate acc.
    :param save_filename: file that save evaluation results.
    :return acc: Weighted Levenshtein ratio.
    :return timing: timing.
    """
    timing = time.time()
    model.eval()
    acc_count = 0  # acc with beam_size
    num_count = 0  # all data count
    results = []  # saved text
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)  # evaluating loop
    with torch.no_grad():
        for step, (nodes, edges, edges_structural, goal, theorems_gt) in loop:
            nodes = nodes.to(device)
            edges = edges.to(device)
            edges_structural = edges_structural.to(device)
            goal = goal.to(device)
            theorems = model(nodes, edges, edges_structural, goal).cpu()

            for i in range(len(theorems)):
                theorems_str = [str(idx) for idx, _ in sorted(enumerate(theorems[i]), key=lambda x: x[1], reverse=True)]
                theorems_gt_str = [str(idx) for idx, t in enumerate(theorems_gt[i]) if t != 0]

                num_count += 1
                if len(set(theorems_str[:beam_size]) & set(theorems_gt_str)) > 0:
                    acc_count += 1
                results.append(f"GT: [{', '.join(theorems_gt_str)}]\tPD: [{', '.join(theorems_str)}]")

            loop.set_description("Evaluating")

    if save_filename is not None:
        with open(save_filename, 'w', encoding='utf-8') as file:
            for line in results:
                file.write(line + '\n')
            file.write(f"result: {acc_count / num_count}({acc_count}/{num_count})")

    return acc_count / num_count, time.time() - timing


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use PGPS!")

    parser.add_argument("--func", type=str, required=True, default="train", choices=["train", "test"],
                        help="function that you want to run")
    parser.add_argument("--model_name", type=str, required=False,
                        help="The tested model name.")
    parser.add_argument("--nodes_model", type=str, required=False,
                        help="Nodes model name.")
    parser.add_argument("--edges_model", type=str, required=False,
                        help="Edges model name.")
    parser.add_argument("--gs_model", type=str, required=False,
                        help="GS model name.")
    parser.add_argument("--freeze_pretrained", type=bool, required=False, default=True,
                        help="Weather freeze pretrained model para or not.")
    parser.add_argument("--beam_size", type=int, required=True,
                        help="Beam size when calculate acc.")
    parser.add_argument("--device", type=str, required=True, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.func == "train":
        train(args.device, args.nodes_model, args.edges_model, args.gs_model, args.freeze_pretrained, args.beam_size)
    elif args.func == "test":
        test(args.device, args.model_name, args.beam_size)
