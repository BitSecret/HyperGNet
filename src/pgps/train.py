from formalgeo.tools import load_json, safe_save_json
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle, get_args
from pgps.model import make_predictor_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
from tqdm import tqdm
import heapq


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


def evaluate(output_list, beam_size, save_filename=None):
    """
    Evaluate pretrain and save evaluation results.
    :param output_list: list of (theorems, theorems_gt), theorems/theorems_gt: torch.Size([batch_size, vocab_theorems]).
    :param beam_size: beam size when calculate acc.
    :param save_filename: File that saving evaluation results.
    :return acc: theorem prediction acc.
    """
    acc_count = 0  # edition distance
    num_count = 0  # sequence length
    results = []
    for theorems, theorems_gt in output_list:  # trans to srt
        for i in range(len(theorems)):
            theorems_idx = [idx for _, idx in
                            heapq.nlargest(beam_size, [(t, idx) for idx, t in enumerate(theorems[i])])]
            theorems_gt_idx = [idx for idx, t in enumerate(theorems_gt[i]) if t != 0]

            num_count += 1
            if len(set(theorems_idx) & set(theorems_gt_idx)) > 0:
                acc_count += 1
            results.append(f"GT: [{', '.join(theorems_gt_idx)}]\tPD: [{', '.join(theorems_idx)}]")

    if save_filename is not None:
        with open(save_filename, 'w', encoding='utf-8') as file:
            for line in results:
                file.write(line + '\n')

    return acc_count / num_count


def train(nodes_model_state_dict=None, edges_model_state_dict=None):
    """Train theorem prediction model."""
    onehot_train_path = os.path.normpath(os.path.join(config.path_data, "training_data/train/one-hot.pk"))
    onehot_val_path = os.path.normpath(os.path.join(config.path_data, "training_data/val/one-hot.pk"))
    dataset_train_path = os.path.normpath(os.path.join(config.path_data, "training_data/train/dataset_pgps.pk"))
    dataset_val_path = os.path.normpath(os.path.join(config.path_data, "training_data/val/dataset_pgps.pk"))
    log_path = os.path.normpath(os.path.join(config.path_data, "log/pgps_train_log.json"))

    loss_save_path = os.path.normpath(os.path.join(config.path_data, "log/train/{}_loss.pk"))
    text_save_path = os.path.normpath(os.path.join(config.path_data, "log/train/{}_eval.text"))
    model_save_path = os.path.normpath(os.path.join(config.path_data, "trained_model/train_{}.pth"))

    print("Loading data (the first time loading may be slow)...")
    if os.path.exists(dataset_train_path):
        dataset_train = load_pickle(dataset_train_path)
    else:
        dataset_train = PGPSDataset(load_pickle(onehot_train_path))
        save_pickle(dataset_train, dataset_train_path)
    if os.path.exists(dataset_val_path):
        dataset_val = load_pickle(dataset_val_path)
    else:
        dataset_val = PGPSDataset(load_pickle(onehot_val_path))
        save_pickle(dataset_val, dataset_val_path)
    log = {
        "batch_size": config.batch_size,
        "max_epoch": config.epoch,
        "lr": config.lr,
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "timing": 1}
        "eval": {}  # epoch: {"acc": 1, "timing": 1}
    }
    if os.path.exists(log_path):
        log = load_json(log_path)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_eval = DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=False)
    print("Nodes loading completed.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    model_filename = None
    if log["next_epoch"] > 1:
        model_filename = model_save_path.format(log['next_epoch'] - 1)
    model = make_predictor_model(model_filename, nodes_model_state_dict, edges_model_state_dict).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Adam optimizer
    loss_func = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    for epoch in range(log["next_epoch"], config.epoch + 1):
        step_list = []
        loss_list = []
        model.train()
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (nodes, edges, edges_structural, goal, theorems_gt) in loop:
            nodes = nodes.to(device)
            edges = edges.to(device)
            edges_structural = edges_structural.to(device)
            goal = goal.to(device)
            theorems_gt = theorems_gt.to(device)
            theorems = model(nodes, edges, edges_structural, goal)  # model prediction results
            loss = loss_func(theorems, theorems_gt)  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize param

            step_list.append((epoch - 1) * len(loop) + step)
            loss_list.append(float(loss))
            loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Pretraining)")
            loop.set_postfix(loss=float(loss))

        save_pickle((step_list, loss_list), loss_save_path.format(epoch))
        log["train"][str(epoch)] = {
            "avg_loss": sum(loss_list) / len(loss_list),
            "timing": time.time() - timing
        }

        model.eval()
        timing = time.time()
        output_list = []
        loop = tqdm(enumerate(data_loader_eval), total=len(data_loader_eval), leave=True)  # evaluating loop
        with torch.no_grad():
            for step, (nodes, edges, edges_structural, goal, theorems_gt) in loop:
                nodes = nodes.to(device)
                edges = edges.to(device)
                edges_structural = edges_structural.to(device)
                goal = goal.to(device)
                theorems = model(nodes, edges, edges_structural, goal)

                output_list.append((theorems.cpu(), theorems_gt))
                loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Evaluating)")

        print("Calculate the predicted results...")

        log["eval"][str(epoch)] = {
            "acc": evaluate(output_list, beam_size=config.beam_size, save_filename=text_save_path.format(epoch)),
            "timing": time.time() - timing
        }

        save_pickle(model.state_dict(), model_save_path.format(epoch))
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


def test(model_state_dict):
    pass


if __name__ == '__main__':
    args = get_args()
    if args.func == "train":
        best_nodes_model_path = os.path.normpath(
            os.path.join(config.path_data, f"log/trained_model/{args.nodes_model}"))
        best_edges_model_path = os.path.normpath(
            os.path.join(config.path_data, f"log/trained_model/{args.edges_model}"))
        train(load_pickle(best_nodes_model_path), load_pickle(best_edges_model_path))
    elif args.func == "test":
        best_model_path = os.path.normpath(
            os.path.join(config.path_data, f"log/trained_model/{args.predictor_model}"))
        test(load_pickle(best_model_path))
    else:
        msg = "No function name {}.".format(args.func)
        raise Exception(msg)
