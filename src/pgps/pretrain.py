import copy
import time
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle
from pgps.model import make_nodes_model, make_edges_model
from formalgeo.tools import load_json, safe_save_json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os
from tqdm import tqdm


class NodesDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []
        for one_hot_nodes, _, _, one_hot_goal, _ in raw_data:
            for node in one_hot_nodes:
                input_seqs = [1] + node  # <start>
                output_seqs = node + [2]  # <end>
                input_seqs.extend([0] * (config.max_len_nodes - len(input_seqs)))  # padding
                output_seqs.extend([0] * (config.max_len_nodes - len(output_seqs)))  # padding
                self.data.append((torch.tensor(input_seqs), torch.tensor(output_seqs)))
            input_seqs = [1] + one_hot_goal  # <start>
            output_seqs = one_hot_goal + [2]  # <end>
            input_seqs.extend([0] * (config.max_len_nodes - len(input_seqs)))  # padding
            output_seqs.extend([0] * (config.max_len_nodes - len(output_seqs)))  # padding
            self.data.append((torch.tensor(input_seqs), torch.tensor(output_seqs)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returned Data point must be a Tensor."""
        return self.data[idx]


class EdgesDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []
        for _, one_hot_edges, _, _, _ in raw_data:
            for edge in one_hot_edges:
                input_seqs = [1] + edge  # <start>
                output_seqs = edge + [2]  # <end>
                input_seqs.extend([0] * (config.max_len_edges - len(input_seqs)))  # padding
                output_seqs.extend([0] * (config.max_len_edges - len(output_seqs)))  # padding
                self.data.append((torch.tensor(input_seqs), torch.tensor(output_seqs)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_nodes_model():
    dataset_path_train = os.path.normpath(os.path.join(config.path_data, "training_data/train"))
    dataset_path_val = os.path.normpath(os.path.join(config.path_data, "training_data/val"))
    log_path = os.path.normpath(os.path.join(config.path_data, "log/training_log_nodes_model.json"))
    print("Loading nodes data (the first time loading may be slow)...")
    if "dataset_nodes.pk" in os.listdir(dataset_path_train):
        dataset_train = load_pickle(os.path.normpath(os.path.join(dataset_path_train, "dataset_nodes.pk")))
    else:
        dataset_train = NodesDataset(load_pickle(os.path.normpath(os.path.join(dataset_path_train, "one-hot.pk"))))
        save_pickle(dataset_train, os.path.normpath(os.path.join(dataset_path_train, "dataset_nodes.pk")))

    if "dataset_nodes.pk" in os.listdir(dataset_path_val):
        dataset_eval = load_pickle(os.path.normpath(os.path.join(dataset_path_val, "dataset_nodes.pk")))
    else:
        dataset_eval = NodesDataset(load_pickle(os.path.normpath(os.path.join(dataset_path_val, "one-hot.pk"))))
        save_pickle(dataset_eval, os.path.normpath(os.path.join(dataset_path_val, "dataset_nodes.pk")))
    log = {
        "batch_size": config.batch_size_nodes,
        "lr": config.lr_nodes,
        "max_epoch": config.epoch_nodes,
        "next_epoch": 1,
        "train": {
            "step": [],
            "loss": [],
            "timing": []
        },
        "eval": {}  # epoch: {"acc": 1, "timing": 1}
    }
    if os.path.exists(log_path):
        log = load_json(log_path)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size_nodes, shuffle=True)
    data_loader_eval = DataLoader(dataset=dataset_eval, batch_size=config.batch_size_nodes, shuffle=False)
    print("Nodes Data loading completed.")

    model = make_nodes_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_nodes)  # Adam optimizer
    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    tril_mask = torch.tril(torch.ones((config.max_len_nodes, config.max_len_nodes)))

    for epoch in range(log["next_epoch"], config.epoch_nodes + 1):
        model.train()
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (input_seqs, output_seqs_gt) in loop:
            outputs = model(x_input=input_seqs, x_output=input_seqs, mask=tril_mask)  # output
            loss = loss_func(outputs.transpose(1, 2), output_seqs_gt)  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize para
            log["train"]["step"].append((epoch - 1) * len(loop) + step)
            log["train"]["loss"].append(float(loss))
            loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Training)")
            loop.set_postfix(loss=float(loss))
            break
        log["train"]["timing"].append(time.time() - timing)

        model.eval()
        timing = time.time()
        output_seqs_list = []
        output_seqs_gt_list = []
        loop = tqdm(enumerate(data_loader_eval), total=len(data_loader_eval), leave=True)  # evaluating loop
        with torch.no_grad():
            for step, (input_seqs, output_seqs_gt) in loop:
                input_encoding = model(x_input=input_seqs)
                output_seqs = torch.zeros(size=output_seqs_gt.size(), dtype=torch.int)
                start_vector = torch.ones(size=(output_seqs_gt.size(0), 1), dtype=torch.int)
                for i in range(config.max_len_nodes):
                    output_seqs = torch.cat([start_vector, output_seqs], dim=1)[:, :-1]
                    output_seqs = model(x_encoding=input_encoding, x_output=output_seqs, mask=tril_mask)
                    output_seqs = torch.argmax(output_seqs, dim=2).int()

                output_seqs_list.append(output_seqs)
                output_seqs_gt_list.append(output_seqs_gt)
                loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Evaluating)")
                break
        print("Calculate the predicted results...")
        acc = evaluate_nodes_model(output_seqs_list, output_seqs_gt_list)
        log["eval"][str(epoch)] = [acc, time.time() - timing]

        # 保存模型
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


def evaluate_nodes_model(output_seqs_list, output_seqs_gt_list):
    return 1


def train_edges_model():
    dataset_path = os.path.normpath(os.path.join(config.path_data, "training_data/train"))
    print("Loading edges data (the first time loading may be slow)...")
    if "dataset_edges.pk" in os.listdir(dataset_path):
        dataset = load_pickle(os.path.normpath(os.path.join(dataset_path, "dataset_edges.pk")))
    else:
        dataset = EdgesDataset(load_pickle(os.path.normpath(os.path.join(dataset_path, "one-hot.pk"))))
        save_pickle(dataset, os.path.normpath(os.path.join(dataset_path, "dataset_edges.pk")))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size_edges,
        shuffle=True
    )
    print("Edges Data loading completed.")

    model = make_edges_model()

    for input_seqs, output_seqs in data_loader:
        print("input_seqs.shape: {}".format(input_seqs.shape))
        print("output_seqs.shape: {}".format(output_seqs.shape))
        result = model(input_seqs)
        print("result.shape: {}".format(result.shape))
        return


def evaluate_edges_model(model, data_loader):
    pass


if __name__ == '__main__':
    """
    Loading nodes data (the first time loading may be slow)...
    Nodes Data loading completed.
    input_seqs.shape: torch.Size([64, 22])
    output_seqs.shape: torch.Size([64, 22])
    result.shape: torch.Size([64, 22, 144])

    Loading edges data (the first time loading may be slow)...
    Edges Data loading completed.
    input_seqs.shape: torch.Size([64, 16])
    output_seqs.shape: torch.Size([64, 16])
    result.shape: torch.Size([64, 16, 257])
    """

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    train_nodes_model()
    print()
    # train_edges_model()
