import time
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle, nodes_words
from pgps.model import make_nodes_model, make_edges_model
from formalgeo.tools import load_json, safe_save_json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os
import Levenshtein
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


def evaluate(output_seqs_list, save_filename=None):
    score_count = 0
    num_count = 0
    results = []
    for seqs, seqs_gt in output_seqs_list:  # trans to srt
        for i in range(seqs.size(0)):
            seqs_cleand = []
            for j in range(len(seqs[i])):
                if seqs[i][j] == 2:  # <end>
                    break
                # seqs_cleand.append(chr(seqs[i][j]))
                seqs_cleand.append(nodes_words[seqs[i][j]])
            seqs_gt_cleand = []
            for j in range(len(seqs_gt[i])):
                if seqs_gt[i][j] == 2:  # <end>
                    break
                # seqs_gt_cleand.append(chr(seqs_gt[i][j]))
                seqs_gt_cleand.append(nodes_words[seqs_gt[i][j]])

            seqs_cleand = "".join(seqs_cleand)
            seqs_gt_cleand = "".join(seqs_gt_cleand)

            score_count += Levenshtein.ratio(seqs_cleand, seqs_gt_cleand) * len(seqs_gt_cleand)  # edition distance
            num_count += len(seqs_gt_cleand)
            results.append(f"GT: {seqs_gt_cleand}\tPD: {seqs_cleand}")

    if save_filename is not None:
        with open(save_filename, 'w', encoding='utf-8') as file:
            for line in results:
                file.write(line + '\n')

    return score_count / num_count


def train_nodes_model():
    dataset_path_train = os.path.normpath(os.path.join(config.path_data, "training_data/train"))
    dataset_path_val = os.path.normpath(os.path.join(config.path_data, "training_data/val"))
    log_path = os.path.normpath(os.path.join(config.path_data, "log/training_log_nodes_model.json"))
    model_save_path = os.path.normpath(os.path.join(config.path_data, "trained_model"))
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    model_filename = None
    if log["next_epoch"] > 1:
        model_filename = os.path.normpath(os.path.join(model_save_path, f"nodes_{log['next_epoch'] - 1}.pth"))
    model = make_nodes_model(model_filename).to(device)  # make model

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_nodes)  # Adam optimizer
    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    tril_mask = torch.tril(torch.ones((config.max_len_nodes, config.max_len_nodes))).to(device)

    for epoch in range(log["next_epoch"], config.epoch_nodes + 1):
        model.train()
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (input_seqs, output_seqs_gt) in loop:
            input_seqs, output_seqs_gt = input_seqs.to(device), output_seqs_gt.to(device)
            outputs = model(x_input=input_seqs, x_output=input_seqs, mask=tril_mask)  # output
            loss = loss_func(outputs.transpose(1, 2), output_seqs_gt)  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize para
            log["train"]["step"].append((epoch - 1) * len(loop) + step)
            log["train"]["loss"].append(float(loss))
            loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Training)")
            loop.set_postfix(loss=float(loss))
        log["train"]["timing"].append(time.time() - timing)

        model.eval()
        timing = time.time()
        output_seqs_list = []
        loop = tqdm(enumerate(data_loader_eval), total=len(data_loader_eval), leave=True)  # evaluating loop
        with torch.no_grad():
            for step, (input_seqs, output_seqs_gt) in loop:
                input_seqs = input_seqs.to(device)
                input_encoding = model(x_input=input_seqs)
                output_seqs = torch.zeros(size=output_seqs_gt.size(), dtype=torch.int).to(device)
                start_vector = torch.ones(size=(output_seqs_gt.size(0), 1), dtype=torch.int).to(device)
                for i in range(config.max_len_nodes):  # iterative decoding
                    output_seqs = torch.cat([start_vector, output_seqs], dim=1)[:, :-1]
                    output_seqs = model(x_encoding=input_encoding, x_output=output_seqs, mask=tril_mask)
                    output_seqs = torch.argmax(output_seqs, dim=2).int()

                output_seqs_list.append((output_seqs.cpu(), output_seqs_gt))
                loop.set_description(f"Epoch [{epoch}/{config.epoch_nodes}] (Evaluating)")

        print("Calculate the predicted results...")
        save_filename = os.path.normpath(os.path.join(config.path_data, f"log/eval_detail/nodes_{epoch}.text"))
        log["eval"][str(epoch)] = {"acc": evaluate(output_seqs_list, save_filename), "timing": time.time() - timing}

        model_filename = os.path.normpath(os.path.join(model_save_path, f"nodes_{log['next_epoch']}.pth"))
        torch.save(model.state_dict(), model_filename)
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


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

    for input_seqs, output_seqs_gt in data_loader:
        print("input_seqs.shape: {}".format(input_seqs.shape))
        print("output_seqs_gt.shape: {}".format(output_seqs_gt.shape))
        result = model(input_seqs)
        print("result.shape: {}".format(result.shape))
        return


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

    train_nodes_model()
    # print()
    # train_edges_model()
