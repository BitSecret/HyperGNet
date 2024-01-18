from formalgeo.tools import load_json, safe_save_json
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle, nodes_words, edges_words
from pgps.model import make_sentence_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import Levenshtein
from tqdm import tqdm
import argparse


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

        self.mask = torch.tril(torch.ones((config.max_len_nodes, config.max_len_nodes)))

        self.batch_size = config.batch_size_nodes
        self.max_epoch = config.epoch_nodes
        self.lr = config.lr_nodes

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

        self.mask = torch.tril(torch.ones((config.max_len_edges, config.max_len_edges)))

        self.batch_size = config.batch_size_edges
        self.max_epoch = config.epoch_edges
        self.lr = config.lr_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GSDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []
        for _, _, graph_structure, _, _ in raw_data:
            for gs in graph_structure:
                input_seqs = [1] + gs  # <start>
                output_seqs = gs + [2]  # <end>
                input_seqs.extend([0] * (config.max_len_gs - len(input_seqs)))  # padding
                output_seqs.extend([0] * (config.max_len_gs - len(output_seqs)))  # padding
                self.data.append((torch.tensor(input_seqs), torch.tensor(output_seqs)))

        self.mask = torch.tril(torch.ones((config.max_len_gs, config.max_len_gs)))

        self.batch_size = config.batch_size_gs
        self.max_epoch = config.epoch_gs
        self.lr = config.lr_gs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_dataset(model_type, dataset_type):
    """
    Pretrain nodes model and edges model.
    :param model_type: 'nodes', 'edges' or 'gs'.
    :param dataset_type: 'train', 'val' or 'test'.
    """
    onehot_path = os.path.normpath(
        os.path.join(config.path_data, f"training_data/{dataset_type}/one-hot.pk"))
    dataset_path = os.path.normpath(
        os.path.join(config.path_data, f"training_data/{dataset_type}/dataset_{model_type}.pk"))

    if os.path.exists(dataset_path):
        return load_pickle(dataset_path)

    if model_type == "nodes":
        dataset = NodesDataset(load_pickle(onehot_path))
    elif model_type == "edges":
        dataset = EdgesDataset(load_pickle(onehot_path))
    else:
        dataset = GSDataset(load_pickle(onehot_path))
    save_pickle(dataset, dataset_path)

    return dataset


def pretrain(model_type, device):
    """
    Pretrain nodes model and edges model.
    :param model_type: 'nodes', 'edges' or 'gs'.
    :param device: 'cpu' or 'cuda:gpu_id'.
    """
    log_path = os.path.normpath(os.path.join(config.path_data, f"log/{model_type}_pretrain_log.json"))
    loss_save_path = os.path.normpath(os.path.join(config.path_data, "log/{}_pretrain/{}_loss.pk"))
    text_save_path = os.path.normpath(os.path.join(config.path_data, "log/{}_pretrain/{}_eval.text"))
    model_path = os.path.normpath(os.path.join(config.path_data, f"trained_model/{model_type}_model.pth"))
    model_bk_path = os.path.normpath(os.path.join(config.path_data, f"trained_model/{model_type}_model_bk.pth"))
    device = torch.device(device)

    print("Load data and make model (the first time may be slow)...")
    dataset_train = load_dataset(model_type, "train")
    dataset_val = load_dataset(model_type, "val")

    log = {
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "timing": 1}
        "eval": {},  # epoch: {"acc": 1, "timing": 1}
        "best_acc": 0,
        "best_epoch": 0,
    }
    if os.path.exists(log_path):
        log = load_json(log_path)

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=dataset_train.batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=dataset_train.batch_size * 2, shuffle=False)
    mask = dataset_train.mask.to(device)

    model_state = None
    optimizer_state = None
    if log["next_epoch"] > 1:
        last_epoch_state = torch.load(model_bk_path, map_location=torch.device("cpu"))
        model_state = last_epoch_state["model"]
        optimizer_state = last_epoch_state["optimizer"]

    model = make_sentence_model(model_type, model_state).to(device)  # make model

    optimizer = torch.optim.Adam(model.parameters(), lr=dataset_train.lr)  # Adam optimizer
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    for epoch in range(log["next_epoch"], dataset_train.max_epoch + 1):
        step_list = []
        loss_list = []
        model.train()
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (input_seqs, output_seqs_gt) in loop:
            input_seqs, output_seqs_gt = input_seqs.to(device), output_seqs_gt.to(device)
            outputs = model(x_input=input_seqs, x_output=input_seqs, mask=mask)  # output
            loss = loss_func(outputs.transpose(1, 2), output_seqs_gt)  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize param

            step_list.append((epoch - 1) * len(loop) + step)
            loss_list.append(float(loss))
            loop.set_description(f"Pretraining {model_type} (epoch [{epoch}/{dataset_train.max_epoch}])")
            loop.set_postfix(loss=float(loss))

        save_pickle((step_list, loss_list), loss_save_path.format(model_type, epoch))
        log["train"][str(epoch)] = {
            "avg_loss": sum(loss_list) / len(loss_list),
            "timing": time.time() - timing
        }

        acc, timing = evaluate(model, data_loader_val, model_type, device, mask,
                               text_save_path.format(model_type, epoch))
        log["eval"][str(epoch)] = {"acc": acc, "timing": timing}
        if acc > log["best_acc"]:
            log["best_acc"] = acc
            log["best_epoch"] = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_bk_path)
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


def test(model_type, device, model_name):
    """
    Test pretrained model.
    :param model_type: 'nodes', 'edges' or 'gs'.
    :param device: 'cpu' or 'cuda:gpu_id'.
    :param model_name: tested model name in data/trained_model.
    """
    print("Load data and make model (the first time may be slow)...")
    text_path = os.path.normpath(os.path.join(config.path_data, f"log/test/{model_type}_eval.text"))
    results_path = os.path.normpath(os.path.join(config.path_data, f"log/test/{model_type}_test_log.json"))
    device = torch.device(device)

    state_dict = torch.load(
        os.path.normpath(os.path.join(config.path_data, "trained_model", model_name)), map_location=torch.device("cpu")
    )["model"]
    model = make_sentence_model(model_type, state_dict).to(device)

    dataset = load_dataset(args.model_type, "test")
    data_loader = DataLoader(dataset=dataset, batch_size=dataset.batch_size * 2, shuffle=False)

    acc, timing = evaluate(model, data_loader, model_type, device, dataset.mask.to(device), text_path)

    safe_save_json({"acc": acc, "timing": timing}, results_path)
    print(f"Acc: {acc}. Details saved in {text_path}")


def evaluate(model, data_loader, model_type, device, mask, save_filename=None):
    """
    Evaluate pretrain and save evaluation results.
    :param model: tested model, instance of <Sentence2Vector>.
    :param data_loader: test datasets data loader.
    :param model_type: 'nodes', 'edges' or 'gs'.
    :param device: device that model run, torch.device().
    :param mask: tril mask.
    :param save_filename: file that save evaluation results.
    :return acc: Weighted Levenshtein ratio.
    :return timing: timing.
    """
    timing = time.time()
    model.eval()
    acc_count = 0  # edition distance
    num_count = 0  # sequence length
    results = []
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)  # evaluating loop
    with torch.no_grad():
        for step, (input_seqs, output_seqs_gt) in loop:
            input_seqs = input_seqs.to(device)
            input_encoding = model(x_input=input_seqs)
            output_seqs = torch.zeros(size=output_seqs_gt.size(), dtype=torch.int).to(device)
            start_vector = torch.ones(size=(output_seqs_gt.size(0), 1), dtype=torch.int).to(device)
            for i in range(config.max_len_nodes):  # iterative decoding
                output_seqs = torch.cat([start_vector, output_seqs], dim=1)[:, :-1]
                output_seqs = model(x_encoding=input_encoding, x_output=output_seqs, mask=mask)
                output_seqs = torch.argmax(output_seqs, dim=2).int()
            output_seqs = output_seqs.cpu()

            for i in range(output_seqs.size(0)):
                seqs_unicode = []  # for compute Levenshtein ratio
                seqs_sematic = []  # for human check
                for j in range(len(output_seqs[i])):
                    if output_seqs[i][j] == 2:  # <end>
                        break
                    seqs_unicode.append(chr(output_seqs[i][j]))
                    if model_type == "nodes":
                        seqs_sematic.append(nodes_words[output_seqs[i][j]])
                    elif model_type == "edges":
                        seqs_sematic.append(edges_words[output_seqs[i][j]])
                    else:
                        seqs_sematic.append(str(output_seqs[i][j].item()))

                seqs_gt_unicode = []  # for compute Levenshtein ratio
                seqs_gt_sematic = []  # for human check
                for j in range(len(output_seqs_gt[i])):
                    if output_seqs_gt[i][j] == 2:  # <end>
                        break
                    seqs_gt_unicode.append(chr(output_seqs_gt[i][j]))
                    if model_type == "nodes":
                        seqs_gt_sematic.append(nodes_words[output_seqs_gt[i][j]])
                    elif model_type == "edges":
                        seqs_gt_sematic.append(edges_words[output_seqs_gt[i][j]])
                    else:
                        seqs_gt_sematic.append(str(output_seqs_gt[i][j].item()))

                acc_count += Levenshtein.ratio("".join(seqs_unicode), "".join(seqs_gt_unicode)) * len(seqs_gt_unicode)
                num_count += len(seqs_gt_unicode)

                if model_type == "nodes":
                    seqs_sematic.insert(1, "(")
                    seqs_sematic.append(")")
                    seqs_gt_sematic.insert(1, "(")
                    seqs_gt_sematic.append(")")
                    results.append(f"GT: {''.join(seqs_gt_sematic)}\tPD: {''.join(seqs_sematic)}")
                else:
                    results.append(f"GT: {', '.join(seqs_gt_sematic)}\tPD: {', '.join(seqs_sematic)}")

            loop.set_description(f"Evaluating")

    if save_filename is not None:
        with open(save_filename, 'w', encoding='utf-8') as file:
            for line in results:
                file.write(line + '\n')

    return acc_count / num_count, time.time() - timing


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use PGPS!")

    parser.add_argument("--func", type=str, required=True, default="pretrain", choices=["pretrain", "test"],
                        help="Function that you want to run.")
    parser.add_argument("--device", type=str, required=True, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")
    parser.add_argument("--model_type", type=str, required=True, default="nodes", choices=["nodes", "edges", "gs"],
                        help="Pretrained model type.")
    parser.add_argument("--model_name", type=str, required=False,
                        help="The tested model name.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.func == "pretrain":
        pretrain(model_type=args.model_type, device=args.device)
    elif args.func == "test":
        test(model_type=args.model_type, device=args.device, model_name=args.model_name)
