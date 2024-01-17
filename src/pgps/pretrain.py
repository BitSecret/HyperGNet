from formalgeo.tools import load_json, safe_save_json
from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle, nodes_words, edges_words, get_args
from pgps.model import make_nodes_model, make_edges_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
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


def evaluate(output_seqs_list, name, save_filename=None):
    """
    Evaluate pretrain and save evaluation results.
    :param output_seqs_list: list of (seqs, seqs_gt), seqs/seqs_gt: torch.Size([batch_size, max_len]).
    :param name: 'nodes' or 'edges'.
    :param save_filename: File that saving evaluation results.
    :return acc: Weighted Levenshtein ratio.
    """
    acc_count = 0  # edition distance
    num_count = 0  # sequence length
    results = []
    for seqs, seqs_gt in output_seqs_list:  # trans to srt
        for i in range(seqs.size(0)):
            seqs_unicode = []  # for compute Levenshtein ratio
            seqs_sematic = []  # for human check
            for j in range(len(seqs[i])):
                if seqs[i][j] == 2:  # <end>
                    break
                seqs_unicode.append(chr(seqs[i][j]))
                if name == "nodes":
                    seqs_sematic.append(nodes_words[seqs[i][j]])
                else:
                    seqs_sematic.append(edges_words[seqs[i][j]])

            seqs_gt_unicode = []  # for compute Levenshtein ratio
            seqs_gt_sematic = []  # for human check
            for j in range(len(seqs_gt[i])):
                if seqs_gt[i][j] == 2:  # <end>
                    break
                seqs_gt_unicode.append(chr(seqs_gt[i][j]))
                if name == "nodes":
                    seqs_gt_sematic.append(nodes_words[seqs_gt[i][j]])
                else:
                    seqs_gt_sematic.append(edges_words[seqs_gt[i][j]])

            acc_count += Levenshtein.ratio("".join(seqs_unicode), "".join(seqs_gt_unicode)) * len(seqs_gt_unicode)
            num_count += len(seqs_gt_unicode)

            if name == "nodes":
                seqs_sematic.insert(1, "(")
                seqs_sematic.append(")")
                seqs_gt_sematic.insert(1, "(")
                seqs_gt_sematic.append(")")
                results.append(f"GT: {''.join(seqs_gt_sematic)}\tPD: {''.join(seqs_sematic)}")
            else:
                results.append(f"GT: {', '.join(seqs_gt_sematic)}\tPD: {', '.join(seqs_sematic)}")

    if save_filename is not None:
        with open(save_filename, 'w', encoding='utf-8') as file:
            for line in results:
                file.write(line + '\n')

    return acc_count / num_count


def pretrain(name="nodes"):
    """
    Pretrain nodes model and edges model.
    :param name: 'nodes' or 'edges'.
    """
    onehot_train_path = os.path.normpath(os.path.join(config.path_data, "training_data/train/one-hot.pk"))
    onehot_val_path = os.path.normpath(os.path.join(config.path_data, "training_data/val/one-hot.pk"))
    dataset_train_path = os.path.normpath(os.path.join(config.path_data, f"training_data/train/dataset_{name}.pk"))
    dataset_val_path = os.path.normpath(os.path.join(config.path_data, f"training_data/val/dataset_{name}.pk"))
    log_path = os.path.normpath(os.path.join(config.path_data, f"log/{name}_pretrain_log.json"))

    loss_save_path = os.path.normpath(os.path.join(config.path_data, "log/{}_pretrain/{}_loss.pk"))
    text_save_path = os.path.normpath(os.path.join(config.path_data, "log/{}_pretrain/{}_eval.text"))
    model_save_path = os.path.normpath(os.path.join(config.path_data, "trained_model/{}_pretrain_{}.pth"))

    print("Loading data (the first time loading may be slow)...")
    if os.path.exists(dataset_train_path):
        dataset_train = load_pickle(dataset_train_path)
    else:
        dataset_train = NodesDataset(load_pickle(onehot_train_path)) if name == "nodes" \
            else EdgesDataset(load_pickle(onehot_train_path))
        save_pickle(dataset_train, dataset_train_path)
    if os.path.exists(dataset_val_path):
        dataset_val = load_pickle(dataset_val_path)
    else:
        dataset_val = NodesDataset(load_pickle(onehot_val_path)) if name == "nodes" \
            else EdgesDataset(load_pickle(onehot_val_path))
        save_pickle(dataset_val, dataset_val_path)
    batch_size = config.batch_size_nodes if name == "nodes" else config.batch_size_edges
    max_epoch = config.epoch_nodes if name == "nodes" else config.epoch_edges
    lr = config.lr_nodes if name == "nodes" else config.lr_edges
    log = {
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "lr": lr,
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "timing": 1}
        "eval": {}  # epoch: {"acc": 1, "timing": 1}
    }
    if os.path.exists(log_path):
        log = load_json(log_path)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_eval = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    print("Data loading completed.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    model_state = None
    optimizer_state = None
    if log["next_epoch"] > 1:
        last_epoch_msg = load_pickle(model_save_path.format(name, log['next_epoch'] - 1))
        model_state = last_epoch_msg["model"]
        optimizer_state = last_epoch_msg["optimizer"]

    model = make_nodes_model(model_state).to(device) if name == "nodes" \
        else make_edges_model(model_state).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    tril_mask = torch.tril(torch.ones((config.max_len_nodes, config.max_len_nodes))).to(device) if name == "nodes" \
        else torch.tril(torch.ones((config.max_len_edges, config.max_len_edges))).to(device)

    for epoch in range(log["next_epoch"], max_epoch + 1):
        step_list = []
        loss_list = []
        model.train()
        timing = time.time()
        loop = tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True)  # training loop
        for step, (input_seqs, output_seqs_gt) in loop:
            input_seqs, output_seqs_gt = input_seqs.to(device), output_seqs_gt.to(device)
            outputs = model(x_input=input_seqs, x_output=input_seqs, mask=tril_mask)  # output
            loss = loss_func(outputs.transpose(1, 2), output_seqs_gt)  # loss
            optimizer.zero_grad()  # clean grad
            loss.backward()  # backward
            optimizer.step()  # optimize param

            step_list.append((epoch - 1) * len(loop) + step)
            loss_list.append(float(loss))
            loop.set_description(f"Epoch [{epoch}/{max_epoch}] (Pretraining)")
            loop.set_postfix(loss=float(loss))

        save_pickle((step_list, loss_list), loss_save_path.format(name, epoch))
        log["train"][str(epoch)] = {
            "avg_loss": sum(loss_list) / len(loss_list),
            "timing": time.time() - timing
        }

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
                loop.set_description(f"Epoch [{epoch}/{max_epoch}] (Evaluating)")

        print("Calculate the predicted results...")

        log["eval"][str(epoch)] = {
            "acc": evaluate(output_seqs_list, name=name, save_filename=text_save_path.format(name, epoch)),
            "timing": time.time() - timing
        }

        save_pickle(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            model_save_path.format(name, epoch)
        )
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


if __name__ == '__main__':
    args = get_args()
    if args.func == "nodes":
        pretrain(name="nodes")
    elif args.func == "edges":
        pretrain(name="edges")
    else:
        msg = "No function name {}.".format(args.func)
        raise Exception(msg)
