from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle
from pgps.model import make_predictor_model
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader


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
                item.insert(0, 0)    # position 0 in edges is <start>, so padding
                item.extend([0] * (config.max_len_edges - len(item)))  # padding

            one_hot_goal.insert(0, 1)  # <start>
            one_hot_goal.extend([0] * (config.max_len_nodes - len(one_hot_goal)))  # padding

            theorems = [0] * config.vocab_theorems
            for t_index in theorems_index:
                theorems[t_index] = 1

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


def train():
    dataset_path = os.path.normpath(os.path.join(config.path_data, "training_data/train"))
    print("Loading training data (the first time loading may be slow)...")
    if "dataset_pgps.pk" in os.listdir(dataset_path):
        dataset = load_pickle(os.path.normpath(os.path.join(dataset_path, "dataset_pgps.pk")))
    else:
        dataset = PGPSDataset(load_pickle(os.path.normpath(os.path.join(dataset_path, "one-hot.pk"))))
        save_pickle(dataset, os.path.normpath(os.path.join(dataset_path, "dataset_pgps.pk")))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    print("Data loading completed.")

    model = make_predictor_model()

    for nodes, edges, edges_structural, goal, theorems in data_loader:
        print("nodes.shape: {}".format(nodes.shape))
        print("edges.shape: {}".format(edges.shape))
        print("edges_structural.shape: {}".format(edges_structural.shape))
        print("goal.shape: {}".format(goal.shape))
        print("theorems.shape: {}".format(theorems.shape))
        result = model(nodes, edges, edges_structural, goal)
        print("result.shape: {}".format(result.shape))
        return


if __name__ == '__main__':
    """
    Loading training data (the first time loading may be slow)...
    Data loading completed.
    nodes.shape: torch.Size([64, 64, 22])
    edges.shape: torch.Size([64, 64, 16])
    edges_structural.shape: torch.Size([64, 64, 16])
    goal.shape: torch.Size([64, 22])
    theorems.shape: torch.Size([64, 251])
    result.shape: torch.Size([64, 251])
    """

    train()
