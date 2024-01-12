from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle
from pgps.model import make_nodes_model, make_edges_model
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


class NodesDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []
        for one_hot_nodes, _, _, one_hot_goal, _ in raw_data:
            for node in one_hot_nodes:
                node.insert(0, 1)  # <start>
                node.append(2)  # <end>
                node.extend([0] * (config.max_len_nodes - len(node)))  # padding
                self.data.append(torch.tensor(node))
            one_hot_goal.insert(0, 1)  # <start>
            one_hot_goal.append(2)  # <end>
            one_hot_goal.extend([0] * (config.max_len_nodes - len(one_hot_goal)))  # padding
            self.data.append(torch.tensor(one_hot_goal))

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
                edge.insert(0, 1)  # <start>
                edge.append(2)  # <end>
                edge.extend([0] * (config.max_len_edges - len(edge)))  # padding
                self.data.append(torch.tensor(edge))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_nodes_model():
    dataset_path = os.path.normpath(os.path.join(config.path_data, "training_data/train"))
    print("Loading nodes data (the first time loading may be slow)...")
    if "dataset_nodes.pk" in os.listdir(dataset_path):
        dataset = load_pickle(os.path.normpath(os.path.join(dataset_path, "dataset_nodes.pk")))
    else:
        dataset = NodesDataset(load_pickle(os.path.normpath(os.path.join(dataset_path, "one-hot.pk"))))
        save_pickle(dataset, os.path.normpath(os.path.join(dataset_path, "dataset_nodes.pk")))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size_nodes,
        shuffle=False
    )
    print("Nodes Data loading completed.")

    model = make_nodes_model()

    for batch_data in data_loader:
        print("batch_data.shape: {}".format(batch_data.shape))
        result = model(batch_data)
        print("result.shape: {}".format(result.shape))
        return


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

    for batch_data in data_loader:
        print("batch_data.shape: {}".format(batch_data.shape))
        result = model(batch_data)
        print("result.shape: {}".format(result.shape))
        return


if __name__ == '__main__':
    """
    Loading nodes data (the first time loading may be slow)...
    Nodes Data loading completed.
    batch_data.shape: torch.Size([64, 22])
    result.shape: torch.Size([64, 22, 144])

    Loading edges data (the first time loading may be slow)...
    Edges Data loading completed.
    batch_data.shape: torch.Size([64, 16])
    result.shape: torch.Size([64, 16, 257])
    """

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    train_nodes_model()
    print()
    train_edges_model()
