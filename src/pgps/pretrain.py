from pgps.utils import Configuration as config
from pgps.utils import load_pickle, save_pickle
from pgps.module import init_weights
from pgps.model import Sentence2Vector
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


def make_nodes_model():
    model = Sentence2Vector(
        vocab=config.vocab_nodes,
        d_model=config.d_model // 2,
        max_len=config.max_len_nodes,
        h=config.h_nodes,
        N_encoder=config.N_encoder_nodes,
        N_decoder=config.N_decoder_nodes,
        p_drop=config.p_drop_nodes
    )
    model.apply(init_weights)
    return model


def make_edges_model():
    model = Sentence2Vector(
        vocab=config.vocab_edges,
        d_model=config.d_model,
        max_len=config.max_len_edges,
        h=config.h_edges,
        N_encoder=config.N_encoder_edges,
        N_decoder=config.N_decoder_edges,
        p_drop=config.p_drop_nodes
    )
    model.apply(init_weights)
    return model


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
    print("Data loading completed.")

    for d in data_loader:
        print(d.shape)
        exit(0)

    # model = make_nodes_model()


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
    print("Data loading completed.")

    for d in data_loader:
        print(d.shape)
        exit(0)

    # model = make_edges_model()


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)

    train_nodes_model()
    train_edges_model()
