from pgps.utils import Configuration as config
from pgps.module import Embedding, SelfAttention, TaskSpecificAttention, LayerNorm, FeedForward, init_weights
from pgps.pretrain import Sentence2Vector
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Encoder, self).__init__()
        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.dp_attn = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.dp_ffd = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len_predictor, d_model])
        :return result: torch.Size([batch_size, max_len_predictor, d_model])
        """
        for i in range(len(self.attentions)):
            # multi-head attention, dropout, res-net and layer norm
            x = self.ln_attn[i](x + self.dp_attn[i](self.attentions[i](x)))
            # feedforward, dropout, res-net and layer norm
            x = self.ln_ffd[i](x + self.dp_ffd[i](self.feedforwards[i](x)))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Sentence decoder, decode sentence embedding to raw sentence.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Decoder, self).__init__()

        self.attentions = nn.ModuleList([TaskSpecificAttention(h, d_model) for _ in range(N)])
        self.dp_attn = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.dp_ffd = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x, task):
        """
        :param x: torch.Size([batch_size, max_len_predictor, d_model])
        :param task: torch.Size([batch_size, d_model])
        :return result: torch.Size([batch_size, d_model])
        """
        for i in range(len(self.attentions)):
            # task-specific attention, dropout, resnet and layer norm
            task = self.ln_attn[i](task + self.dp_attn[i](self.attentions[i](x, task)))
            # feedforward, dropout、res-net and layer norm
            task = self.ln_ffd[i](task + self.dp_ffd[i](self.feedforwards[i](task)))
        return task


class Predictor(nn.Module):

    def __init__(self, vocab_nodes, max_len_nodes, h_nodes, N_encoder_nodes, N_decoder_nodes, p_drop_nodes,
                 vocab_edges, max_len_edges, h_edges, N_encoder_edges, N_decoder_edges, p_drop_edges,
                 vocab, max_len, h, N, p_drop, d_model):
        """
        Theorem predictor.
        :param vocab: The number of words in nodes/path/theorems.
        :param max_len: Max len of words/path/nodes.
        :param h: Head number of MultiHeadAttention in Sentence2Vector/Predictor.
        :param N: Number of MultiHeadAttention in Sentence2Vector/Predictor.
        :param p_drop: Dropout rate.
        :param d_model: Hidden dim.
        """
        super(Predictor, self).__init__()
        self.max_len_nodes = max_len_nodes
        self.max_len_edges = max_len_edges
        self.max_len = max_len
        self.d_model = d_model

        self.nodes_emb = Sentence2Vector(
            vocab_nodes, d_model, max_len_nodes, h_nodes, N_encoder_nodes, N_decoder_nodes, p_drop_nodes)
        self.edges_emb = Sentence2Vector(
            vocab_edges, d_model, max_len_edges, h_edges, N_encoder_edges, N_decoder_edges, p_drop_edges)

        self.encoder = Encoder(d_model, h, N, p_drop)
        self.decoder = Decoder(d_model, h, N, p_drop)  # self.decoder = TaskSpecificAttention(h, d_model)

        self.linear = nn.Linear(d_model, vocab)

    def forward(self, nodes, edges, goal):
        """
        :param nodes: torch.Size([batch_size, max_len, max_len_nodes])
        :param edges: torch.Size([batch_size, max_len, max_len_edges])
        :param goal: torch.Size([batch_size, max_len_nodes])
        :return result: torch.Size([batch_size, theorem_vocab])
        """
        # [batch_size, max_len, d_model]
        hypertree_encoding = self.encoder(
            self.nodes_emb(nodes.view(-1, self.max_len_nodes), True).view(-1, self.max_len, self.d_model) +
            self.edges_emb(edges.view(-1, self.max_len_edges), True).view(-1, self.max_len, self.d_model)
        )

        # [batch_size, d_model]
        goal_embedding = self.nodes_emb(goal.view(-1, self.max_len_nodes), True).view(-1, self.d_model)

        # [batch_size, d_model]
        decoding = self.decoder(hypertree_encoding, goal_embedding)

        # [batch_size, theorem_vocab]
        output = F.softmax(self.linear(decoding), dim=-1)

        return output


def make_model():
    model = Predictor(
        vocab_nodes=config.vocab_nodes,
        max_len_nodes=config.max_len_nodes,
        h_nodes=config.h_nodes,
        N_encoder_nodes=config.N_encoder_nodes,
        N_decoder_nodes=config.N_decoder_nodes,
        p_drop_nodes=config.p_drop_nodes,
        vocab_edges=config.vocab_edges,
        max_len_edges=config.max_len_edges,
        h_edges=config.h_edges,
        N_encoder_edges=config.N_encoder_edges,
        N_decoder_edges=config.N_decoder_edges,
        p_drop_edges=config.p_drop_edges,
        vocab=config.vocab_theorems,
        max_len=config.max_len,
        h=config.h,
        N=config.N,
        p_drop=config.p_drop,
        d_model=config.d_model
    )
    model.apply(init_weights)
    return model


def train():
    model = make_model()


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
