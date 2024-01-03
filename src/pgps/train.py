import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Embedding, MultiHeadAttention, TaskSpecificAttention, LayerNorm, FeedForward
from pretrain import Sentence2Vector
import random
from utils import Configuration as config


class Encoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Encoder, self).__init__()
        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

    def forward(self, x):
        for i in range(self.N):
            # multi-head attention, dropout, res-net and layer norm
            x = self.ln_attn[i](x + self.dp_attn[i](self.attentions[i](x, x, x)))
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

        self.attentions = [TaskSpecificAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

    def forward(self, x, task):
        for i in range(self.N):
            # task-specific attention, dropout, resnet and layer norm
            task = self.ln_attn[i](task + self.dp_attn[i](self.attentions[i](x, task)))
            # feedforward, dropout、res-net and layer norm
            task = self.ln_ffd[i](task + self.dp_ffd[i](self.feedforwards[i](task)))
        return x


class Predictor(nn.Module):

    def __init__(self, nodes_vocab, words_max_len, path_vocab, path_max_len, theorem_vocab,
                 d_model, h, N, p_drop, multi_layer):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param nodes_vocab: The number of words in the nodes.
        :param words_max_len: Max len of words.
        :param path_vocab: The number of words in the path.
        :param path_max_len: Max len of path.
        :param theorem_vocab: The number of theorems.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        :param multi_layer: Weather use multi layer decoder.
        """
        super(Predictor, self).__init__()

        assert d_model % 2 == 0
        self.predicate_emb = Embedding(nodes_vocab, d_model // 2, padding=True)
        self.words_emb = Sentence2Vector(nodes_vocab, d_model // 2, words_max_len, h, N, p_drop)
        self.path_emb = Sentence2Vector(path_vocab, d_model, path_max_len, h, N, p_drop)
        self.goal_predicate_emb = Embedding(nodes_vocab, d_model // 2, padding=True)
        self.goal_words_emb = Sentence2Vector(nodes_vocab, d_model // 2, words_max_len, h, N, p_drop)

        self.encoder = Encoder(d_model, h, N, p_drop)
        if multi_layer:
            self.decoder = Decoder(d_model, h, N, p_drop)
        else:
            self.decoder = TaskSpecificAttention(h, d_model)

        self.linear = nn.Linear(d_model, theorem_vocab)

    def forward(self, predicate, words, path, goal_predicate, goal_words):
        node_embedding = torch.cat((
            self.predicate_emb(predicate),  # predicate embedding
            torch.cat([self.words_emb(words_item, True) for words_item in words], dim=0)),  # words embedding
            dim=1)
        edge_embedding = torch.cat([self.path_emb(path_item, True) for path_item in path], dim=0)
        hypertree_encoding = self.encoder(node_embedding + edge_embedding)

        goal_embedding = torch.cat((
            self.predicate_emb(goal_predicate),
            torch.cat([self.goal_words_emb(words_item, True) for words_item in goal_words], dim=0)),
            dim=1)

        # theorem selection probability
        output = F.softmax(self.linear(self.decoder(hypertree_encoding, goal_embedding)), dim=-1)

        return output


def make_model():
    pass


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
