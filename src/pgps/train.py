from pgps.utils import Configuration as config
from pgps.module import Embedding, SelfAttention, TaskSpecificAttention, LayerNorm, FeedForward
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

    def __init__(self, nodes_vocab, max_len_words, path_vocab, max_len_path, theorem_vocab,
                 max_len, d_model, h, N, p_drop):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param nodes_vocab: The number of words in the nodes.
        :param max_len_words: Max len of words.
        :param path_vocab: The number of words in the path.
        :param max_len_path: Max len of path.
        :param theorem_vocab: The number of theorems.
        :param max_len: Max nodes number.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Predictor, self).__init__()
        self.max_len_words = max_len_words
        self.max_len_path = max_len_path
        self.max_len = max_len
        self.d_model = d_model

        assert d_model % 2 == 0
        self.predicate_emb = Embedding(nodes_vocab, d_model // 2)
        self.words_emb = Sentence2Vector(nodes_vocab, d_model // 2, max_len_words, h, N, p_drop)
        self.path_emb = Sentence2Vector(path_vocab, d_model, max_len_path, h, N, p_drop)

        self.encoder = Encoder(d_model, h, N, p_drop)
        self.decoder = Decoder(d_model, h, N, p_drop)  # self.decoder = TaskSpecificAttention(h, d_model)

        self.linear = nn.Linear(d_model, theorem_vocab)

    def forward(self, predicate, words, path, goal_predicate, goal_words):
        """
        :param predicate: torch.Size([batch_size, max_len])
        :param words: torch.Size([batch_size, max_len, max_len_words])
        :param path: torch.Size([batch_size, max_len, max_len_path])
        :param goal_predicate: torch.Size([batch_size])
        :param goal_words: torch.Size([batch_size, max_len_words])
        :return result: torch.Size([batch_size, theorem_vocab])
        """

        predicate_embedding = self.predicate_emb(predicate)  # [batch_size, max_len, d_model/2]
        words_embedding = (self.words_emb(words.view(-1, self.max_len_words), True)  # [batch_size, max_len, d_model/2]
                           .view(-1, self.max_len, self.d_model // 2))
        node_embedding = torch.cat((predicate_embedding, words_embedding), dim=2)  # [batch_size, max_len, d_model]
        edge_embedding = (self.path_emb(path.view(-1, self.max_len_path), True)  # [batch_size, max_len, d_model]
                          .view(-1, self.max_len, self.d_model))
        hypertree_encoding = self.encoder(node_embedding + edge_embedding)  # [batch_size, max_len, d_model]

        goal_predicate_embedding = (self.predicate_emb(goal_predicate)  # [batch_size, d_model/2]
                                    .view(-1, self.d_model // 2))
        goal_words_embedding = (self.words_emb(goal_words.view(-1, self.max_len_words), True)  # [batch_size, d_model/2]
                                .view(-1, self.d_model // 2))
        goal_embedding = torch.cat((goal_predicate_embedding, goal_words_embedding), dim=1)  # [batch_size, d_model]

        decoding = self.decoder(hypertree_encoding, goal_embedding)  # [batch_size, d_model]

        return F.softmax(self.linear(decoding), dim=-1)


def make_model():
    model = Predictor(
        nodes_vocab=config.vocab_words,
        max_len_words=config.max_len_words,
        path_vocab=config.vocab_path,
        max_len_path=config.max_len_path,
        theorem_vocab=config.vocab_theorems,
        max_len=config.max_len_predictor,
        d_model=config.d_model,
        h=config.h_predictor,
        N=config.N_predictor,
        p_drop=config.p_drop_predictor
    )
    return model


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
