import torch
import torch.nn as nn
from gps.utils import get_config
from gps.data import nodes_words, edges_words, theorem_words

config = get_config()

"""Model definition."""


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embedding, self).__init__()
        self.d_model_sqrt = d_model ** 0.5  # make the variance of `emb` distribution becomes 1
        self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)  # default padding 0

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        return self.emb(x) * self.d_model_sqrt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, use_structural_encoding):
        """Standard positional encoding from original transformer and structural encoding."""
        max_len = config["data"]["max_len_se"]
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        self.use_structural_encoding = use_structural_encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 'pe' will be retained when model saving and loading, but it will not be updated during the training.
        self.register_buffer('pe', pe)  # torch.Size([max_len, d_model])

    def forward(self, x, x_structure=None):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :param x_structure: structure information, torch.Size([batch_size, seq_len])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        if self.use_structural_encoding:  # structural encoding
            se = self.pe[x_structure.view(-1)]  # [batch_size * seq_len, d_model]
            se = se.view(x.size(0), x.size(1), -1)  # [batch_size, seq_len, d_model]
            se = se.masked_fill(x == 0, 0)  # [batch_size, seq_len, d_model]
            x = x + se
        else:  # positional encoding
            pe = self.pe[:x.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
            pe = pe.masked_fill(x == 0, 0)  # [batch_size, seq_len, d_model]
            x = x + pe
        return x


class SelfAttention(nn.Module):
    def __init__(self, h, d_model):
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.d_k_sqrt = self.d_k ** 0.5
        self.h = h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])  # 4 Linear

    def forward(self, x, use_mask=False):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :param use_mask: Bool
        :return result: torch.Size([batch_size, seq_len, d_model])
        """

        batch_size = x.size(0)

        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear[0:3], [x] * 3)]  # [batch_size, h, seq_len, d_k]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k_sqrt  # [batch_size, h, seq_len, seq_len]

        if use_mask:
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).bool().to(x.device)
            scores = scores.masked_fill(~mask, float('-inf'))
        scores = scores.softmax(dim=-1)

        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [batch_size, seq_len, d_model]
        x = self.linear[-1](x)

        return x


class TaskSpecificAttention(nn.Module):
    def __init__(self, h, d_model):
        super(TaskSpecificAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.d_k_sqrt = self.d_k ** 0.5
        self.h = h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])  # 4 Linear

    def forward(self, x, task):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :param task: torch.Size([batch_size, d_model])
        :return result: torch.Size([batch_size, d_model])
        """
        batch_size = x.size(0)

        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        # [batch_size, h, 1, d_k]
        query = self.linear[0](task).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # [batch_size, h, seq_len, d_k]
        key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                      for linear in self.linear[1:3]]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k_sqrt  # [batch_size, h, 1, seq_len]
        scores = scores.softmax(dim=-1)
        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, self.h * self.d_k)  # [batch_size, d_model]
        x = self.linear[-1](x)

        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fw = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        return self.fw(x)


"""----↑----↑----↑----↑----↑--------Module--------↑----↑----↑----↑----↑----"""
"""----↓----↓----↓----↓----↓---Sentence Encoder---↓----↓----↓----↓----↓----"""


class SentenceEncoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(SentenceEncoder, self).__init__()

        self.dropout = nn.Dropout(p_drop)

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, d_model])
        """

        for i in range(len(self.attentions)):
            x = self.ln_attn[i](x + self.dropout(self.attentions[i](x)))  # multi-head attention and layer norm
            x = self.ln_ffd[i](x + self.dropout(self.feedforwards[i](x)))  # feedforward and layer norm

        x = torch.mean(x, dim=1)  # pooling

        return x


class SentenceDecoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Sentence decoder, decode sentence embedding to raw sentence.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(SentenceDecoder, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.reductions = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x_encoding, y):
        """
        :param x_encoding: torch.Size([batch_size, d_model])
        :param y: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        x_encoding = x_encoding.unsqueeze(1).expand(-1, y.size(1), -1)  # [batch_size, seq_len, d_model]

        for i in range(len(self.attentions)):
            # masked multi-head attention, dropout, resnet and layer norm
            y = self.ln_attn[i](y + self.dropout(self.attentions[i](y, use_mask=True)))

            # concatenate x_encoding and y
            y = self.reductions[i](torch.cat((y, x_encoding), dim=-1))  # [batch_size, seq_len, d_model]

            # feedforward, dropout、res-net and layer norm
            y = self.ln_ffd[i](y + self.dropout(self.feedforwards[i](y)))
        return y


class Sentence2Vector(nn.Module):

    def __init__(self, vocab, h, N, p_drop, d_model, use_structural_encoding):
        """
        Sentence to Vector, encode sentence with n words to 1 dimension-fixed vector.
        :param vocab: The number of words in the vocabulary list.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention in Encoder and Decoder.
        :param p_drop: Dropout rate.
        :param d_model: Embedding dim.
        :param use_structural_encoding: Use structural encoding.
        """
        super(Sentence2Vector, self).__init__()

        self.embedding = Embedding(vocab, d_model)
        self.pe = PositionalEncoding(d_model, use_structural_encoding)
        self.encoder = SentenceEncoder(d_model, h, N, p_drop)
        self.decoder = SentenceDecoder(d_model, h, N, p_drop)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, mode, x=None, x_structure=None, x_encoding=None, y=None):
        """
        Training mode: mode='train'; need x, x_structure; return x_decoding.
        Testing mode: mode='test'; need x_encoding, x_structure, y; return x_decoding.
        Encoding mode: mode='encode'; need x, x_structure; return x_encoding.
        :param mode: 'train', 'test' or 'encode'
        :param x: torch.Size([batch_size, seq_len])
        :param x_structure: torch.Size([batch_size, seq_len])
        :param x_encoding: None
        :param y: torch.Size([batch_size, seq_len])
        :return x_encoding: torch.Size([batch_size, d_model])
        :return x_decoding: torch.Size([batch_size, seq_len, vocab])
        """
        if mode == "train":
            x_embedding = self.pe(self.embedding(x), x_structure)
            x_encoding = self.encoder(x_embedding)
            x_decoding = self.linear(self.decoder(x_encoding, x_embedding))
            return x_decoding
        elif mode == "test":
            y_embedding = self.pe(self.embedding(y), x_structure)
            x_decoding = self.linear(self.decoder(x_encoding, y_embedding))
            return x_decoding
        elif mode == "encode":
            x_embedding = self.pe(self.embedding(x), x_structure)
            x_encoding = self.encoder(x_embedding)
            return x_encoding
        else:
            raise ValueError(f"unknown mode {mode}")


"""----↑----↑----↑----↑----↑----Sentence Encoder-----↑----↑----↑----↑----↑----"""
"""----↓----↓----↓----↓----↓----Theorem Predictor----↓----↓----↓----↓----↓----"""


class GraphEncoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Graph encoder, encode hypergraph.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(GraphEncoder, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, node_len, d_model])
        :return result: torch.Size([batch_size, node_len, d_model])
        """
        for i in range(len(self.attentions)):
            # multi-head attention, dropout, res-net and layer norm
            x = self.ln_attn[i](x + self.dropout(self.attentions[i](x)))
            # feedforward, dropout, res-net and layer norm
            x = self.ln_ffd[i](x + self.dropout(self.feedforwards[i](x)))
        return x


class GraphDecoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Graph decoder, decode graph encoding according to specific task.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(GraphDecoder, self).__init__()
        self.dropout = nn.Dropout(p_drop)

        self.attentions = nn.ModuleList([TaskSpecificAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x, task):
        """
        :param x: torch.Size([batch_size, node_len, d_model])
        :param task: torch.Size([batch_size, d_model])
        :return result: torch.Size([batch_size, d_model])
        """
        for i in range(len(self.attentions)):
            # task-specific attention, dropout, resnet and layer norm
            task = self.ln_attn[i](task + self.dropout(self.attentions[i](x, task)))
            # feedforward, dropout、res-net and layer norm
            task = self.ln_ffd[i](task + self.dropout(self.feedforwards[i](task)))
        return task


class Predictor(nn.Module):

    def __init__(self, vocab_nodes, vocab_edges, vocab_theorems,
                 h_encoder, N_encoder, p_drop_encoder,
                 h_predictor, N_predictor, p_drop_predictor,
                 d_model, use_structural_encoding, use_hypertree):
        """Theorem predictor."""
        super(Predictor, self).__init__()
        self.d_model = d_model
        self.use_hypertree = use_hypertree

        self.nodes_emb = Sentence2Vector(
            vocab_nodes, h_encoder, N_encoder, p_drop_encoder, d_model, use_structural_encoding=False
        )
        self.edges_emb = Sentence2Vector(
            vocab_edges, h_encoder, N_encoder, p_drop_encoder, d_model, use_structural_encoding=use_structural_encoding
        )
        self.reduction = nn.Linear(d_model * 2, d_model)

        self.pe = PositionalEncoding(d_model, use_structural_encoding=False)
        self.encoder = GraphEncoder(d_model, h_predictor, N_predictor, p_drop_predictor)
        self.decoder = GraphDecoder(d_model, h_predictor, N_predictor, p_drop_predictor)  # task specific decoder

        self.linear = nn.Linear(d_model, vocab_theorems)

    def forward(self, nodes, edges, structures, goals):
        """
        :param nodes: torch.Size([batch_size, node_len, nodes_seq_len])
        :param edges: torch.Size([batch_size, node_len, edges_seq_len])
        :param structures: torch.Size([batch_size, node_len, edges_seq_len])
        :param goals: torch.Size([batch_size, goal_seq_len])
        :return result: torch.Size([batch_size, theorem_vocab])
        """
        batch_size, node_len, nodes_seq_len = nodes.size()
        _, _, edges_seq_len = edges.size()

        node_encoding = self.nodes_emb(
            mode="encode",
            x=nodes.view(-1, nodes_seq_len)
        )  # torch.Size([batch_size * node_len, self.d_model])

        if self.use_hypertree:
            edge_encoding = self.edges_emb(
                mode="encode",
                x=edges.view(-1, edges_seq_len),
                x_structure=structures.view(-1, edges_seq_len)
            )  # torch.Size([batch_size * node_len, self.d_model])
        else:
            edge_encoding = torch.zeros(batch_size * node_len, self.d_model).to(next(self.parameters()).device)

        goal_encoding = self.nodes_emb(
            mode="encode",
            x=goals
        )  # [batch_size, d_model]

        hypergraph_encoding = self.reduction(torch.cat((node_encoding, edge_encoding), dim=-1))
        hypergraph_encoding = self.pe(hypergraph_encoding.view(batch_size, node_len, self.d_model))
        hypergraph_encoding = self.encoder(hypergraph_encoding)
        hypergraph_decoding = self.decoder(x=hypergraph_encoding, task=goal_encoding)  # [batch_size, d_model]

        predicted_theorems = self.linear(hypergraph_decoding)  # [batch_size, theorem_vocab]
        return predicted_theorems


def make_model(use_structural_encoding, use_hypertree):
    model = Predictor(vocab_nodes=len(nodes_words),
                      vocab_edges=len(edges_words),
                      vocab_theorems=len(theorem_words),
                      h_encoder=config["encoder"]["model"]["h"],
                      N_encoder=config["encoder"]["model"]["N"],
                      p_drop_encoder=config["encoder"]["model"]["p_drop"],
                      h_predictor=config["predictor"]["model"]["h"],
                      N_predictor=config["predictor"]["model"]["N"],
                      p_drop_predictor=config["predictor"]["model"]["p_drop"],
                      d_model=config["d_model"],
                      use_structural_encoding=use_structural_encoding,
                      use_hypertree=use_hypertree)

    return model


def show_parameters():
    """
    Params: 20,380,297 (20.38 M)
    Memory: 77.74 MB
    """
    m = make_model(True, True)
    total_params = sum(p.numel() for p in m.parameters())  # 参数总数
    param_memory = sum(p.numel() * p.element_size() for p in m.parameters())  # 占用字节数
    print("Params: {} ({:.2f} M), Memory: {:.2f} MB.".format(total_params,
                                                             total_params / 1000000,
                                                             param_memory / 1024 / 1024))


if __name__ == '__main__':
    show_parameters()
