from pgps.module import *
import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):
    def __init__(self, d_model, h, N):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        """
        super(SentenceEncoder, self).__init__()

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :return result: torch.Size([batch_size, d_model])
        """
        for i in range(len(self.ln_attn)):
            # multi-head attention and layer norm
            x = self.ln_attn[i](self.attentions[i](x))
            # feedforward and layer norm
            x = self.ln_ffd[i](self.feedforwards[i](x))

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
        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.dp_attn = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.dp_ffd = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x_encoding, x, mask):
        """
        :param x_encoding: torch.Size([batch_size, d_model])
        :param x: torch.Size([batch_size, max_len, d_model])
        :param mask: torch.Size([max_len, max_len])
        :return result: torch.Size([batch_size, max_len, d_model])
        """
        x_encoding = x_encoding.unsqueeze(1)  # [batch_size, 1, d_model]
        for i in range(len(self.attentions)):
            # masked multi-head attention, dropout, resnet and layer norm
            x = self.ln_attn[i](x + self.dp_attn[i](self.attentions[i](x, mask)))
            # add x embedding
            x = x_encoding + x  # x_encoding will auto expand to dim of x
            # feedforward, dropout、res-net and layer norm
            x = self.ln_ffd[i](x + self.dp_ffd[i](self.feedforwards[i](x)))
        return x


class Nodes2Vector(nn.Module):

    def __init__(self, vocab, d_model, max_len, h, N_encoder, N_decoder, p_drop):
        """
        Sentence to Vector, encode sentence with n words to 1 dimension-fixed vector.
        :param vocab: The number of words in the vocabulary list.
        :param d_model: Embedding dim.
        :param max_len: Max length of input sentence.
        :param h: Head number in MultiHeadAttention.
        :param N_encoder: Number of MultiHeadAttention in Encoder.
        :param N_decoder: Number of MultiHeadAttention in Decoder.
        :param p_drop: Dropout rate.
        """
        super(Nodes2Vector, self).__init__()

        self.embedding = Embedding(vocab, d_model)
        self.pe = PositionalEncoding(max_len, d_model)
        self.encoder = SentenceEncoder(d_model, h, N_encoder)
        self.decoder = SentenceDecoder(d_model, h, N_decoder, p_drop)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x_input=None, x_encoding=None, x_output=None, mask=None):
        """
        Training mode: x_input=x, x_encoding=None, x_output=x, mask=tril_mask
        Evaluating mode: x_input=None, x_encoding=x_encoding, x_output=output, mask=tril_mask
        Using mode: x_input=x, x_encoding=None, x_output=None, mask=None
        :param x_input: torch.Size([batch_size, max_len])
        :param x_encoding: torch.Size([batch_size, max_len, d_model])
        :param x_output: torch.Size([batch_size, max_len])
        :param mask: torch.Size([max_len, max_len])
        :return x_encoding: torch.Size([batch_size, 1, d_model])
        :return output: torch.Size([batch_size, max_len, vocab])
        """
        if x_input is None:  # evaluating mode
            output_embedding = self.pe(self.embedding(x_output))
            x_decoding = self.decoder(x_encoding, output_embedding, mask)
            return self.linear(x_decoding)
        elif x_output is None:  # using mode
            return self.encoder(self.pe(self.embedding(x_input)))
        else:  # training mode
            x_embedding = self.pe(self.embedding(x_input))
            x_encoding = self.encoder(x_embedding)
            x_decoding = self.decoder(x_encoding, x_embedding, mask)
            return self.linear(x_decoding)


class Edges2Vector(nn.Module):

    def __init__(self, vocab, d_model, max_len, max_len_se, h, N_encoder, N_decoder, p_drop):
        """
        Sentence to Vector, encode sentence with n words to 1 dimension-fixed vector.
        :param vocab: The number of words in the vocabulary list.
        :param d_model: Embedding dim.
        :param max_len: Max length of input sentence.
        :param max_len_se: Max index of structural msg.
        :param h: Head number in MultiHeadAttention.
        :param N_encoder: Number of MultiHeadAttention in Encoder.
        :param N_decoder: Number of MultiHeadAttention in Decoder.
        :param p_drop: Dropout rate.
        """
        super(Edges2Vector, self).__init__()

        self.embedding = Embedding(vocab, d_model)
        self.pe = PositionalEncoding(max_len, d_model)
        self.se = StructuralEncoding(max_len_se, d_model)
        self.encoder = SentenceEncoder(d_model, h, N_encoder)
        self.decoder = SentenceDecoder(d_model, h, N_decoder, p_drop)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x_input=None, x_structural=None, x_encoding=None, x_output=None, mask=None):
        """
        Training mode: x_input=x, x_encoding=None, x_output=x, mask=tril_mask
        Evaluating mode: x_input=None, x_encoding=x_encoding, x_output=output, mask=tril_mask
        Using mode: x_input=x, x_encoding=None, x_output=None, mask=None
        :param x_input: torch.Size([batch_size, max_len])
        :param x_structural: torch.Size([batch_size, max_len])
        :param x_encoding: torch.Size([batch_size, max_len, d_model])
        :param x_output: torch.Size([batch_size, max_len])
        :param mask: torch.Size([max_len, max_len])
        :return x_encoding: torch.Size([batch_size, 1, d_model])
        :return output: torch.Size([batch_size, max_len, vocab])
        """
        if x_input is None:  # evaluating mode
            output_embedding = self.pe(self.embedding(x_output))
            x_decoding = self.decoder(x_encoding, output_embedding, mask)
            return self.linear(x_decoding)
        elif x_output is None:  # using mode
            x_embedding = self.pe(self.embedding(x_input))
            if x_structural is not None:
                x_embedding = self.se(x_embedding, x_structural)
            return self.encoder(x_embedding)
        else:  # training mode
            x_embedding = self.pe(self.embedding(x_input))
            if x_structural is not None:
                x_embedding = self.se(x_embedding, x_structural)
            x_encoding = self.encoder(x_embedding)
            x_decoding = self.decoder(x_encoding, x_embedding, mask)
            return self.linear(x_decoding)


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
                 vocab_edges, max_len_edges, max_len_se, h_edges, N_encoder_edges, N_decoder_edges, p_drop_edges,
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

        self.nodes_emb = Nodes2Vector(
            vocab_nodes, d_model, max_len_nodes, h_nodes, N_encoder_nodes, N_decoder_nodes, p_drop_nodes)
        self.edges_emb = Edges2Vector(
            vocab_edges, d_model, max_len_edges, max_len_se, h_edges, N_encoder_edges, N_decoder_edges, p_drop_edges
        )

        self.encoder = Encoder(d_model, h, N, p_drop)
        self.decoder = Decoder(d_model, h, N, p_drop)  # self.decoder = TaskSpecificAttention(h, d_model)

        self.linear = nn.Linear(d_model, vocab)

    def forward(self, nodes, edges, edges_structural, goal):
        """
        :param nodes: torch.Size([batch_size, max_len, max_len_nodes])
        :param edges: torch.Size([batch_size, max_len, max_len_edges])
        :param edges_structural: torch.Size([batch_size, max_len, max_len_edges])
        :param goal: torch.Size([batch_size, max_len_nodes])
        :return result: torch.Size([batch_size, vocab])
        """
        # [batch_size, max_len, d_model]
        hypertree_encoding = self.encoder(
            self.nodes_emb(
                x_input=nodes.view(-1, self.max_len_nodes)
            ).view(-1, self.max_len, self.d_model) +
            self.edges_emb(
                x_input=edges.view(-1, self.max_len_edges),
                x_structural=edges_structural.view(-1, self.max_len_edges)
            ).view(-1, self.max_len, self.d_model)
        )

        # [batch_size, d_model]
        goal_embedding = self.nodes_emb(
            x_input=goal.view(-1, self.max_len_nodes)
        ).view(-1, self.d_model)

        # [batch_size, d_model]
        decoding = self.decoder(
            x=hypertree_encoding,
            task=goal_embedding
        )

        return self.linear(decoding)  # [batch_size, theorem_vocab]


def make_nodes_model(model_filename=None):
    model = Nodes2Vector(
        vocab=config.vocab_nodes,
        d_model=config.d_model,
        max_len=config.max_len_nodes,
        h=config.h_nodes,
        N_encoder=config.N_encoder_nodes,
        N_decoder=config.N_decoder_nodes,
        p_drop=config.p_drop_nodes
    )
    if model_filename is None:
        model.apply(init_weights)
    else:
        model.load_state_dict(torch.load(model_filename))
    return model


def make_edges_model(model_filename=None):
    model = Edges2Vector(
        vocab=config.vocab_edges,
        d_model=config.d_model,
        max_len=config.max_len_edges,
        max_len_se=config.max_len_edges_se,
        h=config.h_edges,
        N_encoder=config.N_encoder_edges,
        N_decoder=config.N_decoder_edges,
        p_drop=config.p_drop_nodes
    )
    model.apply(init_weights)
    if model_filename is None:
        model.apply(init_weights)
    else:
        model.load_state_dict(torch.load(model_filename))
    return model


def make_predictor_model(model_filename=None):
    model = Predictor(
        vocab_nodes=config.vocab_nodes,
        max_len_nodes=config.max_len_nodes,
        h_nodes=config.h_nodes,
        N_encoder_nodes=config.N_encoder_nodes,
        N_decoder_nodes=config.N_decoder_nodes,
        p_drop_nodes=config.p_drop_nodes,
        vocab_edges=config.vocab_edges,
        max_len_edges=config.max_len_edges,
        max_len_se=config.max_len_edges_se,
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
    if model_filename is None:
        model.apply(init_weights)
    else:
        model.load_state_dict(torch.load(model_filename))
    return model


def check_model_parameters():
    """
    Nodes model: 4806288 (18.33 MB)
    Edges model: 19165441 (73.11 MB)
    Predictor model: 76147852 (290.48 MB)
    """
    count = sum(p.numel() for p in make_nodes_model().parameters())
    print("Nodes model: {} ({:.2f} MB)".format(count, count * 4 / 1024 / 1024))
    count = sum(p.numel() for p in make_edges_model().parameters())
    print("Edges model: {} ({:.2f} MB)".format(count, count * 4 / 1024 / 1024))
    count = sum(p.numel() for p in make_predictor_model().parameters())
    print("Predictor model: {} ({:.2f} MB)".format(count, count * 4 / 1024 / 1024))


if __name__ == '__main__':
    check_model_parameters()
