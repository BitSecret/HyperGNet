import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Embedding, PositionalEncoding, MultiHeadAttention, LayerNorm, FeedForward
import random
from utils import Configuration as config


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

        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

    def forward(self, x):
        for i in range(self.N):
            # multi-head attention, dropout and layer norm
            x = self.ln_attn[i](self.dp_attn[i](self.attentions[i](x, x, x)))
            # feedforward, dropout, res-net and layer norm
            x = self.ln_ffd[i](x + self.dp_ffd[i](self.feedforwards[i](x)))
        x = torch.mean(x, dim=0)  # pooling
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
        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

    def forward(self, x_encoding, x, mask):
        for i in range(self.N):
            # masked multi-head attention, dropout, resnet and layer norm
            x = self.ln_attn[i](x + self.dp_attn[i](self.attentions[i](x, x, x, mask)))
            # add x embedding
            x = x_encoding + x  # x_encoding will auto expand to dim of x
            # feedforward, dropout、res-net and layer norm
            x = self.ln_ffd[i](x + self.dp_ffd[i](self.feedforwards[i](x)))
        return x


class Sentence2Vector(nn.Module):

    def __init__(self, vocab, d_model, max_len, h, N, p_drop):
        """
        Sentence to Vector, encode sentence with n words to 1 dimension-fixed vector.
        :param vocab: The number of words in the vocabulary list.
        :param d_model: Embedding dim.
        :param max_len: Max length of input sentence.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Sentence2Vector, self).__init__()

        self.embedding = nn.Sequential(
            Embedding(vocab, d_model, padding=True),
            PositionalEncoding(d_model, max_len)
        )
        self.encoder = SentenceEncoder(d_model, h, N, p_drop)
        self.decoder = SentenceDecoder(d_model, h, N, p_drop)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x, use_encoding=False, mask=None):
        x_embedding = self.embedding(x)
        x_encoding = self.encoder(x_embedding)
        if use_encoding:
            return x_encoding

        x_decoding = self.decoder(x_encoding, x, mask)
        output = F.softmax(self.linear(x_decoding), dim=-1)
        return output


def make_model():
    pass


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
