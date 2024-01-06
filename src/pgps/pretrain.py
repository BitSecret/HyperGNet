from pgps.utils import Configuration as config
from pgps.module import Embedding, PositionalEncoding, SelfAttention, LayerNorm, FeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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
            Embedding(vocab, d_model),
            PositionalEncoding(max_len, d_model)
        )
        self.encoder = SentenceEncoder(d_model, h, N)
        self.decoder = SentenceDecoder(d_model, h, N, p_drop)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x, use_encoding=False, mask=None):
        """
        :param x: torch.Size([batch_size, max_len])
        :param use_encoding: set True when use model, set False when train and test model.
        :param mask: torch.Size([max_len, max_len]), only useful when use_encoding=False.
        :return x_encoding: torch.Size([batch_size, 1, d_model])
        :return output: torch.Size([batch_size, max_len, d_model])
        """
        x_embedding = self.embedding(x)

        x_encoding = self.encoder(x_embedding)
        if use_encoding:
            return x_encoding

        x_decoding = self.decoder(x_encoding, x_embedding, mask)
        output = F.softmax(self.linear(x_decoding), dim=-1)

        return output


def make_words_model():
    model = Sentence2Vector(
        vocab=config.vocab_words,
        d_model=config.d_model // 2,
        max_len=config.max_len_words,
        h=config.h_words,
        N=config.N_words,
        p_drop=config.p_drop_words
    )
    return model


def make_path_model():
    model = Sentence2Vector(
        vocab=config.vocab_path,
        d_model=config.d_model,
        max_len=config.max_len_path,
        h=config.h_path,
        N=config.N_path,
        p_drop=config.p_drop_words
    )
    return model


if __name__ == '__main__':
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
