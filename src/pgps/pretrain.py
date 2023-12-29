import torch
import torch.nn as nn
from module import Embedding, PositionalEncoding, MultiHeadAttention, LayerNorm, FeedForward


class Encoder(nn.Module):
    def __init__(self, vocab, d_model, max_len, h, N, p_drop):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param vocab: The number of words in the vocabulary list.
        :param d_model: Embedding dim.
        :param max_len: Max length of input sentence.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Encoder, self).__init__()

        self.embedding = nn.Sequential(
            Embedding(vocab, d_model, padding=True),
            PositionalEncoding(d_model, max_len)
        )

        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

    def forward(self, sentence):
        sentence = self.embedding(sentence)  # embedding

        for i in range(self.N):
            sentence = self.attentions[i](sentence)  # multi-head attention
            sentence = self.ln_attn[i](self.dp_attn[i](sentence))  # dropout and layer norm
            sentence = self.feedforwards[i](sentence)  # feedforward
            sentence = self.ln_ffd[i](sentence + self.dp_ffd[i](sentence))  # dropout、res-net and layer norm

        sentence = torch.mean(sentence, dim=0)  # pooling

        return sentence


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, h, N, p_drop):
        """
        Sentence decoder, decode sentence embedding to raw sentence.
        :param vocab: The number of words in the vocabulary list.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(Decoder, self).__init__()
        self.attentions = [MultiHeadAttention(h, d_model) for _ in range(N)]
        self.dp_attn = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_attn = [LayerNorm(d_model) for _ in range(N)]

        self.feedforwards = [FeedForward(d_model, d_model * 4) for _ in range(N)]
        self.dp_ffd = [nn.Dropout(p_drop) for _ in range(N)]
        self.ln_ffd = [LayerNorm(d_model) for _ in range(N)]

        self.linear = nn.Linear(d_model, vocab)

    def forward(self, sentence_emb, sentence, mask):
        for i in range(self.N):
            sentence = self.attentions[i](sentence, mask)  # masked multi-head attention
            sentence = self.ln_attn[i](sentence + self.dp_attn[i](sentence))  # dropout and layer norm
            sentence = sentence_emb + sentence  # 这里应该是sentence的每一行加到sentence_emb
            sentence = self.feedforwards[i](sentence)  # feedforward
            sentence = self.ln_ffd[i](sentence + self.dp_ffd[i](sentence))  # dropout、res-net and layer norm

        sentence = self.linear(sentence)  # 这里加不加softmax和激活函数呢

        return sentence


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
        self.encoder = Encoder(vocab, d_model, max_len, h, N, p_drop)
        self.decoder = Decoder(vocab, d_model, h, N, p_drop)

    def forward(self, sentence, mask):
        return self.decoder(self.encoder(sentence), sentence, mask)


def make_model():
    pass