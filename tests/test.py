from pgps.model import *
import formalgeo
import torch


def test_env():
    print("formalgeo.__version__: {}".format(formalgeo.__version__))

    print("torch.__version__: {}".format(torch.__version__))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

    device_count = torch.cuda.device_count()
    print("torch.cuda.device_count(): {}".format(device_count))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    for i in range(device_count):
        print("Device {}: {}".format(i, torch.cuda.get_device_name(i)))


def test_module():
    batch_size = 2
    max_len = 4
    d_model = 6
    vocab = 10
    h = 2

    tensor = torch.randint(low=1, high=vocab, size=(batch_size, max_len))
    structural = torch.randint(low=1, high=max_len, size=(batch_size, max_len))
    tensor[0][3] = 0
    tensor[1][3] = 0
    tensor[1][2] = 0
    structural[0][3] = 0
    structural[1][3] = 0
    structural[1][2] = 0

    print("tensor: {}".format(tensor.shape))
    print(tensor)
    print()

    embedding = Embedding(vocab=vocab, d_model=d_model)
    tensor = embedding(tensor)
    print("embedding(tensor): {}".format(tensor.shape))
    print(tensor)
    print()

    p_encoding = PositionalEncoding(max_len, d_model)
    print("p_encoding.pe: {}".format(p_encoding.pe.shape))
    print(p_encoding.pe)
    print()
    tensor = p_encoding(tensor)
    print("tensor + p_encoding.pe: {}".format(tensor.shape))
    print(tensor)
    print()

    self_attention = SelfAttention(h=h, d_model=d_model)
    mask = torch.tril(torch.ones((max_len, max_len)))
    print("self_attention(tensor, mask):")
    print(self_attention(tensor, mask))
    print()

    task_specific_attention = TaskSpecificAttention(h=h, d_model=d_model)
    task = embedding(torch.randint(low=1, high=vocab, size=(batch_size,)))
    print("task:")
    print(task)
    print()
    print("task_specific_attention(tensor, task):")
    print(task_specific_attention(tensor, task))
    print()

    ln = LayerNorm(d_model=d_model)
    print("ln(tensor):")
    print(ln(tensor))
    print()

    ff = FeedForward(d_model=d_model, d_ff=d_model * 4)
    print("ff(tensor):")
    print(ff(tensor))
    print()


class TestSentenceEncoder(nn.Module):
    def __init__(self, d_model, h, N, p_drop):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(TestSentenceEncoder, self).__init__()

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.dp_attn = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model, d_model * 4) for _ in range(N)])
        self.dp_ffd = nn.ModuleList([nn.Dropout(p_drop) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

    def forward(self, x, choice):
        """
        :param x: torch.Size([batch_size, max_len, d_model])
        :param choice: int
        :return result: torch.Size([batch_size, 1, d_model])
        """
        print("initial x:")
        print(x)
        print()

        if choice == 1:
            print("pure attention:")
            for i in range(len(self.ln_attn)):
                x = self.attentions[i](x)
                print("N={}, ".format(i + 1))
                print(x)
        elif choice == 2:
            print("attention + linear:")
            for i in range(len(self.ln_attn)):
                x = self.attentions[i](x)
                x = self.feedforwards[i](x)
                print("N={}, ".format(i + 1))
                print(x)
        elif choice == 3:
            print("attention + norm:")
            for i in range(len(self.ln_attn)):
                x = self.ln_attn[i](self.attentions[i](x))
                print("N={}, ".format(i + 1))
                print(x)
        elif choice == 4:
            print("attention + linear + norm:")
            for i in range(len(self.ln_attn)):
                x = self.ln_attn[i](self.attentions[i](x))
                x = self.ln_ffd[i](self.feedforwards[i](x))
                print("N={}, ".format(i + 1))
                print(x)
        else:
            print("attention + linear + norm + dropout + resnet:")
            for i in range(len(self.ln_attn)):
                # multi-head attention, dropout, res-net and layer norm
                x = self.ln_attn[i](x + self.dp_attn[i](self.attentions[i](x)))
                # feedforward, dropout, res-net and layer norm
                x = self.ln_ffd[i](x + self.dp_ffd[i](self.feedforwards[i](x)))
                print("N={}, ".format(i + 1))
                print(x)

        x = torch.mean(x, dim=1)  # pooling
        return x


def test_sentence_encoder(choice=1):
    batch_size = 1
    max_len = 15
    d_model = 4
    vocab = 512
    h = 2
    p_drop = 0.2
    N = 5

    embedding = Embedding(vocab, d_model)
    tensor = embedding(torch.randint(low=1, high=vocab, size=(batch_size, max_len)))
    sentence_encoder = TestSentenceEncoder(d_model, h, N, p_drop)
    sentence_encoder(tensor, choice)


def test_pretrain():
    batch_size = 2
    max_len = 4
    max_len_se = 8
    d_model = 6
    vocab = 8
    h = 2
    p_drop = 0.2
    N = 5

    tensor = torch.randint(low=1, high=vocab, size=(batch_size, max_len))
    print("tensor:{}".format(str(tensor.shape)))
    structural = torch.randint(low=1, high=max_len_se, size=(batch_size, max_len))
    print("structural:{}".format(str(structural.shape)))
    mask = torch.tril(torch.ones((max_len, max_len)))
    print("mask:{}".format(str(mask.shape)))
    embedding = Embedding(vocab, d_model)
    tensor_emb = embedding(tensor)
    print("tensor_emb:{}".format(str(tensor_emb.shape)))

    sentence_encoder = SentenceEncoder(d_model, h, N)
    tensor_encoding = sentence_encoder(tensor_emb)
    print("tensor_encoding:{}".format(str(tensor_encoding.shape)))

    sentence_decoder = SentenceDecoder(d_model, h, N, p_drop)
    tensor_decoding = sentence_decoder(tensor_encoding, tensor_emb, mask)
    print("tensor_decoding:{}".format(str(tensor_decoding.shape)))
    print()

    sentence_2_vector = Sentence2Vector(vocab, d_model, max_len, h, N, N, p_drop)
    encoding = sentence_2_vector(x_input=tensor)
    print("encoding (using mode):{}".format(str(encoding.shape)))
    output = sentence_2_vector(x_input=tensor, x_output=tensor, mask=mask)
    print("output (training mode):{}".format(str(output.shape)))
    output = sentence_2_vector(x_encoding=encoding, x_output=tensor, mask=mask)
    print("output (evaluating mode):{}".format(str(output.shape)))
    print()


def test_train():
    batch_size = 2
    max_len = 4
    d_model = 6
    vocab = 8
    h = 2
    p_drop = 0.2
    N = 5

    tensor = torch.randint(low=1, high=vocab, size=(batch_size, max_len))
    print("tensor:{}".format(str(tensor.shape)))
    task = torch.randint(low=1, high=vocab, size=(batch_size,))
    print("task:{}".format(str(task.shape)))
    embedding = Embedding(vocab, d_model)
    tensor_emb = embedding(tensor)
    print("tensor_emb:{}".format(str(tensor_emb.shape)))
    task_emb = embedding(task)
    print("task_emb:{}".format(str(task_emb.shape)))

    encoder = Encoder(d_model, h, N, p_drop)
    encoding = encoder(tensor_emb)
    print("encoding:{}".format(str(encoding.shape)))

    decoder = Decoder(d_model, h, N, p_drop)
    decoding = decoder(tensor_emb, task_emb)
    print("decoding:{}".format(str(decoding.shape)))
    print()

    batch_size = 2
    max_len_nodes = 2
    max_len_edges = 4
    max_len = 6
    max_len_se = 8
    vocab_nodes = 13
    vocab_edges = 15
    vocab_theorems = 17
    d_model = 16
    h = 2
    p_drop = 0.1
    N = 2

    nodes = torch.randint(low=1, high=vocab_nodes, size=(batch_size, max_len, max_len_nodes))
    edges = torch.randint(low=1, high=vocab_edges, size=(batch_size, max_len, max_len_edges))
    edges_structural = torch.randint(low=0, high=max_len_se, size=(batch_size, max_len, max_len_edges))
    goal = torch.randint(low=1, high=vocab_nodes, size=(batch_size, max_len_nodes))
    print("nodes:{}".format(str(nodes.shape)))
    print("edges:{}".format(str(edges.shape)))
    print("edges_structural:{}".format(str(edges_structural.shape)))
    print("goal:{}".format(str(goal.shape)))

    predictor = Predictor(
        vocab_nodes=vocab_nodes,
        max_len_nodes=max_len_nodes,
        h_nodes=h,
        N_encoder_nodes=N,
        N_decoder_nodes=N,
        p_drop_nodes=p_drop,
        vocab_edges=vocab_edges,
        max_len_edges=max_len_edges,
        h_edges=h,
        N_encoder_edges=N,
        N_decoder_edges=N,
        p_drop_edges=p_drop,
        vocab_gs=vocab_edges,
        max_len_gs=max_len_edges,
        h_gs=h,
        N_encoder_gs=N,
        N_decoder_gs=N,
        p_drop_gs=p_drop,
        vocab=vocab_theorems,
        max_len=max_len,
        h=h,
        N=N,
        p_drop=p_drop,
        d_model=d_model
    )

    result = predictor(nodes, edges, edges_structural, goal)
    print("result:{}".format(str(result.shape)))


if __name__ == '__main__':
    test_env()
    # test_module()
    # test_sentence_encoder()
    # test_pretrain()
    # test_train()
