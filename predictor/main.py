from config import Config, Path
from utils import load_data
from model import TheoremPredictor


def make_model():
    predicate = load_data(Path.predicate2vec)
    sentence = load_data(Path.sentence2vec_a)
    model = TheoremPredictor(predicate2vec=predicate.predicate2vec,
                             sentence2vec=sentence.sentence2vec,
                             d_model=Config.d_model, h=Config.h, N=Config.N,
                             p_drop=Config.p_drop, d_ff=Config.d_ff, vocab_theo=Config.vocab_theo)
    return model


def train():
    pass


def eval_model():
    pass


def main():
    data = load_data(Path.solution_data_path + "2_hyper.pk")
    print(data)


if __name__ == '__main__':
    main()
