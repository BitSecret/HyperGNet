import os
from utility import load_data, save_data
from gen_data import gen_for_predicate, gen_for_sentence


def vector_for_predicate(data_path):
    if "predicate.vec" in os.listdir("./output/"):
        return load_data("./output/predicate.vec")

    gen_for_predicate(data_path)
    predicate = load_data("./output/predicate.pk")
    for i in predicate:
        print(i)

    # save_data(predicate, "./output/predicate.vec")


def vector_for_sentence(data_path):
    if "sentence.vec" in os.listdir("./output/"):
        return load_data("./output/sentence.vec")

    gen_for_sentence(data_path)
    sentence = load_data("./output/sentence.pk")
    for i in sentence:
        print(i)

    # save_data(sentence, "./output/sentence.vec")

