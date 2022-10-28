import os
from utility import load_data, save_data
from gen_data import gen_for_predicate, gen_for_sentence
predicate_word_list = ["Shape", "Collinear", "Point", "Line", "Angle", "Triangle", "RightTriangle", "IsoscelesTriangle",
                       "EquilateralTriangle", "Polygon", "Length", "Measure", "Area", "Perimeter", "Altitude",
                       "Distance", "Midpoint", "Intersect", "Parallel", "DisorderParallel", "Perpendicular",
                       "PerpendicularBisector", "Bisector", "Median", "IsAltitude", "Neutrality", "Circumcenter",
                       "Incenter", "Centroid", "Orthocenter", "Congruent", "Similar", "MirrorCongruent",
                       "MirrorSimilar", "Equation", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                       "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
                       "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
                       "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
                       "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76",
                       "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90"]
sentence_word_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                      "T", "U", "V", "W", "X", "Y", "Z", ",", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                      "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "+", "-", "*", "/",
                      "^", "@", "#", "$", "(", ")", "nums", "ll_", "ma_", "as_", "pt_", "at_", "f_"]
dim = 64    # 谓词、定理和个体词的嵌入向量维度


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
