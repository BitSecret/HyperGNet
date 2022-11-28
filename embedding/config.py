predicate_word_list = ["Shape", "Collinear", "Point", "Line", "Angle", "Triangle", "RightTriangle", "IsoscelesTriangle",
                       "EquilateralTriangle", "Polygon", "Length", "Measure", "Area", "Perimeter", "Altitude",
                       "Distance", "Midpoint", "Intersect", "Parallel", "DisorderParallel", "Perpendicular",
                       "PerpendicularBisector", "Bisector", "Median", "IsAltitude", "Neutrality", "Circumcenter",
                       "Incenter", "Centroid", "Orthocenter", "Congruent", "Similar", "MirrorCongruent",
                       "MirrorSimilar", "Equation",
                       "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                       "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
                       "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
                       "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65",
                       "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81",
                       "82", "83", "84", "85", "86", "87", "88", "89", "90"]

expr_word_list = ["+", "-", "*", "/", "^", "sin", "cos", "tan", "(", ")",
                  "nums", "ll_", "ma_", "as_", "pt_", "at_", "f_"]  # 表达式词表

sentence_word_list = ["padding", "<start>", "<end>",
                      "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                      "T", "U", "V", "W", "X", "Y", "Z", ",",
                      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                      "t", "u", "v", "w", "x", "y", "z",
                      "+", "-", "*", "/", "^", "sin", "cos", "tan", "(", ")", "nums",
                      "ll_", "ma_", "as_", "pt_", "at_", "f_"]


class Path:
    solution_data = "../reasoner/solution_data/g3k_normal/"


class Config:
    """----------基本配置----------"""
    seed = 3407    # 随机数种子
    version = 1    # 版本

    """----------训练数据生成----------"""
    problem_count = 105    # 已标记的题目总数
    train_ratio = 0.8    # 训练集占比
    walking_depth = 2    # 生成谓词训练数据时，随机游走的深度
    words_max_len = 36    # 个体词的最大长度

    """----------predicate2vec----------"""
    p_vocab = len(predicate_word_list)    # 谓词词表长度
    p_eb_dim = 32    # 嵌入的维度

    """----------sentence2vec----------"""
    s_vocab = len(sentence_word_list)    # 个体词词表长度
    s_eb_dim = 32    # 嵌入的维度
    h = 4    # attention头的数量
    N_encoder = 3    # attention层的数量

    padding_size_a = words_max_len    # plan A  句子最大长度

    padding_size_b = words_max_len + 2    # plan B  句子最大长度 加了起始符
    N_decoder = 3    # plan B  resnet层数
