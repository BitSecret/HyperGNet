from gen_data import get_predicate_embedding, get_sentence_embedding
from utility import load_data
from visualize import eval_embedding
solution_data = "../reasoner/solution_data/g3k_normal/"

if __name__ == '__main__':
    # p_e = get_predicate_embedding(solution_data)
    # for key in p_e.keys():
    #     print(key, end=": ")
    #     print(p_e[key])
    #
    # s_e = get_sentence_embedding(solution_data)
    # for key in s_e.keys():
    #     print(key, end=": ")
    #     print(s_e[key])

    # eval_embedding(evaluate_pre=True, dim=3, use_pca=False)

    data = load_data("../reasoner/solution_data/g3k_normal/0_hyper.pk")
    for i in data:
        print(i)

