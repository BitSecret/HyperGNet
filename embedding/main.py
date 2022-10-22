from gen_data import gen_for_predicate, gen_for_sentence
solution_data = "../GeoMechanical/solution_data/g3k_normal/"


def main():
    for data in gen_for_predicate(solution_data):
        pass
    for step_data in gen_for_sentence(solution_data):
        pass


if __name__ == '__main__':
    main()

