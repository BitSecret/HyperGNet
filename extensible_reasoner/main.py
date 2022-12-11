from solver import Solver
from aux_tools.utils import load_json, save_json, show, save_step_msg, save_solution_tree


def run(problem_id, save_parse_GDL=False, save_parse_CDL=False):
    predicate_GDL = load_json("./preset/predicate.json")
    theorem_GDL = load_json("./preset/theorem.json")
    problem_CDL = load_json("./preset/theo_test.json")[str(problem_id)]

    solver = Solver(predicate_GDL, theorem_GDL)
    solver.load_problem(problem_CDL)
    for theorem in problem_CDL["theorem_seqs"]:
        solver.apply_theorem(theorem)
    solver.check_goal()
    show(solver.problem, simple=False)

    if save_parse_GDL:
        save_json(solver.predicate_GDL, "./solved/predicate_parsed.json")
        save_json(solver.theorem_GDL, "./solved/theorem_parsed.json")
    if save_parse_CDL:
        save_json(solver.problem.problem_CDL, "./solved/problems/{}_parsed.json".format(problem_id))
        save_step_msg(solver.problem, "./solved/problems/")
        save_solution_tree(solver.problem, "./solved/problems/")


if __name__ == '__main__':
    while True:
        pid = int(input("pid:"))
        run(problem_id=pid, save_parse_GDL=True, save_parse_CDL=True)
