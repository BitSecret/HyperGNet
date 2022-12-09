from solver import Solver
from aux_tools.utils import load_json, save_json, show, save_step_msg, save_solution_tree


def run(problem_id, save_parse_GDL=False, save_parse_CDL=True):
    predicate_GDL = load_json("./preset/predicate.json")
    theorem_GDL = load_json("./preset/theorem.json")
    problem_CDL = load_json("./preset/problems/{}.json".format(problem_id))

    solver = Solver(predicate_GDL, theorem_GDL)
    solver.load_problem(problem_CDL)
    solver.apply_theorem("congruent_property_angle_equal")
    solver.check_goal()
    show(solver.problem)

    if save_parse_GDL:
        save_json(solver.predicate_GDL, "./solved/predicate_parsed.json")
        save_json(solver.theorem_GDL, "./solved/theorem_parsed.json")
    if save_parse_CDL:
        save_json(solver.problem.problem_CDL, "./solved/problems/{}_parsed.json".format(problem_id))
        save_step_msg(solver.problem, "./solved/problems/")
        save_solution_tree(solver.problem, "./solved/problems/")


if __name__ == '__main__':
    run(problem_id=0, save_parse_GDL=False, save_parse_CDL=False)
