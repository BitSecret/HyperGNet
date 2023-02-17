from solver import Solver
from aux_tools.utils import load_json, save_json, show, save_step_msg, save_solution_tree


def run(save_parsed_GDL=False, save_parsed_CDL=False):
    predicate_GDL = load_json("./preset/predicate.json")
    theorem_GDL = load_json("./preset/theorem.json")
    solver = Solver(predicate_GDL, theorem_GDL)
    if save_parsed_GDL:
        save_json(solver.predicate_GDL, "./solved/predicate_parsed.json")
        save_json(solver.theorem_GDL, "./solved/theorem_parsed.json")

    while True:
        pid = int(input("pid:"))
        problem_CDL = load_json("./preset/theorem_test.json")[str(pid)]
        solver.load_problem(problem_CDL)

        if solver.problem.goal["type"] in ["equal", "value"]:
            results = solver.find_prerequisite("Equation", solver.problem.goal["item"])
        else:
            results = solver.find_prerequisite(solver.problem.goal["item"], solver.problem.goal["answer"])
        for r in results:
            print(r)
        print()

        for theorem in problem_CDL["theorem_seqs"]:
            solver.apply_theorem(theorem)
        solver.check_goal()
        show(solver.problem, simple=False)

        if save_parsed_CDL:
            save_json(solver.problem.problem_CDL, "./solved/theorem_test/{}_parsed.json".format(pid))
            save_step_msg(solver.problem, "./solved/theorem_test/")
            save_solution_tree(solver.problem, "./solved/theorem_test/")


if __name__ == '__main__':
    run(save_parsed_GDL=True, save_parsed_CDL=True)
