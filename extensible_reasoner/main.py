from solver import Solver
from aux_tools.utils import load_json, save_json
predicate_GDL = load_json("./preset/predicate.json")
theorem_GDL = load_json("./preset/theorem.json")
problem_CDL = load_json("./preset/problems/0.json")

s = Solver(predicate_GDL, theorem_GDL)
s.load_problem(problem_CDL)

# save_json(s.predicate, "./preset/predicate_parsed.json")
# save_json(s.theorem, "./preset/theorem_parsed.json")
# save_json(s.problem.msg, "./preset/problems/0_parsed.json")
