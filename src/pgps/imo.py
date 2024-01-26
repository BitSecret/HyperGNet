from pgps.agent import get_state, apply_theorem
from pgps.model import make_predictor_model
from formalgeo.tools import get_used_pid_and_theorem
from formalgeo.data import DatasetLoader
from formalgeo.solver import Interactor
from formalgeo.parse import inverse_parse_one_theorem
from func_timeout import func_timeout, FunctionTimedOut
import torch
import time
import warnings


def solve_one(problem_CDL, solver, predictor, device, beam_size, greedy_beam):
    timing = time.time()
    solver.load_problem(problem_CDL)
    if solver.problem.goal.type == "algebra":
        raw_goal = solver.problem.goal.item - solver.problem.goal.answer
        raw_goal = "Equation" + "(" + str(raw_goal).replace(" ", "") + ")"
    else:
        raw_goal = problem_CDL["goal_cdl"].split("(", 1)[1]
        raw_goal = raw_goal[0:len(raw_goal) - 1]

    beam_stacks = [[solver.problem, 1]]  # max_count(beam_stack) = beam_size
    step = 0
    while len(beam_stacks) > 0:
        print(f"step: {step}, beam: {len(beam_stacks)}, timing: {time.time() - timing}")

        for problem, _ in beam_stacks:
            solver.problem = problem
            solver.problem.check_goal()
            if solver.problem.goal.solved:
                _, seqs = get_used_pid_and_theorem(solver.problem)
                seqs = [inverse_parse_one_theorem(s, solver.parsed_theorem_GDL) for s in seqs]
                print(f"{problem_CDL['problem_id']} solved.")
                print(seqs)
                return

        nodes, edges, edges_structural, goal = get_state(beam_stacks, raw_goal)
        predicted_theorems = predictor(
            nodes.to(device),
            edges.to(device),
            edges_structural.to(device),
            goal.to(device)
        ).cpu()
        beam_stacks = apply_theorem(solver, predicted_theorems, beam_stacks, beam_size, greedy_beam, timing, True)

        step += 1

    print(f"{problem_CDL['problem_id']} unsolved.")
    print("Out of stacks.")


def main():
    dataset_name = "formalgeo-imo_v1"
    path_datasets = "datasets"
    path_predictor = "data/trained_model/predictor_model_pretrain.pth"
    timeout = 3600
    beam_size = 3
    greedy_beam = True

    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0")
    predictor = make_predictor_model(torch.load(path_predictor, map_location=torch.device("cpu"))["model"]).to(device)
    dl = DatasetLoader(dataset_name, path_datasets)
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)

    while True:
        pid = int(input("input pid (-1 to stop):"))
        if pid == -1:
            break
        print(f"start solving {pid} ...")

        try:
            func_timeout(timeout, solve_one,
                         args=(dl.get_problem(pid), solver, predictor, device, beam_size, greedy_beam))
        except FunctionTimedOut as e:
            print(f"{pid} unsolved.")
            print(repr(e))


if __name__ == '__main__':
    main()
