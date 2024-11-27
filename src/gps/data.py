from formalgeo.tools import get_meta_hypertree, load_json, safe_save_json, save_json
from formalgeo.solver import Interactor
from formalgeo.parse import parse_one_theorem, inverse_parse_one
from formalgeo.data import DatasetLoader
from gps.utils import load_pickle, save_pickle, get_config
import os
import re
import random
from func_timeout import func_timeout, FunctionTimedOut
import warnings
from multiprocessing import Process, Queue
import psutil
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset
import torch

config = get_config()
random.seed(config["random_seed"])

"""--------------Training data generation using FormalGeo symbolic system--------------"""

predicate_words = [
    'Equation', 'Shape', 'Collinear', 'Cocircular', 'Point', 'Line', 'Arc', 'Angle', 'Polygon', 'Circle',
    'RightTriangle', 'IsoscelesTriangle', 'IsoscelesRightTriangle', 'EquilateralTriangle', 'Kite',
    'Parallelogram', 'Rhombus', 'Rectangle', 'Square', 'Trapezoid', 'IsoscelesTrapezoid',
    'RightTrapezoid', 'IsMidpointOfLine', 'IsMidpointOfArc', 'ParallelBetweenLine',
    'PerpendicularBetweenLine', 'IsPerpendicularBisectorOfLine', 'IsBisectorOfAngle',
    'IsMedianOfTriangle', 'IsAltitudeOfTriangle', 'IsMidsegmentOfTriangle', 'IsCircumcenterOfTriangle',
    'IsIncenterOfTriangle', 'IsCentroidOfTriangle', 'IsOrthocenterOfTriangle',
    'CongruentBetweenTriangle', 'MirrorCongruentBetweenTriangle', 'SimilarBetweenTriangle',
    'MirrorSimilarBetweenTriangle', 'IsAltitudeOfQuadrilateral', 'IsMidsegmentOfQuadrilateral',
    'IsCircumcenterOfQuadrilateral', 'IsIncenterOfQuadrilateral', 'CongruentBetweenQuadrilateral',
    'MirrorCongruentBetweenQuadrilateral', 'SimilarBetweenQuadrilateral',
    'MirrorSimilarBetweenQuadrilateral', 'CongruentBetweenArc', 'SimilarBetweenArc',
    'IsDiameterOfCircle', 'IsTangentOfCircle', 'IsCentreOfCircle'
]
theorem_words = [
    'line_addition_1', 'midpoint_of_line_judgment_1', 'parallel_judgment_corresponding_angle_1',
    'parallel_judgment_corresponding_angle_2', 'parallel_judgment_alternate_interior_angle_1',
    'parallel_judgment_alternate_interior_angle_2', 'parallel_judgment_ipsilateral_internal_angle_1',
    'parallel_judgment_par_par_1', 'parallel_judgment_per_per_1', 'parallel_judgment_per_per_2',
    'parallel_property_collinear_extend_1', 'parallel_property_collinear_extend_2',
    'parallel_property_collinear_extend_3', 'parallel_property_corresponding_angle_1',
    'parallel_property_corresponding_angle_2', 'parallel_property_alternate_interior_angle_1',
    'parallel_property_alternate_interior_angle_2', 'parallel_property_ipsilateral_internal_angle_1',
    'parallel_property_par_per_1', 'parallel_property_par_per_2', 'perpendicular_judgment_angle_1',
    'perpendicular_bisector_judgment_per_and_mid_1', 'perpendicular_bisector_judgment_distance_equal_1',
    'perpendicular_bisector_property_distance_equal_1', 'perpendicular_bisector_property_bisector_1',
    'angle_addition_1', 'flat_angle_1', 'adjacent_complementary_angle_1', 'round_angle_1', 'vertical_angle_1',
    'bisector_of_angle_judgment_angle_equal_1', 'bisector_of_angle_property_distance_equal_1',
    'bisector_of_angle_property_line_ratio_1', 'bisector_of_angle_property_length_formula_1',
    'triangle_property_angle_sum_1', 'sine_theorem_1', 'cosine_theorem_1', 'triangle_perimeter_formula_1',
    'triangle_area_formula_common_1', 'triangle_area_formula_sine_1', 'median_of_triangle_judgment_1',
    'altitude_of_triangle_judgment_1', 'altitude_of_triangle_judgment_2', 'altitude_of_triangle_judgment_3',
    'midsegment_of_triangle_judgment_midpoint_1', 'midsegment_of_triangle_judgment_parallel_1',
    'midsegment_of_triangle_judgment_parallel_2', 'midsegment_of_triangle_judgment_parallel_3',
    'midsegment_of_triangle_property_parallel_1', 'midsegment_of_triangle_property_length_1',
    'circumcenter_of_triangle_judgment_intersection_1', 'circumcenter_of_triangle_property_intersection_1',
    'circumcenter_of_triangle_property_intersection_2', 'incenter_of_triangle_judgment_intersection_1',
    'centroid_of_triangle_judgment_intersection_1', 'centroid_of_triangle_property_intersection_1',
    'centroid_of_triangle_property_line_ratio_1', 'orthocenter_of_triangle_judgment_intersection_1',
    'orthocenter_of_triangle_property_intersection_1', 'orthocenter_of_triangle_property_angle_1',
    'congruent_triangle_judgment_sss_1', 'congruent_triangle_judgment_sas_1', 'congruent_triangle_judgment_aas_1',
    'congruent_triangle_judgment_aas_2', 'congruent_triangle_judgment_aas_3', 'congruent_triangle_judgment_hl_1',
    'congruent_triangle_judgment_hl_2', 'congruent_triangle_property_line_equal_1',
    'congruent_triangle_property_angle_equal_1', 'congruent_triangle_property_perimeter_equal_1',
    'congruent_triangle_property_area_equal_1', 'congruent_triangle_property_exchange_1',
    'mirror_congruent_triangle_judgment_sss_1', 'mirror_congruent_triangle_judgment_sas_1',
    'mirror_congruent_triangle_judgment_aas_1', 'mirror_congruent_triangle_judgment_aas_2',
    'mirror_congruent_triangle_judgment_aas_3', 'mirror_congruent_triangle_judgment_hl_1',
    'mirror_congruent_triangle_judgment_hl_2', 'mirror_congruent_triangle_property_line_equal_1',
    'mirror_congruent_triangle_property_angle_equal_1', 'mirror_congruent_triangle_property_perimeter_equal_1',
    'mirror_congruent_triangle_property_area_equal_1', 'mirror_congruent_triangle_property_exchange_1',
    'similar_triangle_judgment_sss_1', 'similar_triangle_judgment_sas_1', 'similar_triangle_judgment_aa_1',
    'similar_triangle_judgment_hl_1', 'similar_triangle_judgment_hl_2', 'similar_triangle_property_ratio_1',
    'similar_triangle_property_line_ratio_1', 'similar_triangle_property_angle_equal_1',
    'similar_triangle_property_perimeter_ratio_1', 'similar_triangle_property_area_square_ratio_1',
    'mirror_similar_triangle_judgment_sss_1', 'mirror_similar_triangle_judgment_sas_1',
    'mirror_similar_triangle_judgment_aa_1', 'mirror_similar_triangle_judgment_hl_1',
    'mirror_similar_triangle_judgment_hl_2', 'mirror_similar_triangle_property_ratio_1',
    'mirror_similar_triangle_property_line_ratio_1', 'mirror_similar_triangle_property_angle_equal_1',
    'mirror_similar_triangle_property_perimeter_ratio_1', 'mirror_similar_triangle_property_area_square_ratio_1',
    'right_triangle_judgment_angle_1', 'right_triangle_judgment_pythagorean_inverse_1',
    'right_triangle_property_pythagorean_1', 'right_triangle_property_length_of_median_1',
    'isosceles_triangle_judgment_line_equal_1', 'isosceles_triangle_judgment_angle_equal_1',
    'isosceles_triangle_property_angle_equal_1', 'isosceles_triangle_property_line_coincidence_1',
    'isosceles_triangle_property_line_coincidence_2', 'isosceles_triangle_property_line_coincidence_3',
    'isosceles_right_triangle_judgment_isosceles_and_right_1', 'isosceles_right_triangle_property_angle_1',
    'equilateral_triangle_judgment_isosceles_and_isosceles_1', 'equilateral_triangle_property_angle_1',
    'quadrilateral_property_angle_sum_1', 'quadrilateral_perimeter_formula_1', 'altitude_of_quadrilateral_judgment_1',
    'altitude_of_quadrilateral_judgment_2', 'altitude_of_quadrilateral_judgment_3',
    'altitude_of_quadrilateral_judgment_4', 'altitude_of_quadrilateral_judgment_5',
    'altitude_of_quadrilateral_judgment_6', 'altitude_of_quadrilateral_judgment_left_vertex_1',
    'altitude_of_quadrilateral_judgment_left_vertex_2', 'altitude_of_quadrilateral_judgment_left_vertex_3',
    'altitude_of_quadrilateral_judgment_left_vertex_4', 'altitude_of_quadrilateral_judgment_left_vertex_5',
    'altitude_of_quadrilateral_judgment_left_vertex_6', 'altitude_of_quadrilateral_judgment_right_vertex_1',
    'altitude_of_quadrilateral_judgment_right_vertex_2', 'altitude_of_quadrilateral_judgment_right_vertex_3',
    'altitude_of_quadrilateral_judgment_right_vertex_4', 'altitude_of_quadrilateral_judgment_right_vertex_5',
    'altitude_of_quadrilateral_judgment_right_vertex_6', 'altitude_of_quadrilateral_judgment_diagonal_1',
    'altitude_of_quadrilateral_judgment_diagonal_2', 'altitude_of_quadrilateral_judgment_diagonal_3',
    'altitude_of_quadrilateral_judgment_diagonal_4', 'midsegment_of_quadrilateral_judgment_midpoint_1',
    'midsegment_of_quadrilateral_judgment_parallel_1', 'midsegment_of_quadrilateral_judgment_parallel_2',
    'midsegment_of_quadrilateral_judgment_parallel_3', 'midsegment_of_quadrilateral_judgment_parallel_4',
    'midsegment_of_quadrilateral_judgment_parallel_5', 'midsegment_of_quadrilateral_judgment_parallel_6',
    'midsegment_of_quadrilateral_property_length_1', 'midsegment_of_quadrilateral_property_parallel_1',
    'midsegment_of_quadrilateral_property_parallel_2', 'circumcenter_of_quadrilateral_property_intersection_1',
    'circumcenter_of_quadrilateral_property_intersection_2', 'congruent_quadrilateral_property_line_equal_1',
    'congruent_quadrilateral_property_angle_equal_1', 'congruent_quadrilateral_property_perimeter_equal_1',
    'congruent_quadrilateral_property_area_equal_1', 'congruent_quadrilateral_property_exchange_1',
    'mirror_congruent_quadrilateral_property_line_equal_1', 'mirror_congruent_quadrilateral_property_angle_equal_1',
    'mirror_congruent_quadrilateral_property_perimeter_equal_1',
    'mirror_congruent_quadrilateral_property_area_equal_1', 'mirror_congruent_quadrilateral_property_exchange_1',
    'similar_quadrilateral_property_ratio_1', 'similar_quadrilateral_property_line_ratio_1',
    'similar_quadrilateral_property_angle_equal_1', 'similar_quadrilateral_property_perimeter_ratio_1',
    'similar_quadrilateral_property_area_square_ratio_1', 'mirror_similar_quadrilateral_property_ratio_1',
    'mirror_similar_quadrilateral_property_line_ratio_1', 'mirror_similar_quadrilateral_property_angle_equal_1',
    'mirror_similar_quadrilateral_property_perimeter_ratio_1',
    'mirror_similar_quadrilateral_property_area_square_ratio_1', 'parallelogram_judgment_parallel_and_parallel_1',
    'parallelogram_judgment_parallel_and_equal_1', 'parallelogram_judgment_equal_and_equal_1',
    'parallelogram_judgment_angle_and_angle_1', 'parallelogram_judgment_diagonal_bisection_1',
    'parallelogram_property_opposite_line_equal_1', 'parallelogram_property_opposite_angle_equal_1',
    'parallelogram_property_diagonal_bisection_1', 'parallelogram_area_formula_common_1',
    'parallelogram_area_formula_sine_1', 'kite_judgment_equal_and_equal_1',
    'kite_property_diagonal_perpendicular_bisection_1', 'kite_property_opposite_angle_equal_1',
    'kite_area_formula_diagonal_1', 'kite_area_formula_sine_1', 'rectangle_judgment_right_angle_1',
    'rectangle_judgment_diagonal_equal_1', 'rectangle_property_diagonal_equal_1',
    'rhombus_judgment_parallelogram_and_kite_1', 'square_judgment_rhombus_and_rectangle_1',
    'trapezoid_judgment_parallel_1', 'trapezoid_area_formula_1', 'right_trapezoid_judgment_right_angle_1',
    'right_trapezoid_area_formular_1', 'isosceles_trapezoid_judgment_line_equal_1',
    'isosceles_trapezoid_judgment_angle_equal_1', 'isosceles_trapezoid_judgment_diagonal_equal_1',
    'isosceles_trapezoid_property_angle_equal_1', 'isosceles_trapezoid_property_diagonal_equal_1', 'round_arc_1',
    'arc_addition_length_1', 'arc_addition_measure_1', 'arc_property_center_angle_1',
    'arc_property_circumference_angle_external_1', 'arc_property_circumference_angle_internal_1',
    'arc_length_formula_1', 'congruent_arc_judgment_length_equal_1', 'congruent_arc_judgment_measure_equal_1',
    'congruent_arc_judgment_chord_equal_1', 'congruent_arc_property_length_equal_1',
    'congruent_arc_property_measure_equal_1', 'congruent_arc_property_chord_equal_1',
    'similar_arc_judgment_cocircular_1', 'similar_arc_property_ratio_1', 'similar_arc_property_length_ratio_1',
    'similar_arc_property_measure_ratio_1', 'similar_arc_property_chord_ratio_1',
    'circle_property_length_of_radius_and_diameter_1', 'circle_property_circular_power_chord_and_chord_1',
    'circle_property_circular_power_tangent_and_segment_line_1',
    'circle_property_circular_power_segment_and_segment_line_1',
    'circle_property_circular_power_tangent_and_segment_angle_1',
    'circle_property_circular_power_tangent_and_segment_angle_2',
    'circle_property_circular_power_segment_and_segment_angle_1',
    'circle_property_circular_power_segment_and_segment_angle_2', 'circle_property_chord_perpendicular_bisect_chord_1',
    'circle_property_chord_perpendicular_bisect_chord_2', 'circle_property_chord_perpendicular_bisect_arc_1',
    'circle_property_chord_perpendicular_bisect_arc_2', 'circle_property_angle_of_osculation_1',
    'circle_property_angle_of_osculation_2', 'circle_perimeter_formula_1', 'circle_area_formula_1',
    'radius_of_circle_property_length_equal_1', 'diameter_of_circle_judgment_pass_centre_1',
    'diameter_of_circle_judgment_length_equal_1', 'diameter_of_circle_judgment_right_angle_1',
    'diameter_of_circle_property_length_equal_1', 'diameter_of_circle_property_right_angle_1',
    'tangent_of_circle_judgment_perpendicular_1', 'tangent_of_circle_judgment_perpendicular_2',
    'tangent_of_circle_property_perpendicular_1', 'tangent_of_circle_property_perpendicular_2',
    'tangent_of_circle_property_length_equal_1', 'sector_perimeter_formula_1', 'sector_area_formula_1',
    'perpendicular_bisector_judgment_per_and_bisect_1'
]
symbol_words = [
    ",", "+", "-", "**", "*", "/", "sin", "asin", "cos", "acos", "tan", "atan", "sqrt", "(", ")",
    "nums", 'll_', 'ma_', 'pt_', 'at_', 'ht_', 'rst_', 'rmt_', 'pq_', 'aq_', 'hq_', 'rsq_', 'rmq_', 'la_',
    'mar_', 'rsa_', 'rc_', 'dc_', 'pc_', 'ac_', 'ps_', 'as_'
]
letter_words = [
    "A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g", "H", "h", "I", "i", "J", "j", "K", "k", "L",
    "l", "M", "m", "N", "n", "O", "o", "P", "p", "Q", "q", "R", "r", "S", "s", "T", "t", "U", "u", "V", "v", "W", "w",
    "X", "x", "Y", "y", "Z", "z"
]
nodes_words = ["<padding>", "<start>", "<end>"] + predicate_words + symbol_words + letter_words
edges_words = ["<padding>", "<start>", "<end>"] + theorem_words

"""----↑----↑----↑----↑----↑----Vocabulary----↑----↑----↑----↑----↑----"""
"""----↓----↓----↓----↓----↓----Training data generation-----↓----↓----↓----↓----↓----"""


def get_hypertree(problem):
    """
    Build hypertree and return.
    :param problem: instance of <formalgeo.problem.Problem>.
    :return nodes: n*1, List of hyper nodes.
    :return path: n*n, Path from hyper node i to other nodes.
    """
    nodes, edges, _, tree = get_meta_hypertree(problem)

    for edge_id in edges:
        if "(" in edges[edge_id]:
            t_name, para = edges[edge_id].split("(")
            edges[edge_id] = "{}_{}".format(t_name, para[0])
    all_nodes = list(nodes.keys())
    path = [["none" for _ in all_nodes] for _ in all_nodes]
    for node_id in all_nodes:
        path[all_nodes.index(node_id)][all_nodes.index(node_id)] = "self"

    for premise, theorem in tree:  # init path
        conclusion = tree[(premise, theorem)]
        for head_node_id in premise:
            for tail_node_id in conclusion:
                path[all_nodes.index(head_node_id)][all_nodes.index(tail_node_id)] = edges[theorem]

    return list(nodes.values()), path


def tokenize_cdl(cdl):
    """
    Tokenize one cdl.
    >> tokenize_cdl('CongruentBetweenTriangle(RST,XYZ)')
    ['CongruentBetweenTriangle', 'R', 'S', 'T', ',', 'X', 'Y', 'Z']
    >> tokenize_cdl('Equation(ll_tr-x-21)')
    ['Equation', 'll_', 't', 'r', '-', 'x', '-', 'nums']
    """
    cdl = cdl[0:len(cdl) - 1].split("(", maxsplit=1)
    tokenized = [cdl[0]]
    if cdl[0] != "Equation":
        tokenized += list(cdl[1])
    else:
        for matched in re.findall(r"sin\(pi\*ma_\w+/180\)", cdl[1]):  # adjust trigonometric
            cdl[1] = cdl[1].replace(matched, "sin({})".format(matched[7:13]))
        for matched in re.findall(r"cos\(pi\*ma_\w+/180\)", cdl[1]):
            cdl[1] = cdl[1].replace(matched, "cos({})".format(matched[7:13]))
        for matched in re.findall(r"tan\(pi\*ma_\w+/180\)", cdl[1]):
            cdl[1] = cdl[1].replace(matched, "tan({})".format(matched[7:13]))

        for matched in re.findall(r"\d+\.*\d*", cdl[1]):  # replace real number with 'nums'
            cdl[1] = cdl[1].replace(matched, "nums", 1)

        while len(cdl[1]) > 0:  # tokenize
            length = len(cdl[1])
            for c in symbol_words:
                if cdl[1].startswith(c):
                    tokenized.append(cdl[1][0:len(c)])
                    cdl[1] = cdl[1][len(c):len(cdl[1])]
            if length == len(cdl[1]):
                tokenized.append(cdl[1][0])
                cdl[1] = cdl[1][1:len(cdl[1])]

    return tokenized


def tokenize_edge(edge):
    """
    Tokenize one edge and add structural information.
    >> tokenize_edge([['none'], ['none'], ['none'], ['self'], ['a', 'b'], ['none'], ['c']])
    >> (['self', 'a', 'b', 'c'], [3, 4, 4, 6])
    """
    tokenized = []
    token_index = []

    for j in range(len(edge)):
        if edge[j] not in ["self", "none", "extended", "solve_eq"]:
            tokenized.append(edge[j])
            token_index.append(j)

    return tokenized, token_index


def tokenize_goal(problem):
    """Tokenize goal."""
    if problem.goal.type == "algebra":
        goal = inverse_parse_one("Equation", problem.goal.item - problem.goal.answer, problem)
    else:
        goal = inverse_parse_one(problem.goal.item, problem.goal.answer, problem)
    return tokenize_cdl(goal)


def tokenize_theorem(theorem):
    """
    Tokenize one theorem.
    >> tokenize_theorem('congruent_triangle_property_angle_equal(1,RST,XYZ)')
    >> 'congruent_triangle_property_angle_equal_1'
    """
    t_name, t_branch, _ = parse_one_theorem(theorem)
    return "{}_{}".format(t_name, t_branch)


def get_problem_state(problem):
    """Get nodes, serialized_edges, edges_structure, goal from Problem."""
    nodes, edges = get_hypertree(problem)
    nodes = [tokenize_cdl(node) for node in nodes]
    serialized_edges = []
    edges_structure = []
    for edge in edges:
        serialized_edge, token_index = tokenize_edge(edge)
        serialized_edges.append(serialized_edge)
        edges_structure.append(token_index)
    goal = tokenize_goal(problem)
    return nodes, serialized_edges, edges_structure, goal


def apply_theorem(solver, theorem):
    t_name, t_branch, _ = parse_one_theorem(theorem)
    solver.apply_theorem(t_name, t_branch)
    solver.problem.check_goal()


def generate(dl, problem_id):
    """
    Generate training data of one problem.
    :return training_data: list of [nodes, serialized_edges, edges_structure, goal, theorems]
    """
    training_data = []
    solver = Interactor(dl.predicate_GDL, dl.theorem_GDL)
    problem_cdl = dl.get_problem(problem_id)
    solver.load_problem(problem_cdl)
    theorem_dag = problem_cdl["theorem_seqs_dag"]

    in_degree = {}
    for theorems in theorem_dag.values():
        for theorem in theorems:
            if theorem not in in_degree:
                in_degree[theorem] = 1
            else:
                in_degree[theorem] += 1

    for theorem in theorem_dag.keys():
        if theorem not in in_degree:
            in_degree[theorem] = 0

    zero_in_degree = []
    for tail in theorem_dag["START"]:
        in_degree[tail] -= 1
        if in_degree[tail] == 0:
            zero_in_degree.append(tail)

    while len(zero_in_degree) > 0:
        nodes, serialized_edges, edges_structure, goal = get_problem_state(solver.problem)
        theorems = list(set(tokenize_theorem(theorem) for theorem in zero_in_degree))
        training_data.append([nodes, serialized_edges, edges_structure, goal, theorems])  # add one step's data

        theorem = random.choice(zero_in_degree)  # random choose one theorem and remove same theorem
        theorem_tokenized = tokenize_theorem(theorem)
        removed = True
        while removed:
            removed = False
            for i in range(len(zero_in_degree))[::-1]:
                if tokenize_theorem(zero_in_degree[i]) == theorem_tokenized:
                    removed_theorem = zero_in_degree.pop(i)
                    removed = True
                    if removed_theorem in theorem_dag:
                        for tail in theorem_dag[removed_theorem]:
                            in_degree[tail] -= 1
                            if in_degree[tail] == 0:
                                zero_in_degree.append(tail)
        try:
            func_timeout(timeout=config["data"]["make_data_timeout"], func=apply_theorem, args=(solver, theorem))
        except FunctionTimedOut:
            pass
        if solver.problem.goal.solved:
            return training_data, solver.problem.goal.solved

    return training_data, solver.problem.goal.solved


def multiprocess_generate(dl, task_queue, reply_queue):
    warnings.filterwarnings("ignore")
    while not task_queue.empty():
        training_data_repeat = []
        msg = []
        problem_id = task_queue.get()
        reply_queue.put((os.getpid(), problem_id, "handled"))

        for i in range(config["data"]["make_data_repeat"]):  # data augmentation
            try:
                training_data, solved = generate(dl, problem_id)
            except BaseException as e:
                msg.append(f"error<{repr(e)}>")
                continue
            else:
                training_data_repeat.append(training_data)
                if solved:
                    msg.append("solved")
                else:
                    msg.append("unsolved")
        msg = ", ".join([f"{i + 1}: {msg[i]}" for i in range(len(msg))])

        save_pickle(training_data_repeat, f"../../data/training_data/{problem_id}.pkl")
        reply_queue.put((os.getpid(), problem_id, msg))


def make_data():
    """Generate original training data."""
    dl = DatasetLoader(dataset_name=config["data"]["dataset_name"], datasets_path=config["data"]["datasets_path"])
    log_filename = "../../data/outputs/log_data_generating.json"
    log = {"handled": [], "generated": {}}  # {pid: ['solved', 'unsolved', 'error']}
    if os.path.exists(log_filename):
        log = load_json(log_filename)

    task_queue = Queue()
    for problem_id in range(1, dl.info["problem_number"] + 1):
        if problem_id not in log["handled"]:
            task_queue.put(problem_id)

    reply_queue = Queue()
    process_ids = []  # child process id
    while True:
        for i in range(len(process_ids))[::-1]:  # Remove non-existent pid
            if not psutil.pid_exists(process_ids[i]):
                process_ids.pop(i)
        while not task_queue.empty() and config["multiprocess"] - len(process_ids) > 0:  # start new process
            process = Process(target=multiprocess_generate, args=(dl, task_queue, reply_queue))
            process.start()
            process_ids.append(process.pid)

        process_id, problem_id, msg = reply_queue.get()
        if msg == "handled":
            log["handled"].append(problem_id)
        else:
            log["generated"][problem_id] = msg
            print("{}: {}".format(problem_id, msg))
        safe_save_json(log, log_filename)


def make_train_val_test_split():
    filename = "../../data/outputs/problem_split.json"
    if os.path.exists(filename):
        return load_json(filename)

    dl = DatasetLoader(dataset_name=config["data"]["dataset_name"], datasets_path=config["data"]["datasets_path"])
    problem_ids = list(range(1, dl.info["problem_number"] + 1))
    random.shuffle(problem_ids)
    train, val, test = config["data"]["training_train_val_test"]
    train_problem_ids = sorted(problem_ids[:int(dl.info["problem_number"] * train / (train + val + test))])
    val_problem_ids = sorted(problem_ids[int(dl.info["problem_number"] * train / (train + val + test)):
                                         int(dl.info["problem_number"] * (train + val) / (train + val + test))])
    test_problem_ids = sorted(problem_ids[int(dl.info["problem_number"] * (train + val) / (train + val + test)):])
    problem_split = {"train": train_problem_ids, "val": val_problem_ids, "test": test_problem_ids}
    print(f"train: {len(train_problem_ids)}, val: {len(val_problem_ids)}, test: {len(test_problem_ids)}")
    save_json(problem_split, filename)

    return problem_split


def random_truncate_seqs(seqs, max_len):
    if len(seqs) <= max_len:
        return seqs, None

    pop_idx = sorted(random.sample(range(1, len(seqs)), len(seqs) - max_len))[::-1]
    for idx in pop_idx:
        seqs.pop(idx)

    return seqs, pop_idx


def random_truncate_problem_state(nodes, edges, edges_structure, goal):
    max_len_nodes = config["data"]["max_len_nodes"]
    max_len_edges = config["data"]["max_len_edges"]
    max_len_se = config["data"]["max_len_se"]
    max_len_graphs = config["data"]["max_len_graphs"]

    nodes, pop_idx = random_truncate_seqs(nodes, max_len_graphs)  # pop node
    if pop_idx is not None:
        for idx in pop_idx:
            edges.pop(idx)
            edges_structure.pop(idx)
        for idx in pop_idx:  # pop edge, edges structure from deleted node
            for i in range(len(edges_structure)):
                for k in range(len(edges_structure[i]))[::-1]:
                    if edges_structure[i][k] == idx:
                        edges_structure[i].pop(k)
                        edges[i].pop(k)

    for i in range(len(nodes)):  # nodes
        nodes[i], _ = random_truncate_seqs(nodes[i], max_len_nodes)

    for i in range(len(edges)):  # edges
        edges[i], pop_idx = random_truncate_seqs(edges[i], max_len_edges)
        if pop_idx is not None:
            for idx in pop_idx:
                edges_structure[i].pop(idx)  # edges structure

    for i in range(len(edges_structure)):  # edges structure
        for j in range(len(edges_structure[i])):
            if edges_structure[i][j] > max_len_se - 1:
                edges_structure = max_len_se - 1

    goal, _ = random_truncate_seqs(list(goal), max_len_nodes)  # goal

    return nodes, edges, edges_structure, goal


def make_problem_state_onehot(nodes, edges, edges_structure, goal):
    nodes, edges, edges_structure, goal = random_truncate_problem_state(
        nodes, edges, edges_structure, goal)
    for i in range(len(nodes)):
        for j in range(len(nodes[i])):
            nodes[i][j] = nodes_words.index(nodes[i][j])
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            edges[i][j] = edges_words.index(edges[i][j])

    for i in range(len(goal)):
        goal[i] = nodes_words.index(goal[i])

    return nodes, edges, edges_structure, goal


def make_theorems_onehot(theorems):
    theorems_onehot = [0] * len(theorem_words)
    for t in theorems:
        theorems_onehot[theorem_words.index(t)] = 1
    return theorems_onehot


def make_onehot():
    """Truncate oversize data and make onehot data."""
    dl = DatasetLoader(dataset_name=config["data"]["dataset_name"], datasets_path=config["data"]["datasets_path"])
    log = {"non_exist": [], "generated": [], "error": []}
    problem_split = load_json("../../data/outputs/problem_split.json")
    nodes_pretrain = []  # list of nodes
    edges_pretrain = []  # list of [edges, edges_structure]
    pretrain_ratio = config["data"]["training_train_val_test"][0] + config["data"]["training_train_val_test"][1]
    pretrain_ratio = pretrain_ratio / sum(config["data"]["training_train_val_test"])
    train = []  # list of training_data
    val = []  # list of training_data
    test = []  # list of training_data

    for problem_id in range(1, dl.info["problem_number"] + 1):
        what_sets = "train"
        if problem_id in problem_split["val"]:
            what_sets = "val"
        if problem_id in problem_split["test"]:
            what_sets = "test"
        if not os.path.exists(f"../../data/training_data/{problem_id}.pkl"):
            log["non_exist"].append(f"{problem_id}: {what_sets}")
            continue
        try:
            training_data_repeat = load_pickle(f"../../data/training_data/{problem_id}.pkl")
            for training_data in training_data_repeat:
                for nodes, edges, edges_structure, goal, theorems in training_data:
                    nodes, edges, edges_structure, goal = make_problem_state_onehot(
                        nodes, edges, edges_structure, goal
                    )
                    theorems = make_theorems_onehot(theorems)

                    for node in nodes:  # pretrain data
                        nodes_pretrain.append(node)
                    nodes_pretrain.append(goal)
                    for i in range(len(edges)):
                        edges_pretrain.append([edges[i], edges_structure[i]])

                    if what_sets == "train":  # training data
                        train.append([nodes, edges, edges_structure, goal, theorems])
                    elif what_sets == "val":
                        val.append([nodes, edges, edges_structure, goal, theorems])
                    else:
                        test.append([nodes, edges, edges_structure, goal, theorems])
        except IOError as e:
            log["error"].append(f"{problem_id}: {repr(e)}")
        else:
            log["generated"].append(problem_id)
            print("{} ok.".format(problem_id))

    random.shuffle(nodes_pretrain)
    nodes_pretrain = (nodes_pretrain[:int(pretrain_ratio * len(nodes_pretrain))],
                      nodes_pretrain[int(pretrain_ratio * len(nodes_pretrain)):])  # train, val
    save_pickle(nodes_pretrain, "../../data/training_data/nodes_pretrain_data.pkl")

    random.shuffle(edges_pretrain)
    edges_pretrain = (edges_pretrain[:int(pretrain_ratio * len(edges_pretrain))],
                      edges_pretrain[int(pretrain_ratio * len(edges_pretrain)):])  # train, val
    save_pickle(edges_pretrain, "../../data/training_data/edges_pretrain_data.pkl")

    save_pickle((train, val, test), "../../data/training_data/train_data.pkl")

    save_json(log, "../../data/outputs/log_make_data.json")


class GeoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def padding_sentence(added_token, sentence, add_start=True):
    """0 as padding token"""
    max_len = max([len(s) for s in sentence])

    if add_start:
        sentence_new = [[added_token] + s + [0] * (max_len - len(s)) for s in sentence]
    else:
        sentence_new = [s + [added_token] + [0] * (max_len - len(s)) for s in sentence]

    sentence_new = torch.tensor(sentence_new, dtype=torch.long)
    return sentence_new


def nodes_collate_fn(nodes_batch):
    input_nodes = padding_sentence(nodes_words.index("<start>"), nodes_batch)
    output_nodes = padding_sentence(nodes_words.index("<end>"), nodes_batch, add_start=False)
    return input_nodes, output_nodes


def edges_collate_fn(edges_batch):
    edges = [edge[0] for edge in edges_batch]
    structures = [edge[1] for edge in edges_batch]
    input_nodes = padding_sentence(nodes_words.index("<start>"), edges)
    input_structures = padding_sentence(nodes_words.index("<padding>"), structures)
    output_nodes = padding_sentence(nodes_words.index("<end>"), edges, add_start=False)

    return input_nodes, output_nodes, input_structures


def padding_matrix_batch(start_token, matrix_batch):
    """0 as padding token"""
    len_rows = []
    len_matrix = []
    for matrix in matrix_batch:
        len_matrix.append(len(matrix))
        for rows in matrix:
            len_rows.append(len(rows))
    max_len_rows = max(len_rows)
    max_len_matrix = max(len_matrix)

    matrix_batch_new = []
    for matrix in matrix_batch:
        matrix_batch_new.append([])
        for rows in matrix:
            matrix_batch_new[-1].append([start_token] + rows + [0] * (max_len_rows - len(rows)))
        matrix_batch_new[-1].extend([[0] * (max_len_rows + 1)] * (max_len_matrix - len(matrix_batch_new[-1])))

    matrix_batch_new = torch.tensor(matrix_batch_new, dtype=torch.long)
    return matrix_batch_new


def graph_collate_fn(graph_batch):
    nodes = padding_matrix_batch(nodes_words.index("<start>"), [graph[0] for graph in graph_batch])
    edges = padding_matrix_batch(edges_words.index("<start>"), [graph[1] for graph in graph_batch])
    structures = padding_matrix_batch(edges_words.index("<padding>"), [graph[2] for graph in graph_batch])
    goals = padding_sentence(nodes_words.index("<start>"), [graph[3] for graph in graph_batch])
    theorems = torch.tensor([graph[4] for graph in graph_batch], dtype=torch.long)

    return nodes, edges, structures, goals, theorems


"""----↑----↑----↑----↑----↑----Training Data Generation-----↑----↑----↑----↑----↑----"""
"""----↓----↓----↓----↓----↓----↓----↓----Tools----↓----↓----↓----↓----↓----↓----↓----"""


def show_training_data(problem_id):
    dl = DatasetLoader(dataset_name=config["data"]["dataset_name"], datasets_path=config["data"]["datasets_path"])
    training_data, solved = generate(dl, problem_id)
    for i in range(len(training_data)):
        print("nodes (step {}):".format(i + 1))
        for node in training_data[i][0]:
            print(node)
        print("edges (step {}):".format(i + 1))
        for edge in training_data[i][1]:
            print(edge)
        print("edges_structure (step {}):".format(i + 1))
        for edges_structure in training_data[i][2]:
            print(edges_structure)
        print("goal (step {}):".format(i + 1))
        print(training_data[i][3])
        print("theorems (step {}):".format(i + 1))
        print(training_data[i][4])
        print()


def statistic_training_data():
    nodes_pretrain, nodes_pretrain_val = load_pickle("../../data/training_data/nodes_pretrain_data.pkl")
    edges_pretrain, edges_pretrain_val = load_pickle("../../data/training_data/edges_pretrain_data.pkl")
    print(f"nodes pretrain: {len(nodes_pretrain)}, nodes val: {len(nodes_pretrain_val)}")
    print(f"edges pretrain: {len(edges_pretrain)}, edges val: {len(edges_pretrain_val)}")

    print("train (problem, item):")
    problem_split = make_train_val_test_split()
    print(f"train: {len(problem_split['train'])}, val: {len(problem_split['val'])}, test: {len(problem_split['test'])}")

    train, val, test = load_pickle("../../data/training_data/train_data.pkl")
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")


def check_log():
    """solved, unsolved, error"""
    log_filename = os.path.normpath(os.path.join(config.path_data, "log/gen_training_data_log.json"))
    log = load_json(log_filename)
    log = {int(k): log[k] for k in log}
    log = {k: log[k] for k in sorted(log)}
    safe_save_json(log, log_filename)  # save sorted log

    timeout = []
    unsolved = []
    error = []
    unhandled = []
    for pid in range(1, 6982):
        if pid not in log:
            unhandled.append(pid)
        elif "timeout" in log[pid]:
            timeout.append(pid)
        elif "error" in log[pid]:
            error.append(pid)
        elif "unsolved" in log[pid]:
            unsolved.append(pid)

    print("timeout ({}):".format(len(timeout)))
    for pid in timeout:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("unsolved ({}):".format(len(unsolved)))
    for pid in unsolved:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("error ({}):".format(len(error)))
    for pid in error:
        print("{}: {}".format(pid, log[pid]))
    print()
    print("unhandled ({}):".format(len(unhandled)))
    print(unhandled)


def check_len():
    """Show length distribution of training data."""
    graph_len = {}
    nodes_len = {}
    edges_len = {}
    se_len = {}
    for pid in load_json("../../data/outputs/log_data_generating.json")["generated"]:
        training_data_repeat = load_pickle(f"../../data/training_data/{pid}.pkl")
        for training_data in training_data_repeat:
            for nodes, serialized_edges, edges_structure, goal, theorems in training_data:
                if len(nodes) not in graph_len:  # graph len
                    graph_len[len(nodes)] = 1
                else:
                    graph_len[len(nodes)] += 1

                for node in nodes:  # nodes_len
                    if len(node) not in nodes_len:
                        nodes_len[len(node)] = 1
                    else:
                        nodes_len[len(node)] += 1

                for edge in serialized_edges:  # edges_len
                    if len(edge) not in edges_len:
                        edges_len[len(edge)] = 1
                    else:
                        edges_len[len(edge)] += 1

                for structural in edges_structure:  # se_len
                    for se in structural:
                        if se not in se_len:
                            se_len[se] = 1
                        else:
                            se_len[se] += 1

                if len(goal) not in nodes_len:  # goal is node
                    nodes_len[len(goal)] = 1
                else:
                    nodes_len[len(goal)] += 1
        print("{} ok.".format(pid))

    graph_len = [(k, graph_len[k]) for k in sorted(graph_len)]
    nodes_len = [(k, nodes_len[k]) for k in sorted(nodes_len)]
    edges_len = [(k, edges_len[k]) for k in sorted(edges_len)]
    se_len = [(k, se_len[k]) for k in sorted(se_len)]

    draw_length_pic(graph_len, "graph_len", (128, 8))
    draw_length_pic(nodes_len, "nodes_len", (32, 8))
    draw_length_pic(edges_len, "edges_len", (64, 8))
    draw_length_pic(se_len, "se_len", (128, 8))


def draw_length_pic(data, title, fig_size):
    """draw plot of data length"""
    log = {}
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    y_sum = sum(y)
    for i in range(len(x)):
        log[x[i]] = "count:{}, percent: {:.6f}%".format(y[i], y[i] / y_sum * 100)
    y_integral = [y[0]]
    for i in range(1, len(y)):
        y_integral.append(y_integral[i - 1] + y[i])
    y_integral = [item / y_integral[-1] for item in y_integral]
    for i in range(len(x)):
        log[x[i]] += ", accumulate: {:.6f}%".format(y_integral[i] * 100)
    safe_save_json(log, f"../../data/outputs/check_len_{title}.json")
    print("{}:".format(title))
    for i in range(len(x) - 1):
        if y_integral[i] <= 0.9:
            if y_integral[i + 1] >= 0.9:
                print("0.9: {}".format(x[i + 1]))
        elif y_integral[i] <= 0.95:
            if y_integral[i + 1] >= 0.95:
                print("0.95: {}".format(x[i + 1]))
        elif y_integral[i] <= 0.99:
            if y_integral[i + 1] >= 0.99:
                print("0.99: {}".format(x[i + 1]))
    print("1.0: {}".format(x[-1]))
    print()

    plt.figure(figsize=fig_size)
    plt.plot(x, y, marker='o')
    plt.title('{} (Density)'.format(title))
    plt.savefig(f"../../data/outputs/check_len_{title}_density.png")
    plt.close()

    plt.figure(figsize=fig_size)
    for i in range(len(x) - 1):  # draw line
        plt.plot(x[i:i + 2], y_integral[i:i + 2], 'b-')
    for i in range(len(x)):  # draw point
        if y_integral[i] < 0.9:
            plt.plot(x[i], y_integral[i], 'o', color="green")
        elif y_integral[i] < 0.95:
            plt.plot(x[i], y_integral[i], 'o', color="yellow")
        elif y_integral[i] < 0.99:
            plt.plot(x[i], y_integral[i], 'o', color="pink")
        else:
            plt.plot(x[i], y_integral[i], 'o', color="red")
    plt.title('{} (Integral)'.format(title))
    plt.savefig(f"../../data/outputs/check_len_{title}_integral.png")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["generate_training_data", "make_data"],
                        help="function that you want to run")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    kill subprocess:
    python utils.py --func kill --py_filename data.py
    
    Generating raw data using FormalGeo:
    python data.py --func make_data
    
    Making training data:
    python data.py --func make_onehot
    """
    args = get_args()
    if args.func == "make_data":
        make_data()
    elif args.func == "make_onehot":
        make_onehot()
