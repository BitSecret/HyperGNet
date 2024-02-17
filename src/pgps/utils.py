from formalgeo.data import download_dataset, DatasetLoader
from formalgeo.tools import load_json
import matplotlib.pyplot as plt
import os
import zipfile
import pickle
import psutil
import random
import argparse

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
edges_words = ["<padding>", "<start>", "<end>"] + theorem_words + ["self", "extended", "solve_eq"]


class Configuration:
    """---------os---------"""
    path_data = "data"  # path

    path_datasets = "datasets"  # datasets
    dataset_name = "formalgeo7k_v1"

    random_seed = 619  # I set 619 as the random seed to commemorate June 19, 2023.
    torch_seed = 619
    cuda_seed = 619

    process_count = int(psutil.cpu_count() * 0.8)  # training data generation

    """---------model hyperparameter---------"""
    # pretrain - nodes
    batch_size_nodes = 64
    epoch_nodes = 50
    lr_nodes = 1e-5
    vocab_nodes = len(nodes_words)
    max_len_nodes = 22
    h_nodes = 4
    N_encoder_nodes = 4
    N_decoder_nodes = 4
    p_drop_nodes = 0.5

    # pretrain - edges
    batch_size_edges = 64
    epoch_edges = 50
    lr_edges = 1e-5
    vocab_edges = len(edges_words)
    max_len_edges = 16
    max_len_edges_se = 1070
    h_edges = 4
    N_encoder_edges = 4
    N_decoder_edges = 4
    p_drop_edges = 0.5

    # pretrain - graph structure
    batch_size_gs = 64
    epoch_gs = 50
    lr_gs = 1e-5
    vocab_gs = 1070
    max_len_gs = max_len_edges
    h_gs = 4
    N_encoder_gs = 4
    N_decoder_gs = 4
    p_drop_gs = 0.5

    # train
    batch_size = 64
    epoch = 50
    lr = 1e-5
    vocab_theorems = len(theorem_words)
    max_len = 64
    h = 4
    N = 4
    p_drop = 0.5
    d_model = 256


random.seed(Configuration.random_seed)


def show_word_list():
    print("Input - nodes and goal (special_words + predicate_words + symbol_words + letter_words): {} words.".format(
        len(nodes_words)))
    print("Input - edges (special_words + theorem_words + special_theorems): {} words.".format(len(edges_words)))
    print("Output - theorem (theorem_words): {} words.".format(len(theorem_words)))


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def project_init():
    if not os.path.exists(Configuration.path_datasets):  # download_datasets
        os.makedirs(Configuration.path_datasets)
        download_dataset("formalgeo7k_v1", Configuration.path_datasets)
        download_dataset("formalgeo-imo_v1", Configuration.path_datasets)

    filepaths = [  # create_archi
        os.path.normpath(os.path.join(Configuration.path_data, "log/words_length")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/nodes_pretrain")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/edges_pretrain")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/gs_pretrain")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/train")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/test")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/experiments")),
        os.path.normpath(os.path.join(Configuration.path_data, "trained_model")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/train/raw")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/val/raw")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/test/raw"))
    ]
    for filepath in filepaths:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    zip_data_path = "../../data.zip"
    if os.path.exists(zip_data_path):  # unzip_results
        with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
            zip_ref.extractall()


def search_alignment(log, test_pids, timeout=30):
    """
    Alignment search results.
    1.select the test set data.
    2.view the problem which search time greater than <timeout> as timeout.
    """
    alignment_log = {"solved": {}, "unsolved": {}, "timeout": {}, "error": {}}
    for pid in test_pids:
        pid = str(pid)
        for key in log:
            if pid in log[key]:
                if log[key][pid]["timing"] > timeout:
                    alignment_log["timeout"][pid] = log[key][pid]
                else:
                    alignment_log[key][pid] = log[key][pid]
                break
    return pac_alignment(alignment_log, test_pids, timeout)


def pac_alignment(log, test_pids, timeout=30):
    """
    Alignment PAC results.
    1.set the unhandled problems as unsolved and set it timeout as <timeout>.
    2.sort by problem id.
    3.set the search time greater than <timeout> as timeout.
    """
    alignment_log = {"solved": {}, "unsolved": {}, "timeout": {}}
    for pid in test_pids:
        pid = str(pid)
        added = False
        for key in log:
            if pid in log[key]:
                if key != "error":
                    alignment_log[key][pid] = log[key][pid]
                    if alignment_log[key][pid]["timing"] > timeout:
                        alignment_log[key][pid]["timing"] = timeout
                else:
                    alignment_log["unsolved"][pid] = log[key][pid]
                    if alignment_log["unsolved"][pid]["timing"] > timeout:
                        alignment_log["unsolved"][pid]["timing"] = timeout
                added = True
                break

        if not added:
            alignment_log["unsolved"][pid] = {"msg": "Unhandled problems.", "timing": timeout * 0.5, "step_size": 1}

    return alignment_log


def evaluate():
    evaluation_data_path = os.path.normpath(os.path.join(Configuration.path_data, "log/experiments/{}"))
    figure_save_path = os.path.normpath(os.path.join(Configuration.path_data, "log/{}"))
    dl = DatasetLoader(Configuration.dataset_name, Configuration.path_datasets)
    test_pids = dl.get_problem_split()["split"]["test"]

    level_count = 6
    level_map = {}
    for pid in test_pids:
        t_length = dl.get_problem(pid)["problem_level"]
        if t_length <= 2:
            level_map[pid] = 0
        elif t_length <= 4:
            level_map[pid] = 1
        elif t_length <= 6:
            level_map[pid] = 2
        elif t_length <= 8:
            level_map[pid] = 3
        elif t_length <= 10:
            level_map[pid] = 4
        else:
            level_map[pid] = 5
    level_total = [0 for _ in range(level_count)]
    for pid in test_pids:
        level_total[level_map[pid]] += 1

    contrast_log_files = [  # table 1 and 2
        search_alignment(load_json(evaluation_data_path.format("formalgeo7k_v1-data-fw-bfs.json")), test_pids),
        search_alignment(load_json(evaluation_data_path.format("formalgeo7k_v1-data-fw-dfs.json")), test_pids),
        search_alignment(load_json(evaluation_data_path.format("formalgeo7k_v1-data-fw-rs.json")), test_pids),
        search_alignment(load_json(evaluation_data_path.format("formalgeo7k_v1-data-fw-bs.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_5.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_greedy_beam_5.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_best.json")), test_pids, 600)
    ]
    dl = DatasetLoader(Configuration.dataset_name, Configuration.path_datasets)
    test_pids = dl.get_problem_split()["split"]["test"]

    print("Table 1:")
    print("solved\tunsolved\ttimeout")
    for file in contrast_log_files:
        print("{:.2f}\t{:.2f}\t{:.2f}".format(
            len(file["solved"]) / len(test_pids) * 100,
            len(file["unsolved"]) / len(test_pids) * 100,
            len(file["timeout"]) / len(test_pids) * 100
        ))
    print()

    table_data = [[0 for _ in range(level_count)] for _ in range(len(contrast_log_files))]
    for i in range(len(contrast_log_files)):
        for pid in test_pids:
            if str(pid) in contrast_log_files[i]["solved"]:
                table_data[i][level_map[pid]] += 1

    print("Table 2:")
    print("total" + "".join([f"\tl{i + 1}" for i in range(level_count)]))
    for i in range(len(contrast_log_files)):
        print("{:.2f}".format(len(contrast_log_files[i]["solved"]) / len(test_pids) * 100), end="")
        for j in range(level_count):
            print("\t{:.2f}".format(table_data[i][j] / level_total[j] * 100), end="")
        print()
    print()

    step_wised_log_files = [  # table 3
        load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_1.json")),
        load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_3.json")),
        load_json(evaluation_data_path.format("predictor_test_log_pretrain_beam_5.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_1.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_3.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_pretrain_beam_5.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_1.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_3.json")),
        load_json(evaluation_data_path.format("predictor_test_log_no_hyper_beam_5.json"))
    ]
    pac_log_files = [  # table 3
        pac_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_1.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_3.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_pretrain_beam_5.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_1.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_3.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_pretrain_beam_5.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_1.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_3.json")), test_pids),
        pac_alignment(load_json(evaluation_data_path.format("pac_log_no_hyper_beam_5.json")), test_pids)
    ]
    print("Table 3:")
    print("step_wised_acc\toverall_acc\tavg_time\tavg_step")
    for i in range(len(step_wised_log_files)):
        time_sum = 0
        step_sum = 0
        for key in pac_log_files[i]:
            for pid in pac_log_files[i][key]:
                time_sum += pac_log_files[i][key][pid]["timing"]
                step_sum += pac_log_files[i][key][pid]["step_size"]
        print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            step_wised_log_files[i]["acc"] * 100,
            len(pac_log_files[i]["solved"]) / len(test_pids) * 100,
            time_sum / len(test_pids),
            step_sum / len(test_pids)
        ))

    timing = [[0 for _ in range(level_count)] for _ in range(len(contrast_log_files))]
    step = [[0 for _ in range(level_count)] for _ in range(len(contrast_log_files))]

    for i in range(len(contrast_log_files)):
        for key in contrast_log_files[i]:
            for pid in contrast_log_files[i][key]:
                timing[i][level_map[int(pid)]] += contrast_log_files[i][key][pid]["timing"]
                step[i][level_map[int(pid)]] += contrast_log_files[i][key][pid]["step_size"]

    for i in range(len(contrast_log_files)):
        for j in range(level_count):
            timing[i][j] /= level_total[j]
            step[i][j] /= level_total[j]

    x = [i + 1 for i in range(level_count)]
    fontsize = 24
    axis_fontsize = 15
    line_width = 3
    plt.figure(figsize=(16, 8))  # figure 1

    plt.subplot(121)
    plt.plot(x, timing[0], label="FW-BFS", linewidth=line_width)
    plt.plot(x, timing[1], label="FW-DFS", linewidth=line_width)
    plt.plot(x, timing[2], label="FW-RS", linewidth=line_width)
    plt.plot(x, timing[3], label="FW-RBS", linewidth=line_width)
    plt.plot(x, timing[4], label="HyperGNet-NB", linewidth=line_width)
    plt.plot(x, timing[5], label="HyperGNet-GB", linewidth=line_width)
    plt.xlabel("Problem Difficulty", fontsize=fontsize)
    plt.ylabel("Avg Time (s)", fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=axis_fontsize)
    plt.tick_params(axis='both', labelsize=axis_fontsize)

    plt.subplot(122)
    plt.plot(x, step[4], label="HyperGNet-NB", linewidth=line_width)
    plt.plot(x, step[5], label="HyperGNet-GB", linewidth=line_width)
    plt.xlabel("Problem Difficulty", fontsize=fontsize)
    plt.ylabel("Avg Step", fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=axis_fontsize)
    plt.tick_params(axis='both', labelsize=axis_fontsize)

    plt.tight_layout()
    plt.savefig(figure_save_path.format("time_and_step.pdf"), format='pdf')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use PGPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["project_init", "show_word_list", "evaluate", "draw", "kill"],
                        help="function that you want to run")
    parser.add_argument("--py_filename", type=str, required=False,
                        help="python filename that you want to kill")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


def clean_process(py_filename):
    for pid in psutil.pids():
        process = psutil.Process(pid)
        if process.name() == "python" and py_filename in process.cmdline():
            print(f"kill -9 {pid}")


if __name__ == '__main__':
    args = get_args()
    if args.func == "project_init":
        project_init()
    elif args.func == "show_word_list":
        show_word_list()
    elif args.func == "evaluate":
        evaluate()
    elif args.func == "kill":
        clean_process(args.py_filename)
