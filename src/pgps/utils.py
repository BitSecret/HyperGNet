import os
import zipfile
import pickle
import psutil
import random
from formalgeo.data import download_dataset

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
    """---------path---------"""
    path_data = "./231221"

    """---------datasets---------"""
    path_datasets = "./datasets"
    dataset_name = "formalgeo7k_v1"

    """---------random_seed---------"""
    random_seed = 619

    """---------training data generation---------"""
    process_count = int(psutil.cpu_count() * 0.8)

    """---------model hyperparameter---------"""
    # pretrain - nodes
    batch_size_nodes = 64
    epoch_nodes = 10
    lr_nodes = 0.001

    vocab_nodes = len(nodes_words)
    max_len_nodes = 22
    h_nodes = 8
    N_encoder_nodes = 4
    N_decoder_nodes = 4
    p_drop_nodes = 0.5

    # pretrain - edges
    batch_size_edges = 64
    epoch_edges = 10
    lr_edges = 0.001

    vocab_edges = len(edges_words)
    max_len_edges = 16
    max_len_edges_se = 1070
    h_edges = 8
    N_encoder_edges = 4
    N_decoder_edges = 4
    p_drop_edges = 0.5

    # train
    batch_size = 64
    epoch = 10
    lr = 0.001

    vocab_theorems = len(theorem_words)
    max_len = 64
    h = 8
    N = 6
    p_drop = 0.5
    d_model = 512


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


def random_index(n, k):
    """Return k random idx. 1 is set intentionally."""
    return sorted(random.sample(range(1, n), k))


def project_init():
    if not os.path.exists("./datasets"):  # download_datasets
        os.makedirs("./datasets")
        download_dataset("formalgeo7k_v1", "./datasets")
        download_dataset("formalgeo-imo_v1", "./datasets")

    filepaths = [  # create_archi
        os.path.normpath(os.path.join(Configuration.path_data, "log/words_len")),
        os.path.normpath(os.path.join(Configuration.path_data, "log/tensorboard")),
        os.path.normpath(os.path.join(Configuration.path_data, "trained_model")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/example_data")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/train/raw")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/val/raw")),
        os.path.normpath(os.path.join(Configuration.path_data, "training_data/test/raw"))
    ]
    for filepath in filepaths:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    if os.path.exists("./231221.zip"):  # unzip_results
        with zipfile.ZipFile("231221.zip", 'r') as zip_ref:
            zip_ref.extractall("./")


if __name__ == '__main__':
    # project_init()
    show_word_list()
