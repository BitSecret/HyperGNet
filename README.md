# FGeo-HyperGNet

[![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen)](https://github.com/BitSecret/HyperGNet)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Survey](https://img.shields.io/badge/Survey-FormalGeo-blue)](https://github.com/FormalGeo/FormalGeo)

This project is the official implementation of the IJCAI 2025 paper ["FGeo-HyperGNet: Geometric Problem Solving
Integrating FormalGeo Symbolic System and Hypergraph Neural Network"](https://github.com/BitSecret/HyperGNet). We built a
neural-symbolic system, called
FGeo-HyperGNet, to automatically perform human-like geometric problem solving.

<div>
    <img src="architecture.png" alt="overall architecture">
</div>

The symbolic component is a formal system built on [FormalGeo](https://github.com/FormalGeo/FormalGeo), which can
automatically perform geometric relational reasoning and algebraic calculations and organize the solution into a
hypergraph with conditions as hypernodes and theorems as hyperedges.

The neural component, called HyperGNet, is a hypergraph neural network based on the attention mechanism, including an
encoder to effectively encode the structural and semantic information of the hypergraph, and a solver to provide
guidance of solving geometric problem.

The neural component predicts theorems according to the hypergraph, and the symbolic component applies theorems and
updates the hypergraph, thus forming a predict-apply cycle to ultimately achieve readable and traceable automatic
solving of geometric problems. Experiments demonstrate the correctness and effectiveness of this neural-symbolic
architecture. Experiments demonstrate the correctness and effectiveness of this neural-symbolic architecture. We
achieved a TAP of 93.50% and a PSSR of 88.36% on the [FormalGeo7K-v2](https://github.com/FormalGeo/FormalGeo) datasets.

More information about FormalGeo will be found in [homepage](https://formalgeo.github.io/). FormalGeo is in its early
stages and brimming with potential. We welcome anyone to join us in this exciting endeavor.

## Installation

Clone project:

    $ git clone --depth 1 https://github.com/BitSecret/HyperGNet.git

Create Python environments using Conda:

    $ conda create -n HyperGNet python=3.10
    $ conda activate HyperGNet

Install Python dependencies:

    $ cd HyperGNet
    $ pip install -e .
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Enter the code path (all subsequent code will be executed in this path):

    $ cd src/gps

We provide a short code to test if the environment setup is successful:

    $ python utils.py --func test_env

Modify the addresses of datasets `datasets_path` (for windows or macOS) and `datasets_path_linux` (for linux) in
the `data/config.json` file, and then run the script to download the FormalGeo7K-v2 datasets:

    $ python utils.py --func download_dataset

Download our trained model and generated data
through [Google Drive](https://drive.google.com/file/d/1hqlObUC7GKFrJ4hakMZAplx6GJdVdVH-/view?usp=sharing)
or [Baidu NetDisk](https://pan.baidu.com/s/14zMcV3dXAPKvbF0kKgYNeg?pwd=3m9y) and put them
into `data/` path.

Now, your working directory should looks like this:

```
|--data
|  |--checkpoints
|  |--outputs
|  |--training_data
|  |--config.json
|
|--src
|  |--gps
|     |--__init__.py
|     |--data.py
|     |--model.py
|     |--pac.py
|     |--train.py
|     |--utils.py
|
|--architecture.png
|
|--LICENSE
|
|--pyproject.toml
|
|--README.md
```

## Preparing Training Data

We use **FormalGeo** as the formal environment for solving geometric problems and generate training data using the
**FormalGeo7K-v2** dataset. The dataset is split into `training:validation:test=3:1:1`. Each annotated theorem
sequence can be organized into a directed acyclic graph (DAG). We randomly traverse this DAG and apply each step's
theorem while obtaining the state information of the problem. Each problem will repeat 5 times (the number of
repetitions is defined in the `data/config.json` file).

This process ultimately generated 71,234 training data entries (from 4,200 problems), 23,878 validation data entries (
from 1,400 problems), and 23,522 test data entries (from 1,400 problems). Each data entry can be viewed as an
input-label pair `((nodes, edges, structural_encoding, goals), theorems)`.

We have provided the generated data, located at the path `data/training_data`. You can also use a multi-process approach
to generate training data:

    $ python data.py --func make_data
    $ python data.py --func make_onehot

In the end, you will obtain the files `nodes_pretrain_data.pkl`, `edges_pretrain_data.pkl`, and `train_data.pkl` in
the `data/training_data` path.

## Training

We first pretrain the hypernode embedding networks using a self-supervised method.

    $ python train.py --func pretrain_nodes
    $ python train.py --func pretrain_edges

Then, train the theorem prediction network:

    $ python train.py --func train

The training log will be saved in `data/outputs`.

## Testing

Test theorem prediction accuracy (TPA):

    $ python train.py --func test

Test problem solving success rate (PSSR):

    $ python pac.py

The test results of the model will be saved in `HyperGNet/data/outputs`.

## Results

Show experimental results:

    $ python utils.py --func show_contrast_results
    $ python utils.py --func show_ablation_results

### Comparative Experiment

Table 1: PSSR of different methods on the FormalGeo7K dataset.

| Method              | Total | L1    | L2    | L3    | L4    | L5    | L6    |
|---------------------|-------|-------|-------|-------|-------|-------|-------|
| Forward Search      | 39.71 | 58.47 | 41.01 | 34.16 | 16.40 | 5.45  | 4.79  |
| Backward Search     | 35.44 | 66.43 | 34.98 | 11.78 | 6.56  | 6.09  | 1.03  |
| T5-small with FGeo  | 36.14 | 49.90 | 34.84 | 34.59 | 23.57 | 8.06  | 3.33  |
| BART-base with FGeo | 54.00 | 73.90 | 56.12 | 50.38 | 26.75 | 16.13 | 8.33  |
| DeepSeek-v3         | 60.79 | 75.99 | 56.38 | 63.91 | 43.31 | 32.26 | 28.33 |
| Inter-GPS           | 60.50 | 76.2  | 63.30 | 60.90 | 39.49 | 17.74 | 15.00 |
| NGS                 | 62.60 | 62.22 | 64.97 | 72.79 | 57.47 | 56.41 | 36.59 |
| DualGeoSolver       | 62.11 | 62.96 | 67.80 | 65.44 | 60.92 | 53.85 | 34.15 |
| FGeo-TP             | 80.86 | 96.43 | 85.44 | 76.12 | 62.26 | 48.88 | 29.55 |
| FGeo-DRL            | 80.85 | 97.61 | 91.88 | 70.82 | 57.55 | 36.17 | 27.59 |
| FGeo-HyperGNet      | 88.36 | 96.24 | 91.76 | 87.59 | 82.17 | 56.45 | 56.67 |

Table 2: PSSR of existing state-of-the-art methods and FGeo-HyperGNet on different datasets.

| Method         | Geometry3K | GeoQA | FormalGeo7K |
|----------------|------------|-------|-------------|
| GeoDRL         | 89.40      | -     | -           |
| E-GPS          | 90.40      | -     | -           |
| SCA-GPS        | -          | 64.10 | -           |
| DualGeoSolver  | -          | 65.20 | -           |
| FGeo-TP        | -          | -     | 80.86       |
| DFE-GPS        | -          | -     | 82.38       |
| FGeo-HyperGNet | 91.99      | 85.64 | 88.36       |

### Ablation study

Table 3: Ablation study results of HyperGNet on the FormalGeo7k dataset.

| Method         | Beam Size | TPA   | PSSR  |
|----------------|-----------|-------|-------|
|                | 1         | 71.58 | 44.86 |
| FGeo-HyperGNet | 3         | 88.91 | 62.93 |
|                | 5         | 93.50 | 67.79 |
|                | 1         | 70.73 | 41.57 |
| -w/o Pretrain  | 3         | 87.36 | 59.36 |
|                | 5         | 92.21 | 64.43 |
|                | 1         | 70.33 | 39.64 |
| -w/o SE        | 3         | 88.14 | 60.21 |
|                | 5         | 92.48 | 64.14 |
|                | 1         | 68.11 | 36.93 |
| -w/o Hypertree | 3         | 87.38 | 57.57 |
|                | 5         | 92.00 | 63.07 |

## Acknowledge

This project is maintained by [FormalGeo Development Team](https://formalgeo.github.io/).  
Please contact with [Xiaokai Zhang](https://bitsecret.github.io/) if you encounter any issues.

## Citation

To cite HyperGNet in publications use:
> Zhang X, Li Y, Zhu N, Qin C, Zeng Z, and Leng T. FGeo-HyperGNet: Geometric Problem Solving Integrating FormalGeo
> Symbolic System and Hypergraph Neural Network[C]. Proceedings of the Thirty-Fourth International Joint Conference on
> Artificial Intelligence. 2025.

A BibTeX entry for LaTeX users is:
> @inproceedings{zhang2025fgeohypergnet,  
> title={FGeo-HyperGNet: Geometric Problem Solving Integrating FormalGeo Symbolic System and Hypergraph Neural Network},  
> author={Zhang, Xiaokai and Li, Yang and Zhu, Na and Qin, Cheng and Zeng, Zhengbing and Leng, Tuo},  
> booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},  
> year={2025},  
> }
