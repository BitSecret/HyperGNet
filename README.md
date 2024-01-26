# FGeo-HyperGNet

[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/BitSecret/PGPS)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Survey](https://img.shields.io/badge/Survey-FormalGeo-blue)](https://github.com/FormalGeo/FormalGeo)

We built a neural-symbolic system to automatically perform human-like geometric deductive reasoning.

<div>
    <img src="architecture.png" alt="overall architecture">
</div>

The symbolic part is a formal system built on [FormalGeo](https://github.com/FormalGeo/FormalGeo), which can
automatically perform algebraic calculations and relational reasoning and organize the solving process into a solution
hypertree with conditions as hypernodes and theorems as hyperedges.

The neural part is a hypergraph neural network based on the attention mechanism, including a encoder to effectively
encode the structural and semantic information of the hypertree, and a solver to provide problem-solving guidance.

The neural part predicts theorems according to the hypertree, and the symbolic part applies theorems and updates the
hypertree, thus forming a predict-apply cycle to ultimately achieve readable and traceable automatic solving of
geometric problems. Experiments demonstrate the correctness and effectiveness of this neural-symbolic architecture. We
achieved a step-wised accuracy of 87.65% and an overall accuracy of 85.53% on
the [formalgeo7k](https://github.com/FormalGeo/Datasets) datasets.

More information about FormalGeo will be found in [homepage](https://formalgeo.github.io/). FormalGeo is in its early
stages and brimming with potential. We welcome anyone to join us in this exciting endeavor.

## Installation

Clone project:

    $ git clone --depth 1 https://github.com/BitSecret/PGPS.git
    $ cd PGPS

Create Python environments using Conda:

    $ conda create -n PGPS python=3.10
    $ conda activate PGPS

Install Python dependencies:

    $ cd PGPS
    $ pip install -e .
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

We provide a short code to test if the environment setup is successful:

    $ cd PGPS/tests
    $ python test.py

Download datasets and initialize the project:

    $ cd PGPS/src/pgps
    $ python utils.py --func project_init

Enter the code path (all subsequent code will be executed in this path):

    $ cd PGPS/src/pgps

## Preparing Training Data

We use **FormalGeo** as the formal environment for solving geometric problems and generate training data using the
**formalgeo7k_v1** dataset. Following the dataset's recommended division ratio and random seed, the dataset is split
into `training:validation:test=3:1:1`. Each problem-solving theorem sequence can be organized into a directed acyclic
graph (DAG). We randomly traverse this DAG and apply each step's theorem while obtaining the state information of the
problem.  
This process ultimately generated 20,571 training data entries (from 4,079 problems), 7,072 validation data entries (
from 1,370 problems), and 7,046 test data entries (from 1,372 problems). Each data entry can be viewed as a
5-tuple `(nodes, edges, edges_structural, goal, theorems)`.   
These data are saved in `data/training_data`. View the generated data (problem 1):

    $ python symbolic_system.py --func show_training_data

View the statistical information of the generated data:

    $ python symbolic_system.py --func check

This will generate message about the length distribution and visual images of the training data in
the `data/log/words_length` folder.  
If you want to regenerate the training data:

    $ python symbolic_system.py --func main

The training data will be regenerated using multiple processes. The log files will be saved
in `data/log/gen_training_data_log.json`. Subsequently, the generated data need to be converted into a vector form
suitable for neural networks input.

    $ python symbolic_system.py --func make_onehot

## Training

Before starting the training, ensure that the training data `data/training_data/train/one-hot.pk` has been generated.
If not, run `python symbolic_system.py --func make_onehot` to generate the data.  
We first pretrain the embedding networks for nodes, edges and graph structure using a self-supervised method.

    $ python pretrain.py --func pretrain --model_type nodes
    $ python pretrain.py --func pretrain --model_type edges
    $ python pretrain.py --func pretrain --model_type gs

Then, train the theorem prediction network:

    $ python train.py --func train --nodes_model nodes_model.pth --edges_model edges_model.pth --gs_model gs_model.pth --use_hypertree true

Pretraining log information will be saved in `data/log/xxx_pretrain_log.json` and `data/log/xxx_pretrain`. Training log
information will be saved in `data/log/train_log.json` and `data/log/train`.

## Testing

### Testing Pretrained

We first test pretrained nodes model, edges model and gs model:

    $ python pretrain.py --func test --model_type nodes --model_name nodes_model.pth
    $ python pretrain.py --func test --model_type edges --model_name edges_model.pth
    $ python pretrain.py --func test --model_type gs --model_name gs_model.pth

### Testing Step-wised Prediction

Test theorem prediction model:

    $ python train.py --func test --model_name predictor_model.pth

The test results of the model will be saved in `data/log/test`.

### Testing PA Cycle

Obtain the overall problem-solving success rate:

    $ python agent.py --model_name predictor_model.pth --use_hypertree true

## Results

You can obtain the figure or tabular data in the paper using the following command:

    $ python utils.py --func evaluate

### An overview of the problem-solving success rates

| Method         | Strategy      | Timeout | Solved | Unsolved | Timeout |
|----------------|---------------|---------|--------|----------|---------|
| Forward Search | Breadth-First | 30      | 11.46  | 3.87     | 84.67   |
| Forward Search | Depth-First   | 30      | 15.40  | 5.44     | 79.15   |
| Forward Search | Random        | 30      | 17.91  | 5.95     | 76.15   |
| Forward Search | Random Beam   | 30      | 13.90  | 11.82    | 74.28   |
| HyperGNet      | Normal Beam   | 30      | 62.18  | 28.15    | 9.67    |
| HyperGNet      | Greedy Beam   | 30      | 75.00  | 0.50     | 24.50   |
| HyperGNet      | Greedy Beam   | 600     | 85.53  | 2.22     | 12.25   |

### Details of the problem-solving success rates

| Method         | Strategy      | Timeout | Total | L1    | L2    | L3    | L4    | L5    | L6    |
|----------------|---------------|---------|-------|-------|-------|-------|-------|-------|-------|
| Forward Search | Breadth-First | 30      | 11.46 | 26.56 | 8.38  | 0.39  | 0.00  | 0.00  | 0.00  |
| Forward Search | Depth-First   | 30      | 15.40 | 31.54 | 15.41 | 1.57  | 0.60  | 0.00  | 1.69  |
| Forward Search | Random        | 30      | 17.91 | 37.97 | 14.59 | 2.76  | 2.99  | 0.00  | 1.69  |
| Forward Search | Random Beam   | 30      | 13.90 | 32.57 | 8.65  | 1.18  | 1.20  | 0.00  | 0.00  |
| HyperGNet      | Normal Beam   | 30      | 62.18 | 82.57 | 65.14 | 51.57 | 46.71 | 20.31 | 11.86 |
| HyperGNet      | Greedy Beam   | 30      | 75.00 | 91.49 | 80.27 | 68.11 | 61.08 | 29.69 | 25.42 |
| HyperGNet      | Greedy Beam   | 600     | 85.53 | 95.44 | 89.46 | 84.25 | 77.84 | 50.00 | 45.76 |

### Ablation study

| Method         | Beam Size | Step-wised Acc (%) | Overall Acc (%) | Avg Time (s) | Avg Step |
|----------------|-----------|--------------------|-----------------|--------------|----------|
| HyperGNet      | 1         | 63.03              | 30.23           | 0.96         | 2.62     |
| HyperGNet      | 3         | 82.05              | 53.30           | 3.05         | 3.31     |
| HyperGNet      | 5         | 87.65              | 62.18           | 5.47         | 3.46     |
| -w/o Pretrain  | 1         | 62.80              | 27.15           | 0.86         | 2.57     |
| -w/o Pretrain  | 3         | 82.86              | 48.21           | 2.70         | 3.33     |
| -w/o Pretrain  | 5         | 88.66              | 57.95           | 4.86         | 3.50     |
| -w/o Hypertree | 1         | 62.48              | 29.66           | 0.80         | 2.55     |
| -w/o Hypertree | 3         | 83.08              | 51.29           | 2.80         | 3.28     |
| -w/o Hypertree | 5         | 89.24              | 60.67           | 4.73         | 3.38     |

## Acknowledge

This project is maintained by
[FormalGeo Development Team](https://formalgeo.github.io/)
and Supported by
[Geometric Cognitive Reasoning Group of Shanghai University (GCRG, SHU)](https://euclidesprobationem.github.io/).  
Please contact with the author (xiaokaizhang1999@163.com) if you encounter any issues.

## Citation

coming soon...
