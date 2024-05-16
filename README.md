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

    $ git clone --depth 1 https://github.com/BitSecret/HyperGNet.git
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

Download the trained model
through [Google Drive](https://drive.google.com/file/d/1XELvToJji-AIJDZaVUSIAwdsThvAkBOd/view?usp=sharing)
or [Baidu NetDisk](https://pan.baidu.com/s/1HER9YGf_L-0gJMq5Kfioow?pwd=ddjb) and extract them
into `src/pgps/trained_model`.

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

### Details of the problem-solving success rates

| Method       | Strategy             | Timeout | Total | L1    | L2    | L3    | L4    | L5    | L6    |
|--------------|----------------------|---------|-------|-------|-------|-------|-------|-------|-------|
| FW           | Random Search        | 600     | 39.71 | 59.24 | 40.04 | 33.68 | 16.38 | 5.43  | 4.79  |
| BW           | Breadth-First Search | 600     | 35.44 | 67.22 | 33.72 | 11.15 | 6.67  | 6.07  | 1.03  |
| Inter-GPS    | Beam Search          | 600     | 40.76 | 63.90 | 36.49 | 27.95 | 23.95 | 12.50 | 11.86 |
| FGeo-TP (FW) | Random Search        | 600     | 68.76 | 84.68 | 70.78 | 66.51 | 51.09 | 30.03 | 25.09 |
| FGeo-TP (BW) | Breadth-First Search | 600     | 80.12 | 96.55 | 85.60 | 74.36 | 59.59 | 45.69 | 28.18 |
| FGeo-TP (BW) | Depth-First Search   | 600     | 79.56 | 96.18 | 84.18 | 73.72 | 60.32 | 45.05 | 28.52 |
| FGeo-TP (BW) | Random Search        | 600     | 80.86 | 96.43 | 85.44 | 76.12 | 62.26 | 48.88 | 29.55 |
| FGeo-TP (BW) | Beam Search          | 600     | 79.06 | 96.10 | 84.55 | 72.92 | 58.37 | 43.45 | 25.43 |
| FGeo-DRL     | Beam Search          | 1200    | 86.40 | 97.65 | 94.21 | 85.87 | 70.45 | 46.81 | 32.18 |
| HyperGNet    | Normal Beam Search   | 30      | 62.18 | 82.57 | 65.14 | 51.57 | 46.71 | 20.31 | 11.86 |
| HyperGNet    | Greedy Beam Search   | 30      | 79.58 | 94.61 | 84.32 | 75.98 | 67.66 | 32.81 | 27.12 |
| HyperGNet    | Greedy Beam Search   | 600     | 85.53 | 95.44 | 89.46 | 84.25 | 77.84 | 50.00 | 45.76 |

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
Please contact with [Xiaokai Zhang](https://bitsecret.github.io/) if you encounter any issues.

## Citation

To cite HyperGNet in publications use:
> Zhang X, Zhu N, He Y, et al. FGeo-HyperGNet: Geometric Problem Solving Integrating Formal Symbolic System and
> Hypergraph
> Neural Network[J]. arXiv preprint arXiv:2402.11461, 2024.

A BibTeX entry for LaTeX users is:
> @misc{zhang2024fgeohypergnet,  
> title={FGeo-HyperGNet: Geometric Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network},  
> author={Xiaokai Zhang and Na Zhu and Yiming He and Jia Zou and Cheng Qin and Yang Li and Zhenbing Zeng and Tuo
> Leng},  
> year={2024},  
> eprint={2402.11461},  
> primaryClass={cs.AI}  
> }
