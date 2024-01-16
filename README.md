# PGPS

[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/BitSecret/PGPS)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Survey](https://img.shields.io/badge/Survey-FormalGeo-blue)](https://github.com/FormalGeo/FormalGeo)

Human-like IMO-level Plane Geometry Problem Solving. This project combines
the [FormalGeo](https://github.com/FormalGeo/FormalGeo) with deep learning to achieve automatic solving of IMO-level
plane geometry problems.  
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

We use **FormalGeo** as the formal environment for solving geometric problems and generate training data using the *
*formalgeo7k_v1** dataset. Following the dataset's recommended division ratio and random seed, the dataset is split
into `training:validation:test=3:1:1`. Each problem-solving theorem sequence can be organized into a directed acyclic
graph (DAG). We randomly traverse this DAG and apply each step's theorem while obtaining the state information of the
problem.  
This process ultimately generated 20,571 training data entries (from 4,079 problems), 7,072 validation data entries (
from 1,370 problems), and 7,046 test data entries (from 1,372 problems). Each data entry can be viewed as a
5-tuple `(nodes, edges, edges_structural, goal, theorems)`.   
These data are saved in `231121/training_data`. View the generated data (problem 1):

    $ python symbolic_system.py --func show_training_data

View the statistical information of the generated data:

    $ python symbolic_system.py --func check

This will generate message about the length distribution and visual images of the training data in
the `231221/log/words_length` folder.  
If you want to regenerate the training data:

    $ python symbolic_system.py --func main

The training data will be regenerated using multiple processes. The log files will be saved
in `231221/log/gen_training_data_log.json`. Subsequently, the generated data need to be converted into a vector form
suitable for neural networks input.

    $ python symbolic_system.py --func make_onehot

## Training

Before starting the training, ensure that the training data `231121/training_data/train/one-hot.pk` has been generated.
If not, run `python symbolic_system.py --func make_onehot` to generate the data.  
We first pretrain the embedding networks for nodes and edges using a self-supervised method.

    $ python pretrain.py --func nodes
    $ python pretrain.py --func edges

Then, train the theorem prediction network:

    $ python train.py --func train --nodes_model your_best_nodes_model_name --edges_model your_best_edges_model_name

Pretraining and Training log information will be saved in `231221/log`.

## Testing

The test results of the model will be saved in `231221/log/testing`.

    $ python train.py --func test --predictor_model your_best_model_name

## Results

coming soon...

## Acknowledge

This project is maintained by
[FormalGeo Development Team](https://formalgeo.github.io/)
and Supported by
[Geometric Cognitive Reasoning Group of Shanghai University (GCRG, SHU)](https://euclidesprobationem.github.io/).  
Please contact with the author (xiaokaizhang1999@163.com) if you encounter any issues.

## Citation

A BibTeX entry for LaTeX users is:
> @misc{arxiv2023formalgeo,  
> title={FormalGeo: The First Step Toward Human-like IMO-level Geometric Automated Reasoning},  
> author={Xiaokai Zhang and Na Zhu and Yiming He and Jia Zou and Qike Huang and Xiaoxiao Jin and Yanjun Guo and Chenyang
> Mao and Zhe Zhu and Dengfeng Yue and Fangzhen Zhu and Yang Li and Yifan Wang and Yiwen Huang and Runan Wang and Cheng
> Qin and Zhenbing Zeng and Shaorong Xie and Xiangfeng Luo and Tuo Leng},  
> year={2023},  
> eprint={2310.18021},  
> archivePrefix={arXiv},  
> primaryClass={cs.AI}  
> }
