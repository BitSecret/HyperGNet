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

    $ cd tests
    $ python test.py

Download datasets and initialize the project:

    $ cd src/pgps
    $ python utils.py

## Run

Enter the code path:

    $ cd src/pgps

### Check Our Results

### Run Your Own

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
