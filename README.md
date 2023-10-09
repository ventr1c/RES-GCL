# Certifiably Robust Graph Contrastive Learning
An official PyTorch implementation of "Certifiably Robust Graph Contrastive Learning" (NeurIPS 2023). [[paper]](https://arxiv.org/abs/2310.03312)
```
@inproceedings{lin2023certifiably,
  title={Certifiably Robust Graph Contrastive Learning},
  author={Lin, Minhua and Xiao, Teng and Dai, Enyan and Zhang, Xiang and Wang, Suhang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
}
```
## Content
- [Certifiably Robust Graph Contrastive Learning](#certifiably-robust-graph-contrastive-learning)
  - [Content](#content)
  - [1. Overview](#1-overviews)
  - [2. Requirements](#2-requirements)
  - [3. RES](#3-res)
    - [Abstract](#abstract)
    - [Reproduce the Results](#reproduce-the-results)
  - [4. Dataset](#5-dataset)

## Running the code
For instance, to check the performance of our RES-GRACE on clean Cora graph, run the following code:

```
python run_smooth_node.py --if_smoothed --encoder_model GRACE --dataset Cora --attack none
```

For PRBCD-perturbed Cora graph, run the following code:

```
python run_smooth_node.py --if_smoothed --encoder_model GRACE --dataset Cora --attack PRBCD
```

## 1. Overview
* `./models`: This directory contains the model of RES.
* `./data`: The directory contains the original datasets used in the experiments
* `./scripts`: It contains the scripts to reproduce the major reuslts of our paper.
* `./eval.py`: The program to evaluate the performance of GCL method in downstream tasks
* `./generate_prbcd_attack.py`: The program to run PRBCD attack from Geisler, Simon, et al. ["Robustness of Graph Neural Networks at Scale"](https://arxiv.org/abs/2110.14038).
* `./run_certify_graph.py`: The program to run RES-GCL to calculate certified accuracy in graph classification setting.
* `./run_certify_node.py`: The program to run RES-GCL to calculate certified accuracy in node classification setting.
* `./run_smooth_graph.py`: The program to run RES-GCL to calculate robust accuracy in graph classification setting.
* `./run_smooth_node.py`: The program to run RES-GCL to calculate robust accuracy in node classification setting.

## 2. Requirements
```
python==3.8.13
torch==1.12.1
torch-geometric==2.1.0
numpy==1.22.4
scipy==1.7.3
scikit-learn==1.1.1
ogb==1.2.1
deeprobust==0.2.8
PyGCL==0.1.2
```
## 3. RES

### Abstract
Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model.

### Reproduce the Results
To reproduce the performance reported in the paper, you can check and run the bash file:
```
bash scripts\scripts_graph.sh
bash scripts\scripts_node.sh
```

## 4. Dataset
The experiments are conducted on four node-level public real-world datasets, i.e., Cora, Pubmed, Coauthor-Physics and OGB-Arxiv, and three graph-level real-world datasets, i.e., MUTAG, PROTEINS and OGB-molhiv. These datasets can be automatically downloaded to `./data` through torch-geometric API.

