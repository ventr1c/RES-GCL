#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8


# In[31]:

import yaml
from yaml import SafeLoader
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import TUDataset

from torch_geometric.loader import DataLoader
# from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp
import utils

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--base_model', type=str, default='GCN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--if_smoothed', action='store_true', default=False)
parser.add_argument('--encoder_model', type=str, default='BGRL-G2L', help='propagation model for encoder',
                    choices=['BGRL-G2L','GraphCL','Node2vec'])
parser.add_argument('--dataset', type=str, default='ogbg-molhiv', 
                    help='Dataset',
                    choices=['PROTEINS','MUTAG','COLLAB','ENZYMES','ogbg-molhiv'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units of backdoor model.')
parser.add_argument('--num_hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=128,
                    help='Number of hidden units in MLP.')
# parser.add_argument('--thrd', type=float, default=0.5)
# parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=2,
                    help="Threshold of prunning edges")
# Contrastive Learning setting
parser.add_argument('--config', type=str, default="config.yaml")
parser.add_argument('--cl_lr', type=float, default=0.0005)
# parser.add_argument('--cl_num_proj_hidden', type=int, default=128)
parser.add_argument('--cl_num_layers', type=int, default=2)
parser.add_argument('--cl_activation', type=str, default='relu')
parser.add_argument('--cl_base_model', type=str, default='GCNConv')
parser.add_argument('--cont_weight', type=float, default=1)
parser.add_argument('--add_edge_rate_1', type=float, default=0)
parser.add_argument('--add_edge_rate_2', type=float, default=0)
parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
parser.add_argument('--drop_edge_rate_2', type=float, default=0)
parser.add_argument('--drop_feat_rate_1', type=float, default=0.3)
parser.add_argument('--drop_feat_rate_2', type=float, default=0.2)
parser.add_argument('--drop_node_rate_1', type=float, default=0.1)
parser.add_argument('--drop_node_rate_2', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--cl_num_epochs', type=int, default=200)
parser.add_argument('--cl_weight_decay', type=float, default=1e-5)
parser.add_argument('--cont_batch_size', type=int, default=0)

parser.add_argument('--batch_size', default=1024, type=int,
                    help="batch_size of graph dataset")
parser.add_argument('--walk_length_2', default=10, type=int,
                    help="")
# parser.add_argument('--select_thrh', type=float, default=0.8)

# Attack
parser.add_argument('--attack', type=str, default='none',
                    choices=['nettack','random','none', 'PRBCD'],)
parser.add_argument('--select_target_ratio', type=float, default=0.15,
                    help="The number of selected target test nodes for targeted attack")
# Randomized Smoothing
# Randomized Smoothing
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")
parser.add_argument('--num_sample', default=20, type=int,
                    help="the number of noisy samples to calculate the bounded probability")
parser.add_argument('--n0', default=20, type=int,
                    help="the number of noisy samples to calculate the bounded probability")
parser.add_argument('--n', default=200, type=int,
                    help="the number of noisy samples to estimate")
parser.add_argument('--alpha', default=0.01, type=float,
                    help="the number of noisy samples to estimate")

parser.add_argument('--if_keep_structure1', action='store_true', default=False)
parser.add_argument('--if_ignore_structure2', action='store_true', default=False)
parser.add_argument('--sample_way', type=str, default='random_drop',
                    choices=['random_drop','keep_structure'])
# Prune
parser.add_argument('--prune_thrh', type=float, default=0.03,
                    help="the threshold for pruning dissimilar edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

print(args)
np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
print(args)


# In[13]:
# In[13]:


from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])


if(args.dataset == 'PROTEINS' or args.dataset == 'MUTAG' or args.dataset == 'COLLAB' or args.dataset == 'ENZYMES'):
    dataset = TUDataset(root='./data/', name=args.dataset, transform=None,use_node_attr = True)
elif(args.dataset == 'ogbg-molhiv'):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.data import DataLoader
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root='./data/') 
    # split_idx = dataset.get_idx_split() 
    
# data = dataset[0].to(device)

from torch_geometric.utils import to_undirected
for data in dataset:
    data.edge_index = to_undirected(data.edge_index)
dataloader = DataLoader(dataset, batch_size=args.batch_size)
if(args.dataset=='ogbg-molhiv'):
    for data in dataloader:
        data.x = data.x.float()

from GCL.eval import get_split

# split = get_split(num_samples=len(dataset), train_ratio=0.8, test_ratio=0.1)
split = utils.get_split_self(num_samples=len(dataset), train_ratio=0.8, test_ratio=0.1,device=device)


# In[28]:

config_path = "./config/config_Smooth_{}.yaml".format(args.encoder_model)   
# if(args.if_smoothed == True):
#     config_path = "./config/config_Smooth_{}.yaml".format(args.encoder_model)    
# else:
#     config_path = "./config/config_{}.yaml".format(args.encoder_model)
config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset]

args.drop_edge_rate_1 = config['drop_edge_rate_1']
args.drop_edge_rate_2 = config['drop_edge_rate_2']
args.drop_feat_rate_1 = config['drop_feat_rate_1']
args.drop_feat_rate_2 = config['drop_feat_rate_2']
if(args.encoder_model == 'GraphCL'):
    args.drop_node_rate_1 = config['drop_node_rate_1']
    args.drop_node_rate_2 = config['drop_node_rate_2']
args.tau = config['tau']
args.cl_lr = config['cl_lr']
args.weight_decay = config['weight_decay']
args.cl_num_epochs = config['cl_num_epochs']
args.num_hidden = config['num_hidden']
args.num_proj_hidden = config['num_proj_hidden']

print(args)
# In[29]:


import copy 
from models.construct import model_construct_global
from construct_graph import *
from models.GCN_CL import GCN_Encoder, Grace

import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

from eval import label_classification,label_evaluation,label_classification_origin, smoothed_linear_evaluation

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
# from deeprobust.graph.defense import GCN
# from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
# from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

from deeprobust.graph.data import Dataset, Pyg2Dpr, Dpr2Pyg
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack

# from models.random_smooth import sample_noise_all_dense,sample_noise_1by1_dense, _lower_confidence_bound

num_class = int(data.y.max()+1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=args.num_repeat)

accs = []
# betas = [0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.9,0.005,]
betas = [0.05,0.1,0.9,0.001]
for beta in betas:
    args.prob = beta
    print("beta {}".format(beta))
    total_cer_accs = [] 
    for seed in seeds:
        cer_accs = []
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("seed {}".format(seed))
        # Construct and train encoder
        # model = model_construct(args,args.encoder_model,data,device)
        # model.fit(data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        # model = model_construct(args,args.encoder_model,data,device)
        model = model_construct_global(args,args.encoder_model, dataset, device)
        # model.fit(data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        model.fit(dataloader, train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        # Evaluation 
        import models.random_smooth_graph as random_smooth_graph
        # num_class = int(data.y.max()+1)
        num_class = dataset.num_classes
        smooth_model = random_smooth_graph.Smooth_Ber(args, model, num_class, args.prob, data.x, if_node_level=True,device=device)   
# for beta in betas:
#     args.prob = beta
#     print("beta {}".format(beta))
#     total_cer_accs = [] 
#     for seed in seeds:
#         cer_accs = []

#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         print("seed {}".format(seed))
#         # Construct and train encoder
#         model = model_construct(args,args.encoder_model,data,device)
#         model.fit(data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        # Evaluation 
        k_range = list(range(0,21))
        total_Deltas, probs, classes = smooth_model.calculate_certify_K_approx(dataset=dataset, dataloader=dataloader, n0=args.n0, n=args.n, alpha=args.alpha, idx_train=split['train'], idx_test=split['test'], k_range=k_range)
        probs = np.array(probs)
        for i in k_range:
            cert_acc = ((2*probs - 1)>2 * total_Deltas[i]).sum()
            print(i,cert_acc)
            cer_accs.append(cert_acc)
        total_cer_accs.append(cer_accs)
        print("Seed:{} Cert Acc:{}".format(seed, [cer_acc/len(split['test']) for cer_acc in cer_accs]))
    total_cer_accs = np.array(total_cer_accs)
    mean_cer_accs = np.mean(total_cer_accs,axis=0)
    print("Beta:{} Mean Cert Acc:{}".format(beta,mean_cer_accs/len(split['test'])))
