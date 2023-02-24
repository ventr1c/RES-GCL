#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8


# In[31]:


import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI, TUDataset
from torch_geometric.loader import DataLoader

# from torch_geometric.loader import DataLoader
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
parser.add_argument('--base_model', type=str, default='GCN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--if_smoothed', action='store_true', default=False)
parser.add_argument('--encoder_model', type=str, default='Grace', help='propagation model for encoder',
                    choices=['Grace','GraphCL'])
parser.add_argument('--dataset', type=str, default='PROTEINS', 
                    help='Dataset',
                    choices=['PROTEINS','MUTAG'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units of GConv model.')
parser.add_argument('--num_hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=32,
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
parser.add_argument('--cl_lr', type=float, default=0.01)
# parser.add_argument('--cl_num_proj_hidden', type=int, default=128)
parser.add_argument('--cl_num_layers', type=int, default=2)
parser.add_argument('--cl_activation', type=str, default='relu')
parser.add_argument('--cl_base_model', type=str, default='GCNConv')
parser.add_argument('--cont_weight', type=float, default=1)
parser.add_argument('--add_edge_rate_1', type=float, default=0)
parser.add_argument('--add_edge_rate_2', type=float, default=0)
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1)
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1)
parser.add_argument('--drop_feat_rate_1', type=float, default=0.1)
parser.add_argument('--drop_feat_rate_2', type=float, default=0.1)
parser.add_argument('--drop_node_rate_1', type=float, default=0.1)
parser.add_argument('--drop_node_rate_2', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.2)
parser.add_argument('--cl_num_epochs', type=int, default=100)
parser.add_argument('--cl_weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', default=128, type=int,
                    help="batch_size of graph dataset")
parser.add_argument('--walk_length', default=10, type=int)
# parser.add_argument('--select_thrh', type=float, default=0.8)

# Attack
parser.add_argument('--attack', type=str, default='none',
                    choices=['nettack','random','none'],)
parser.add_argument('--select_target_ratio', type=float, default=0.1,
                    help="The number of selected target test nodes for targeted attack")
# Randomized Smoothing
parser.add_argument('--prob', default=1.0, type=float,
                    help="probability to keep the status for each binary entry")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
print(args)


# In[13]:


from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])


if(args.dataset == 'PROTEINS' or args.dataset == 'MUTAG'):
    dataset = TUDataset(root='./data/', name=args.dataset, transform=None,use_node_attr = True)

data = dataset[0].to(device)

from torch_geometric.utils import to_undirected
# for data in dataset:
#     data.edge_index = to_undirected(data.edge_index)
dataloader = DataLoader(dataset, batch_size=args.batch_size)

from GCL.eval import get_split

# split = get_split(num_samples=len(dataset), train_ratio=0.8, test_ratio=0.1)
split = utils.get_split_self(num_samples=len(dataset), train_ratio=0.8, test_ratio=0.1,device=device)

# if(args.dataset == 'ogbn-arxiv'):
#     nNode = data.x.shape[0]
#     setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
#     # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
#     data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.y = data.y.squeeze(1)
# # we build our own train test split 
# print(data)

# In[14]:


# # from utils import get_split
# # # data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
# idx_train = data.train_mask.nonzero().flatten()
# idx_val = data.val_mask.nonzero().flatten()
# idx_clean_test = data.test_mask.nonzero().flatten()

# from torch_geometric.utils import to_undirected
# from utils import subgraph
# data.edge_index = to_undirected(data.edge_index)
# train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
# mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
# # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
# unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()


# In[28]:


if(args.dataset == 'Cora'):
    args.drop_edge_rate_1 = 0.2
    args.drop_edge_rate_2 = 0.4
    args.drop_feat_rate_1 = 0.3
    args.drop_feat_rate_2 = 0.4
    args.tau = 0.1
    args.cl_lr = 0.0005
    args.weight_decay = 1e-5
    args.cl_num_epochs = 500
    args.num_hidden = 128
    args.hidden = 128
    args.num_proj_hidden = 128
elif(args.dataset == "Pubmed"):
    args.drop_edge_rate_1 = 0.4
    args.drop_edge_rate_2 = 0.1
    args.drop_feat_rate_1 = 0.0
    args.drop_feat_rate_2 = 0.2
    args.tau = 0.1
    args.cl_lr = 0.001
    args.weight_decay = 1e-5
    args.cl_num_epochs = 500
    args.num_hidden = 256
    args.hidden = 256
elif(args.dataset == "Citeseer"):
    args.drop_edge_rate_1 = 0.2
    args.drop_edge_rate_2 = 0.0
    args.drop_feat_rate_1 = 0.3
    args.drop_feat_rate_2 = 0.2
    args.tau = 0.1
    args.cl_lr = 0.001
    args.weight_decay = 1e-5
    args.cl_num_epochs = 500
    args.num_hidden = 256
    args.hidden = 256

# In[29]:


import copy 
from models.construct import model_construct
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

from eval import label_classification,label_evaluation,label_classification_origin

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

import GCL.augmentors as A
from models.random_smooth import sample_noise,sample_noise_1by1,sample_noise_all
from models.GraphCL import GConv, Encoder
import construct_graph

data = data.to(device)
num_class = int(data.y.max()+1)
input_dim = max(dataset.num_features, 1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)

accs = []
perturbation_sizes = list(range(1,6))
noisy_dataloaders = []
noisy_datasets = []
for n_perturbation in perturbation_sizes:
    print("Perturbation Size:{}".format(n_perturbation))
    # noisy_dataset = copy.deepcopy(dataset)
    print(n_perturbation)
    noisy_dataset = []
    if(len(noisy_datasets)==0):
        for data in dataset:
            noisy_data = construct_graph.generate_graph_noisy(args,data,1,device)
            noisy_dataset.append(noisy_data)
    else:
        for data in noisy_datasets[-1]:
            noisy_data = construct_graph.generate_graph_noisy(args,data,1,device)
            noisy_dataset.append(noisy_data)
    noisy_datasets.append(noisy_dataset)
    noisy_dataloader = DataLoader(noisy_dataset, batch_size=args.batch_size)
    noisy_dataloaders.append(noisy_dataloader)

for seed in seeds:
    # Construct and train encoder
    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=args.seed, walk_length=args.walk_length),
                           A.NodeDropping(pn=args.drop_edge_rate_2),
                           A.FeatureMasking(pf=args.drop_feat_rate_2),
                           A.EdgeRemoving(pe=args.drop_node_rate_2)], 1)

    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=2).to(device)
    model = Encoder(args = args, encoder=gconv, augmentor=(aug1, aug2), input_dim=input_dim, hidden_dim=args.num_hidden, lr=args.cl_lr, tau=args.tau,num_epoch = args.cl_num_epochs, if_smoothed = args.if_smoothed,device = device)
    model.fit(dataloader)
    test_result = model.test(dataloader,split)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    
# In[30]:
    # Evaluation Metric: Robust Accuracy
    # Node-level task
    print(args.attack)
    if(args.attack == 'random'):
        import construct_graph
        import copy
        perturbation_sizes = list(range(1,6))
        for n_perturbation in perturbation_sizes:
            print("Perturbation Size:{}".format(n_perturbation))
            # noisy_dataset = copy.deepcopy(dataset)
            print(n_perturbation)
            noisy_dataloader = noisy_dataloaders[n_perturbation-1]
            # noisy_dataset = []
            # for data in dataset:
            # # for i in range(len(noisy_dataset)):
            #     noisy_data = construct_graph.generate_graph_noisy(args,data,n_perturbation,device)
            #     # noisy_dataset[i] = noisy_data.to(device)
            #     noisy_dataset.append(noisy_data)
            # noisy_dataloader = DataLoader(noisy_dataset, batch_size=args.batch_size)

            # for i in range(3):        
            test_result = model.test(noisy_dataloader,split)
            # for i in range(len(noisy_dataset)):
            #     noisy_data = construct_graph.generate_graph_noisy(args,noisy_dataset[i],n_perturbation,device)
            #     # noisy_dataset[i] = noisy_data.to(device)
            #     # noisy_dataloader = DataLoader(noisy_dataset, batch_size=args.batch_size)
            #     test_result = model.single_test(noisy_data)
            print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
            
    # elif(args.attack == 'none'):
    #     model.eval()
    #     if(args.if_smoothed == True):
    #         rs_edge_index, rs_edge_weight = sample_noise_all(args, data.edge_index, data.edge_weight, device)
    #         z, _, _ = model(data.x, rs_edge_index, rs_edge_weight)
    #     else:
    #         z, _, _ = model(data.x, data.edge_index, data.edge_weight)
        # acc = label_evaluation(z, data.y, idx_train, idx_clean_test)
        # print("Accuracy:",acc)


# In[ ]:




