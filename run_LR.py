#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8


# In[31]:

import yaml
from yaml import SafeLoader
import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Amazon, Coauthor, WikiCS


# from torch_geometric.loader import DataLoader
# from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp

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
parser.add_argument('--encoder_model', type=str, default='Grace', help='propagation model for encoder',
                    choices=['Grace','GraphCL','BGRL','DGI','GAE','LR'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','Flickr','ogbn-arxiv','Computers','Photo','CS','Physics','WikiCS'])
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
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--cl_num_epochs', type=int, default=200)
parser.add_argument('--cl_weight_decay', type=float, default=1e-5)
parser.add_argument('--cont_batch_size', type=int, default=0)
parser.add_argument('--noisy_level', type=float, default=0.3)
parser.add_argument('--clf_weight', type=float, default=1)
parser.add_argument('--inv_weight', type=float, default=1)
# parser.add_argument('--select_thrh', type=float, default=0.8)

# Attack
parser.add_argument('--attack', type=str, default='none',
                    choices=['nettack','random','none'],)
parser.add_argument('--select_target_ratio', type=float, default=0.1,
                    help="The number of selected target test nodes for targeted attack")
# Randomized Smoothing
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")
parser.add_argument('--num_sample', default=20, type=int,
                    help="the number of noisy samples to calculate the bounded probability")
parser.add_argument('--if_keep_structure1', action='store_true', default=False)
parser.add_argument('--if_ignore_structure2', action='store_true', default=False)
parser.add_argument('--sample_way', type=str, default='random_drop',
                    choices=['random_drop','keep_structure'])
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

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/',                         name=args.dataset,                        transform=transform)
elif(args.dataset == 'Computers' or args.dataset == 'Photo'):
    dataset = Amazon(root='./data/', name=args.dataset,                    transform=transform)
elif(args.dataset == 'CS' or args.dataset == 'Physics'):
    dataset = Coauthor(root='./data/', name=args.dataset,                    transform=transform)  
elif(args.dataset == 'WikiCS'):
    dataset = WikiCS(root='./data/WikiCS', transform=transform)    
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/',                     transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/',                     transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)


# from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
# dr_data = Dataset(root='/tmp/', name='pubmed') # load clean graph
# data = Dpr2Pyg(dr_data)
# data = data[0].to(device)

# we build our own train test split 
if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'CS' or args.dataset == 'Physics'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
# elif(args.dataset == 'WikiCS'):
#     data.train_mask = data.train_mask[:,0]
#     data.val_mask = data.val_mask[:,0]
#     data.stopping_mask = data.stopping_mask[:,0]

# In[14]:

print(data)
from utils import get_split
if(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'Physics' or args.dataset == 'ogbn-arxiv'):
    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
# data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
idx_train = data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
idx_clean_test = data.test_mask.nonzero().flatten()

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
# unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()



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

from eval import label_evaluation

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



from torch_geometric.utils import degree

data = data.to(device)
num_class = int(data.y.max()+1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=args.num_repeat)

mean_degree = int(torch.mean(degree(data.edge_index[0])).item())+2
atk_budget = min(2*mean_degree,20)
accs = []
betas = [-1]


accuracys = {}
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("seed {}".format(seed))
    # Construct and train encoder
    model = model_construct(args,args.encoder_model,data,device)

    # classes, probs = smooth_model.certify_Ber(n0=10,n=100,alpha=0.01,idx_train=idx_train, idx_test = idx_clean_test)
    import construct_graph
    import copy
    perturbation_sizes = [0]
    for n_perturbation in perturbation_sizes:
        # print("Perturbation Size:{}".format(n_perturbation))
        noisy_data = copy.deepcopy(data)
        if(n_perturbation > 0):
            for idx in idx_clean_test:
                noisy_data = construct_graph.generate_node_noisy(args,noisy_data,idx,n_perturbation,device)
                noisy_data = noisy_data.to(device)
        model.eval()
        z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
        acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)

        print("Accuracy:",acc)
        accuracys[n_perturbation].append(acc)


for n_perturbation in perturbation_sizes:
    mean_acc =  np.mean(accuracys[n_perturbation])  
    std_acc =  np.std(accuracys[n_perturbation])     
    print("Ptb size:{} Accuracy:{:.4f}+-{:.4f}".format(n_perturbation,mean_acc,std_acc))

# In[ ]:


