# In[24]:


#!/usr/bin/env python
# coding: utf-8

# In[1]: 

import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


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
parser.add_argument('--base_model', type=str, default='GCN', help='propagation model for encoder',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--if_smoothed', action='store_true', default=False)
parser.add_argument('--encoder_model', type=str, default='Grace', help='propagation model for encoder',
                    choices=['Grace','GraphCL'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
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
parser.add_argument('--cl_num_proj_hidden', type=int, default=128)
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
parser.add_argument('--attack', type=str, default='nettack',
                    choices=['nettack','random_attack'],)
parser.add_argument('--select_target_ratio', type=float, default=0.1,
                    help="The number of selected target test nodes for targeted attack")
# Randomized Smoothing
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")

# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/',                         name=args.dataset,                        transform=transform)
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

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
# from utils import get_split
# # data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
idx_train = data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
idx_clean_test = data.test_mask.nonzero().flatten()

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()


# In[25]:
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


# In[26]:


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

data = data.to(device)
num_class = int(data.y.max()+1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=1)

accs = []
for seed in seeds:
    # Construct and train encoder
    model = model_construct(args,args.encoder_model,data,device)
    model.fit(data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
    # Evaluation 


# In[27]:
from deeprobust.graph.data import Dataset, Pyg2Dpr, Dpr2Pyg
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

# Evaluation Metric: Robust Accuracy
# Node-level task
print(args.attack)
if((args.dataset=='Cora') or (args.dataset=='Citeseer') or (args.dataset=='Pubmed') or (args.dataset=='Flickr') or (args.dataset=='ogbn-arxiv')):
    if(args.attack == 'nettack'):
        dpr_data = Pyg2Dpr(dataset)
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        features = csr_matrix(features)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        # Train surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
        surrogate = surrogate.to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        # Select target nodes
        np.random.seed(42)
        # idx = np.arange(0,adj.shape[0])
        idx = np.array(idx_clean_test.cpu())
        np.random.shuffle(idx)
        target_node_list = idx[:int(args.select_target_ratio*len(idx))]
        idx_perturn_test = torch.LongTensor(np.array(target_node_list)).to(device)
        # Conduct Attack
        # degrees = adj.sum(0).A1
        modified_adj = adj
        target_num = len(target_node_list)
        print('=== [Evasion] Attacking %s nodes respectively ===' % target_num)
        perturbation_sizes = list(range(1,6))
        for n_perturbation in perturbation_sizes:
            cnt = 0
            cl_cnt=0
            print('=== Perturbation Size %s ===' % n_perturbation)
            for target_node in tqdm(target_node_list):
                # n_perturbations = int(degrees[target_node])
                atk_model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
                atk_model = atk_model.to(device)
                atk_model.attack(features, modified_adj, labels, target_node, n_perturbation, verbose=False)
                modified_adj = atk_model.modified_adj
                modified_features = atk_model.modified_features
                acc = single_test(modified_adj, modified_features, target_node)
                print("GCN",acc)
                if acc == 0:
                    cnt += 1

                perturb_edge_index, perturb_edge_weight = from_scipy_sparse_matrix(modified_adj)
                perturb_edge_index, perturb_edge_weight = perturb_edge_index.to(device), perturb_edge_weight.to(device)
                
                perturb_edge_index,perturb_edge_weight = model.sample_noise(perturb_edge_index, perturb_edge_weight,idx_perturn_test)
                z = model(data.x, perturb_edge_index,perturb_edge_weight)
                idx_target = torch.LongTensor([target_node]).to(device)
                cl_acc = label_evaluation(z, data.y, idx_train, idx_target)
                print("CL",cl_acc)
                if cl_acc == 0:
                    cl_cnt += 1
                # break
                
            print('[GCN] misclassification rate : %s' % (cnt/target_num))
            print('[CL] misclassification rate : %s' % (cl_cnt/target_num))