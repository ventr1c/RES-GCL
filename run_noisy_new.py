#!/usr/bin/env python
# coding: utf-8

# # Initialization

# * implement unifyGNN+Contrastive
# * try different noisy and augmentation
# * try against with/without unnoticeable 

# In[5]:


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
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
parser.add_argument('--temperature', type=float,  default=0.5, help='Temperature')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=0,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='cluster_degree',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# GRACE setting
# parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--config', type=str, default="config.yaml")
## Contrasitve setting
parser.add_argument('--num_hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=128,
                    help='Number of hidden units in MLP.')
parser.add_argument('--cl_lr', type=float, default=0.0002)
parser.add_argument('--cl_num_hidden', type=int, default=128)
parser.add_argument('--cl_num_proj_hidden', type=int, default=128)
parser.add_argument('--cl_num_layers', type=int, default=2)
parser.add_argument('--cl_activation', type=str, default='relu')
parser.add_argument('--cl_base_model', type=str, default='GCNConv')
parser.add_argument('--cont_weight', type=float, default=0.01)
parser.add_argument('--add_edge_rate_1', type=float, default=0)
parser.add_argument('--add_edge_rate_2', type=float, default=0)
parser.add_argument('--drop_edge_rate_1', type=float, default=0)
parser.add_argument('--drop_edge_rate_2', type=float, default=0.5)
parser.add_argument('--drop_feat_rate_1', type=float, default=0)
parser.add_argument('--drop_feat_rate_2', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.4)
parser.add_argument('--cl_num_epochs', type=int, default=1000)
parser.add_argument('--cl_weight_decay', type=float, default=0.00001)
parser.add_argument('--cont_batch_size', type=int, default=0)
parser.add_argument('--noisy_level', type=float, default=0.2)

args = parser.parse_args()
# args = parser.parse_known_args()[0]
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

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
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
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()


# In[6]:


from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset

import os

import numpy as np
from scipy.linalg import expm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

# from seeds import development_seed
DATA_PATH = './data/'

def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))
        
        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_adj_matrix(data) -> np.ndarray:
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))


def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

class PPRDataset(InMemoryDataset):
    """
    Dataset preprocessed with GDC using PPR diffusion.
    Note that this implementations is not scalable
    since we directly invert the adjacency matrix.
    """
    def __init__(self,noisy_data,
                 name: str = 'Cora',
                 use_lcc: bool = True,
                 alpha: float = 0.1,
                 k: int = 16,
                 eps: float = None):
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps
        self.noisy_data = noisy_data

        super(PPRDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        # base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(self.noisy_data)
        # obtain exact PPR matrix
        ppr_matrix = get_ppr_matrix(adj_matrix,
                                        alpha=self.alpha)

        if self.k:
            print(f'Selecting top {self.k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            print(f'Selecting edges with weight greater than {self.eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(ppr_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(ppr_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(
            x=self.noisy_data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=self.noisy_data.y,
            train_mask=torch.zeros(self.noisy_data.train_mask.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(self.noisy_data.test_mask.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(self.noisy_data.val_mask.size()[0], dtype=torch.bool)
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self) -> str:
        return f'{self.name}_ppr_alpha={self.alpha}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}'


# # Unify Contrastive GNN

# ## Structure noise

# ### Noisy

# In[7]:


import copy 
from model import UnifyModel, Encoder
from models.construct import model_construct
from construct_graph import *

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

data = data.to(device)
der1s = [0.2, 0.3, 0.4]
der2s = [0, 0.3, 0.4]

dfr1s = [0, 0.4, 0.5]
dfr2s = [0.2, 0.3, 0.4]
noisy_levels = [0.2,0.3,0.1]
for noisy_level in noisy_levels:
    for der1 in der1s:
        for der2 in der2s:
            for dfr1 in dfr1s:
                for dfr2 in dfr2s:
                    print("noisy_level:{},der1:{},der2:{},dfr1:{},dfr2:{}".format(noisy_level,der1,der2,dfr1,dfr2))
                    args.drop_edge_rate_1 = der1
                    args.drop_edge_rate_2 = der2
                    args.drop_edge_rate_1 = dfr1
                    args.drop_edge_rate_1 = dfr2
                    args.noisy_level = noisy_level

                    num_class = int(data.y.max()+1)
                    # args.cl_activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
                    # args.cl_base_model = ({'GCNConv': GCNConv})[config['base_model']]

                    noisy_data = construct_noisy_graph(data,perturb_ratio=args.noisy_level,mode='random_noise')
                    noisy_data = noisy_data.to(device)

                    # diff_dataset = PPRDataset(noisy_data,args.dataset)
                    # diff_noisy_data = diff_dataset.data.to(device)

                    seen_node_idx = (torch.bitwise_not(data.test_mask)).nonzero().flatten()
                    idx_overall_test = (torch.bitwise_not(data.train_mask)&torch.bitwise_not(data.val_mask)).nonzero().flatten()



                    final_cl_acc_noisy = []
                    final_gnn_acc_noisy = []
                    print("=== Noisy graph ===")
                    rs = np.random.RandomState(args.seed)
                    seeds = rs.randint(1000,size=3)
                    for seed in seeds:
                        np.random.seed(seed)
                        # torch.manual_seed(seed)
                        # torch.cuda.manual_seed(seed)
                        '''Transductive'''
                        encoder = Encoder(dataset.num_features, args.cl_num_hidden, args.cl_activation,
                                                base_model=args.cl_base_model, k=args.cl_num_layers).to(device)
                        # print(args, encoder, args.cl_num_hidden, args.cl_num_proj_hidden, num_class, args.tau, args.cl_lr, args.cl_weight_decay, device)
                        # model = UnifyModel(args, encoder, args.cl_num_hidden, args.cl_num_proj_hidden, num_class, args.tau, lr=args.cl_lr, weight_decay=args.cl_weight_decay, device=device).to(device)
                        model = UnifyModel(args, encoder, args.num_hidden, args.num_proj_hidden, args.cl_num_proj_hidden, num_class, args.tau, lr=args.cl_lr, weight_decay=args.cl_weight_decay, device=device).to(device)
                        # model = UnifyModel(args, encoder, args.cl_num_hidden, args.cl_num_proj_hidden, num_class, args.tau, lr=args.cl_lr, weight_decay=weight_decay, device=device,data1=noisy_data,data2=diff_noisy_data).to(device)
                        model.fit(args, noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight,noisy_data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,cont_iters=args.cl_num_epochs,seen_node_idx=None)
                        # model.fit_1(args, noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight,noisy_data.y,idx_train,idx_val=idx_val,train_iters=num_epochs,cont_iters=num_epochs,seen_node_idx=None)
                        # x, edge_index,edge_weight,labels,idx_train,idx_val=None,cont_iters=None,train_iters=200,seen_node_idx = None
                        acc_cl = model.test(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight,noisy_data.y,idx_clean_test)
                        print("Accuracy of GNN+CL: {}".format(acc_cl))
                        final_cl_acc_noisy.append(acc_cl)
                        gnn_model = model_construct(args,'GCN',noisy_data,device)
                        gnn_model.fit(noisy_data.x, noisy_data.edge_index, None, noisy_data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
                        clean_acc = gnn_model.test(noisy_data.x,noisy_data.edge_index,noisy_data.edge_weight,noisy_data.y,idx_overall_test)
                        print(clean_acc)
                        final_gnn_acc_noisy.append(clean_acc)


                    print('The final CL Acc:{:.5f}, {:.5f}, The final GNN Acc:{:.5f}, {:.5f}'            .format(np.average(final_cl_acc_noisy),np.std(final_cl_acc_noisy),np.average(final_gnn_acc_noisy),np.std(final_gnn_acc_noisy)))


                    # ### Raw

                    # In[10]:


                    import copy 
                    from model import UnifyModel
                    from models.construct import model_construct


                    import os.path as osp
                    import random
                    from time import perf_counter as t
                    import yaml
                    from yaml import SafeLoader

                    import torch
                    import torch_geometric.transforms as T
                    from model import Encoder, Model, drop_feature

                    data = data.to(device)
                    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
                    # num_epochs = config['num_epochs']
                    # args.cl_lr = config['args.cl_lr']
                    # weight_decay = config['weight_decay']
                    # args.seed = config['seed']
                    # args.cont_batch_size = config['cont_batch_size']
                    # args.cont_weight = config['cont_weight']
                    # args.add_edge_rate_1 = config['add_edge_rate_1']
                    # args.add_edge_rate_2 = config['add_edge_rate_2']
                    # args.drop_edge_rate_1 = config['drop_edge_rate_1']
                    # args.drop_edge_rate_2 = config['drop_edge_rate_2']
                    # args.drop_feat_rate_1 = config['drop_feature_rate_1']
                    # args.drop_feat_rate_2 = config['drop_feature_rate_2']
                    # args.add_edge_rate_1 = 0
                    # args.add_edge_rate_2 = 0
                    # args.drop_edge_rate_1 = 0.3
                    # args.drop_edge_rate_2 = 0.5
                    # args.drop_feat_rate_1 = 0.4
                    # args.drop_feat_rate_2 = 0.4
                    num_class = int(data.y.max()+1)

                    # noisy_data = construct_noisy_graph(data,perturb_ratio=0.25,mode='random_noise')
                    # noisy_data = noisy_data.to(device)
                    seen_node_idx = (torch.bitwise_not(data.test_mask)).nonzero().flatten()
                    idx_overall_test = (torch.bitwise_not(data.train_mask)&torch.bitwise_not(data.val_mask)).nonzero().flatten()


                    final_cl_acc = []
                    final_gnn_acc = []
                    print("=== Raw graph ===")
                    rs = np.random.RandomState(args.seed)
                    seeds = rs.randint(1000,size=3)
                    for seed in seeds:
                        np.random.seed(seed)
                        # torch.manual_seed(seed)
                        # torch.cuda.manual_seed(seed)
                        '''Transductive'''
                        encoder = Encoder(dataset.num_features, args.cl_num_hidden, args.cl_activation,
                                                base_model=args.cl_base_model, k=args.cl_num_layers).to(device)
                        model = UnifyModel(args, encoder, args.num_hidden, args.num_proj_hidden, args.cl_num_proj_hidden, num_class, args.tau, lr=args.cl_lr, weight_decay=args.cl_weight_decay, device=device).to(device)
                        # model = UnifyModel(args, encoder, args.cl_num_hidden, args.cl_num_proj_hidden, num_class, args.tau, lr=args.cl_lr, weight_decay=args.cl_weight_decay, device=None).to(device)
                        model.fit(args, data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,cont_iters=args.cl_num_epochs,seen_node_idx=None)
                        # x, edge_index,edge_weight,labels,idx_train,idx_val=None,cont_iters=None,train_iters=200,seen_node_idx = None
                        acc_cl = model.test(data.x, data.edge_index,data.edge_weight,data.y,idx_clean_test)
                        print("Accuracy of GNN+CL: {}".format(acc_cl))
                        final_cl_acc.append(acc_cl)
                        gnn_model = model_construct(args,'GCN',data,device)
                        gnn_model.fit(data.x, data.edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
                        clean_acc = gnn_model.test(data.x,data.edge_index,data.edge_weight,data.y,idx_overall_test)
                        print(clean_acc)
                        final_gnn_acc.append(clean_acc)


                    print('The final CL Acc:{:.5f}, {:.5f}, The final GNN Acc:{:.5f}, {:.5f}'            .format(np.average(final_cl_acc),np.std(final_cl_acc),np.average(final_gnn_acc),np.std(final_gnn_acc)))
