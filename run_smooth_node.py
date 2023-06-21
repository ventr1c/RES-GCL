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
                    choices=['Grace','GraphCL','BGRL','DGI','GAE','Node2vec','Grace-Jaccard','Ariel','Grace_batch'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','Flickr','ogbn-arxiv','Computers','Photo','CS','Physics'])
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
                    choices=['nettack','random','none', 'PRBCD','random_global'],)
parser.add_argument('--select_target_ratio', type=float, default=0.15,
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
# Prune
parser.add_argument('--prune_thrh', type=float, default=0.03,
                    help="the threshold for pruning dissimilar edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)


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
if(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'ogbn-arxiv' or args.dataset == 'Physics'):
    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
# data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
idx_train = data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
idx_clean_test = data.test_mask.nonzero().flatten()

print(idx_train.shape, idx_val.shape, idx_clean_test.shape)
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
# unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()

# In[28]:
if(args.if_smoothed == True):
    config_path = "./config/config_Smooth_{}.yaml".format(args.encoder_model)    
else:
    config_path = "./config/config_{}.yaml".format(args.encoder_model)
config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset]

args.drop_edge_rate_1 = config['drop_edge_rate_1']
args.drop_edge_rate_2 = config['drop_edge_rate_2']
args.drop_feat_rate_1 = config['drop_feat_rate_1']
args.drop_feat_rate_2 = config['drop_feat_rate_2']
args.tau = config['tau']
args.cl_lr = config['cl_lr']
args.weight_decay = config['weight_decay']
args.cl_num_epochs = config['cl_num_epochs']
args.num_hidden = config['num_hidden']
args.num_proj_hidden = config['num_proj_hidden']
if(args.encoder_model == 'Ariel'):
    args.eps = config['eps']
    args.alpha = config['alpha']
    args.beta = config['beta']
    args.lamb = config['lamb']
print(args)

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

from eval import label_classification,label_evaluation,lr_evaluation,label_classification_origin, smoothed_linear_evaluation

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

from models.random_smooth import sample_noise_all_dense,sample_noise_1by1_dense, _lower_confidence_bound

from torch_geometric.utils import degree

data = data.to(device)
num_class = int(data.y.max()+1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=args.num_repeat)

mean_degree = int(torch.mean(degree(data.edge_index[0])).item())+2
# atk_budget = min(2*mean_degree,20)
# atk_budget = min(2*mean_degree,20)
atk_budget = 20

accs = []
# betas = [0.1,0.2,0.3,0.4]
# betas = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# betas = [0.,0.1,0.3,0.5,0.7,0.9]
if(args.if_smoothed):
    # betas = [0.2,0.1,0.3,0.4,0.5,0.9,0.01,0.05,0.001,0.005]
    # betas = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.9,0.001,0.005]
    # betas = [0.1,0.2,0.3]
    betas = [0.5,0.4,0.3,0.2]
    # betas = [0.3,0.4,0.5,0.9,0.1,0.2]
    # betas = [0.2,0.4]
    # betas = [0.01]
else:
    betas = [-1]
for beta in betas:
    args.prob = beta
    print("beta {}".format(beta))
    # initialize 
    if(args.attack == 'nettack'):
        perturbation_sizes = list(range(0,6))
        # perturbation_sizes = [0]
        # perturbation_sizes = list(range(1,6))
        misclf_rates_cl = {}
        misclf_rates_gcn = {}
        for n_perturbation in perturbation_sizes:
            misclf_rates_cl[n_perturbation] = []
            misclf_rates_gcn[n_perturbation] = []
    elif(args.attack == 'random'):
        if(args.dataset == 'ogbn-arxiv'):
            atk_budget = 7
        elif(args.dataset == 'Cora'):
            atk_budget = 5
        elif(args.dataset == 'Pubmed'):
            atk_budget = 5
        elif(args.dataset == 'Computers'):
            atk_budget = 20
        perturbation_sizes = [0, atk_budget]
        # perturbation_sizes = [0]
        # perturbation_sizes = list(range(0,atk_budget+1))
        accuracys = {}
        for n_perturbation in perturbation_sizes:
            accuracys[n_perturbation] = []
    elif(args.attack == 'PRBCD'):
        # perturbation_sizes = list(range(0,atk_budget+1))
        # perturbation_sizes = [0, 0.05, 0.10,0.25, 0.5, 1.0]
        perturbation_sizes = [0, 0.10,]
        accuracys = {}
        for n_perturbation in perturbation_sizes:
            accuracys[n_perturbation] = []
    elif(args.attack == 'random_global'):
        # perturbation_sizes = list(range(0,atk_budget+1))
        # perturbation_sizes = [0, 0.05, 0.10,0.25, 0.5, 1.0]
        perturbation_sizes = [0, 0.10,]
        accuracys = {}
        for n_perturbation in perturbation_sizes:
            accuracys[n_perturbation] = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("seed {}".format(seed))
        # Construct and train encoder
        model = model_construct(args,args.encoder_model,data,device)
        # model.fit(data.x, data.edge_index,data.edge_weight,data.y,idx_train,idx_val=idx_val,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        model.fit(data.x, data.edge_index,data.edge_weight,data.y,train_iters=args.cl_num_epochs,seen_node_idx=None, verbose=True)
        # Evaluation 
    # In[30]:
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
        import models.random_smooth as random_smooth
        num_class = int(data.y.max()+1)
        smooth_model = random_smooth.Smooth_Ber(args, model, num_class, args.prob, data.x, if_node_level=True,device=device)
        # if((args.dataset=='Cora') or (args.dataset=='Citeseer') or (args.dataset=='Pubmed') or (args.dataset=='Flickr') or (args.dataset=='ogbn-arxiv')):
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
            # perturbation_sizes = list(range(0,6))
            for n_perturbation in perturbation_sizes:
                if(n_perturbation == 0):
                    model.eval()
                    if(args.if_smoothed):
                        _, cl_acc = smooth_model._sample_noise_ber(args.num_sample, data.edge_index, data.edge_weight, data.y, if_node_level = True, idx_test = idx_perturn_test, idx_train = idx_train)
                        # prediction = prediction_distribution.argmax(1)
                    else:
                        z = model(data.x, data.edge_index, data.edge_weight)
                        cl_acc = label_evaluation(z, data.y, idx_train, idx_perturn_test)
                    misclf_rates_cl[n_perturbation].append(1-cl_acc)
                    print('[CL] misclassification rate : %s' % (1-cl_acc))
                else:
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
                        perturb_edge_index, perturb_edge_weight = perturb_edge_index.to(device), perturb_edge_weight.to(torch.float).to(device)
                        idx_target = torch.LongTensor([target_node]).to(device)

                        model.eval()
                        if(args.if_smoothed):
                            _, cl_acc = smooth_model._sample_noise_ber(args.num_sample, perturb_edge_index, perturb_edge_weight, data.y, if_node_level = True, idx_test = idx_target, idx_train = idx_train)
                            # prediction = prediction_distribution.argmax(1)
                        else:
                            z = model(data.x, perturb_edge_index,perturb_edge_weight)
                            cl_acc = label_evaluation(z, data.y, idx_train, idx_target)
                        # if(args.if_smoothed):
                        #     # perturb_edge_index,perturb_edge_weight = sample_noise_1by1_dense(args,perturb_edge_index, perturb_edge_weight,idx_perturn_test, device)
                        #     perturb_edge_index,perturb_edge_weight = model.sample_noise(perturb_edge_index, perturb_edge_weight,idx_perturn_test)
                        #     # perturb_edge_index,perturb_edge_weight = model.sample_noise_1by1(perturb_edge_index, perturb_edge_weight,idx_perturn_test)
                        # z = model(data.x, perturb_edge_index,perturb_edge_weight)
                        # cl_acc = label_evaluation(z, data.y, idx_train, idx_target)
                        print("CL",cl_acc)
                        if cl_acc == 0:
                            cl_cnt += 1
                        # break
                    misclf_cl = cl_cnt/target_num
                    misclf_gcn = cnt/target_num

                    misclf_rates_cl[n_perturbation].append(misclf_cl)
                    misclf_rates_gcn[n_perturbation].append(misclf_gcn)
                    print('[GCN] misclassification rate : %s' % (cnt/target_num))
                    print('[CL] misclassification rate : %s' % (cl_cnt/target_num))

        elif(args.attack == 'random'):
            # classes, probs = smooth_model.certify_Ber(n0=10,n=100,alpha=0.01,idx_train=idx_train, idx_test = idx_clean_test)
            import construct_graph
            import copy
            # perturbation_sizes = list(range(0,atk_budget+1))
            for n_perturbation in perturbation_sizes:
                # print("Perturbation Size:{}".format(n_perturbation))
                model = model.to(device)
                noisy_data = copy.deepcopy(data).to(device)
                idxs = np.array(idx_clean_test.cpu())
                np.random.shuffle(idxs)
                target_node_list = idxs[:int(args.select_target_ratio*len(idxs))]
                idx_perturn_test = torch.LongTensor(np.array(target_node_list)).to(device)
                if(n_perturbation > 0):
                    for idx in idx_perturn_test:
                        noisy_data = construct_graph.generate_node_noisy(args,noisy_data,idx,n_perturbation,device)
                        noisy_data = noisy_data.to(device)
                model.eval()
                if(args.if_smoothed):
                    _, acc = smooth_model._sample_noise_ber(args.num_sample, noisy_data.edge_index, noisy_data.edge_weight, noisy_data.y, if_node_level = True, idx_test = idx_perturn_test, idx_train = idx_train)
                    # prediction = prediction_distribution.argmax(1)
                    # acc, prediction = smoothed_linear_evaluation(args, model, noisy_data.x, noisy_data.edge_index, noisy_data.edge_weight, 100, noisy_data.y, idx_train, idx_clean_test, device)
                else:
                    z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight).to(device)
                    if(args.dataset == 'ogbn-arxiv'):
                        acc = lr_evaluation(z, noisy_data.y, idx_train, idx_perturn_test)
                    else:
                        acc = label_evaluation(z, noisy_data.y, idx_train, idx_perturn_test)
                # if(args.if_smoothed):
                #     noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                #     # noisy_data.edge_index,noisy_data.edge_weight = sample_noise_1by1_dense(args,noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test, device)
                #     # noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise_1by1(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                # z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                # # Calculate Robust Accuracy and Bounded Probability 
                # acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                print("Accuracy:",acc)
                accuracys[n_perturbation].append(acc)
                if(args.dataset == 'ogbn-arxiv'):
                    noisy_data = noisy_data.cpu()
                    # z = z.cpu()
                    # torch.cuda.empty_cache()
            model.to(torch.device('cpu'))
        elif(args.attack == 'PRBCD'):
            import generate_prbcd_attack
            # classes, probs = smooth_model.certify_Ber(n0=10,n=100,alpha=0.01,idx_train=idx_train, idx_test = idx_clean_test)
            import construct_graph
            import copy
            # perturbation_sizes = list(range(0,atk_budget+1))
            # perturbation_sizes = [0, 0.05, 0.10,0.25, 0.5, 1.0]
            for global_ptb_rate in perturbation_sizes:
            # for n_perturbation in perturbation_sizes:
                # print("Perturbation Size:{}".format(n_perturbation))
                noisy_data = copy.deepcopy(data)
                # global_atk_budget = len(idx_clean_test) * n_perturbation
                # global_ptb_rate = global_atk_budget/data.edge_index.shape[1]
                # print("Ptb size:{} Ptb rate:{}".format(n_perturbation, global_ptb_rate))
                print("prate:{}".format(global_ptb_rate))
                if(global_ptb_rate > 0):
                    noisy_data = generate_prbcd_attack.generate_prbcd_attack(noisy_data, global_ptb_rate, device)
                    # for idx in idx_clean_test:
                    #     noisy_data = construct_graph.generate_node_noisy(args,noisy_data,idx,n_perturbation,device)
                    #     noisy_data = noisy_data.to(device)
                model.eval()
                if(args.if_smoothed):
                    prediction_distribution, acc = smooth_model._sample_noise_ber(args.num_sample, noisy_data.edge_index, noisy_data.edge_weight, noisy_data.y, if_node_level = True, idx_test = idx_clean_test, idx_train = idx_train)
                    prediction = prediction_distribution.argmax(1)
                    # acc, prediction = smoothed_linear_evaluation(args, model, noisy_data.x, noisy_data.edge_index, noisy_data.edge_weight, 100, noisy_data.y, idx_train, idx_clean_test, device)
                else:
                    z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                    if(args.dataset == 'ogbn-arxiv'):
                        acc = lr_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                    else:
                        acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                # if(args.if_smoothed):
                #     noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                #     # noisy_data.edge_index,noisy_data.edge_weight = sample_noise_1by1_dense(args,noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test, device)
                #     # noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise_1by1(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                # z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                # # Calculate Robust Accuracy and Bounded Probability 
                # acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                print("Accuracy:",acc)
                accuracys[global_ptb_rate].append(acc)
        elif(args.attack == 'random_global'):
            import generate_prbcd_attack
            # classes, probs = smooth_model.certify_Ber(n0=10,n=100,alpha=0.01,idx_train=idx_train, idx_test = idx_clean_test)
            import construct_graph
            import copy
            # perturbation_sizes = list(range(0,atk_budget+1))
            # perturbation_sizes = [0, 0.05, 0.10,0.25, 0.5, 1.0]
            for global_ptb_rate in perturbation_sizes:
            # for n_perturbation in perturbation_sizes:
                # print("Perturbation Size:{}".format(n_perturbation))
                noisy_data = copy.deepcopy(data)
                # global_atk_budget = len(idx_clean_test) * n_perturbation
                # global_ptb_rate = global_atk_budget/data.edge_index.shape[1]
                # print("Ptb size:{} Ptb rate:{}".format(n_perturbation, global_ptb_rate))
                print("prate:{}".format(global_ptb_rate))
                if(global_ptb_rate > 0):
                    noisy_data = construct_graph.generate_node_noisy_global(args,noisy_data,n_perturbation,device)
                    # noisy_data = generate_prbcd_attack.generate_prbcd_attack(noisy_data, global_ptb_rate, device)
                    # for idx in idx_clean_test:
                    #     noisy_data = construct_graph.generate_node_noisy(args,noisy_data,idx,n_perturbation,device)
                    #     noisy_data = noisy_data.to(device)
                model.eval()
                if(args.if_smoothed):
                    prediction_distribution, acc = smooth_model._sample_noise_ber(args.num_sample, noisy_data.edge_index, noisy_data.edge_weight, noisy_data.y, if_node_level = True, idx_test = idx_clean_test, idx_train = idx_train)
                    prediction = prediction_distribution.argmax(1)
                    # acc, prediction = smoothed_linear_evaluation(args, model, noisy_data.x, noisy_data.edge_index, noisy_data.edge_weight, 100, noisy_data.y, idx_train, idx_clean_test, device)
                else:
                    z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                    if(args.dataset == 'ogbn-arxiv'):
                        acc = lr_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                    else:
                        acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                # if(args.if_smoothed):
                #     noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                #     # noisy_data.edge_index,noisy_data.edge_weight = sample_noise_1by1_dense(args,noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test, device)
                #     # noisy_data.edge_index,noisy_data.edge_weight = model.sample_noise_1by1(noisy_data.edge_index,noisy_data.edge_weight,idx_clean_test)
                # z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                # # Calculate Robust Accuracy and Bounded Probability 
                # acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                print("Accuracy:",acc)
                accuracys[global_ptb_rate].append(acc)
    if(args.attack == 'nettack'):
        for n_perturbation in perturbation_sizes:
            mean_misclf_rate_gcn = np.mean(misclf_rates_gcn[n_perturbation])
            std_misclf_rate_gcn = np.std(misclf_rates_gcn[n_perturbation])

            mean_misclf_rate_cl = np.mean(misclf_rates_cl[n_perturbation])
            std_misclf_rate_cl = np.std(misclf_rates_cl[n_perturbation])

            print('[GCN] Beta=%s Ptb size=%s total misclassification rate : %s+-%s' % (args.prob, n_perturbation,mean_misclf_rate_gcn,std_misclf_rate_gcn))
            print('[CL]  Beta=%s Ptb size=%s total misclassification rate : %s+-%s' % (args.prob, n_perturbation,mean_misclf_rate_cl,std_misclf_rate_cl))
    elif(args.attack == 'random'):
        for n_perturbation in perturbation_sizes:
            mean_acc =  np.mean(accuracys[n_perturbation])  
            std_acc =  np.std(accuracys[n_perturbation])     
            print("Beta:{} Ptb size:{} Accuracy:{:.4f}+-{:.4f}".format(args.prob, n_perturbation,mean_acc,std_acc))
    elif(args.attack == 'PRBCD'):
        for n_perturbation in perturbation_sizes:
            # global_atk_budget = len(idx_clean_test) * n_perturbation
            # global_prb_rate = global_atk_budget/data.edge_index.shape[1]
            global_prb_rate = n_perturbation

            mean_acc =  np.mean(accuracys[n_perturbation])  
            std_acc =  np.std(accuracys[n_perturbation])     
            print("Beta:{} Ptb rate:{} Accuracy:{:.4f}+-{:.4f}".format(args.prob,global_prb_rate,mean_acc,std_acc))
            # print("Beta:{} Ptb size:{} Ptb rate:{} Accuracy:{:.4f}+-{:.4f}".format(args.prob, n_perturbation,global_prb_rate,mean_acc,std_acc))
    elif(args.attack == 'random_global'):
        for n_perturbation in perturbation_sizes:
            # global_atk_budget = len(idx_clean_test) * n_perturbation
            # global_prb_rate = global_atk_budget/data.edge_index.shape[1]
            global_prb_rate = n_perturbation

            mean_acc =  np.mean(accuracys[n_perturbation])  
            std_acc =  np.std(accuracys[n_perturbation])     
            print("Beta:{} Ptb rate:{} Accuracy:{:.4f}+-{:.4f}".format(args.prob,global_prb_rate,mean_acc,std_acc))
            # print("Beta:{} Ptb size:{} Ptb rate:{} Accuracy:{:.4f}+-{:.4f}".format(args.prob, n_perturbation,global_prb_rate,mean_acc,std_acc))

    # In[ ]:


