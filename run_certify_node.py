import yaml
from yaml import SafeLoader
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Amazon, Coauthor, WikiCS



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
                    choices=['Grace','GraphCL','BGRL','DGI','GAE','Node2vec','Grace-Jaccard'])
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

# Attack
parser.add_argument('--attack', type=str, default='none',
                    choices=['nettack','random','none', 'PRBCD'],)
parser.add_argument('--select_target_ratio', type=float, default=0.15,
                    help="The number of selected target test nodes for targeted attack")
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
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
print(args)


# In[13]:
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

# we build our own train test split 
if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'CS' or args.dataset == 'Physics'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)

# In[14]:

print(data)
from utils import get_split
if(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'ogbn-arxiv' or args.dataset == 'Physics'):
    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
idx_train = data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
idx_clean_test = data.test_mask.nonzero().flatten()


from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


config_path = "./config/config_Smooth_{}.yaml".format(args.encoder_model)   

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

import torch_geometric.transforms as T


import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.utils import *
import argparse



data = data.to(device)
num_class = int(data.y.max()+1)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=args.num_repeat)

accs = []
betas = [0.001, 0.05, 0.1, 0.9]
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
        model = model_construct(args,args.encoder_model,data,device)
        model.fit(data.x, data.edge_index,data.edge_weight,data.y,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
        # Evaluation 
        import models.random_smooth as random_smooth
        num_class = int(data.y.max()+1)
        smooth_model = random_smooth.Smooth_Ber(args, model, num_class, args.prob, data.x, if_node_level=True,device=device)   

        # Evaluation 
        k_range = list(range(0,21))
        total_Deltas, probs, classes = smooth_model.calculate_certify_K_approx(data.edge_index, data.edge_weight, data.y, n0=args.n0, n=args.n, alpha=args.alpha, idx_train=idx_train, idx_test=idx_clean_test, num_nodes=data.x.shape[0], fn='Cora', k_range=k_range)
        probs = np.array(probs)
        for i in k_range:
            cert_acc = ((2*probs - 1)>2 * total_Deltas[i]).sum()
            print(i,cert_acc)
            cer_accs.append(cert_acc)
        total_cer_accs.append(cer_accs)
        print("Seed:{} Cert Acc:{}".format(seed, [cer_acc/len(idx_clean_test) for cer_acc in cer_accs]))
    total_cer_accs = np.array(total_cer_accs)
    mean_cer_accs = np.mean(total_cer_accs,axis=0)
    print("Beta:{} Mean Cert Acc:{}".format(beta,mean_cer_accs/len(idx_clean_test)))
