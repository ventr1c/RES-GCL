import numpy as np
import torch
from torch_geometric.utils import mask_feature,add_random_edge,dropout_adj
import copy

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset


def construct_noisy_graph(data,perturb_ratio,mode='raw'):
    noisy_data = copy.deepcopy(data)
    if(mode=='raw'):
        noisy_edge_index = data.edge_index
        noisy_edge_weights = data.edge_weight
        noisy_x = data.x
    elif(mode=='random_noise'):
        # random noise: inject/remove edges 
        print("raw graph:",data.edge_index.shape)
        noisy_edge_index,added_edges=add_random_edge(data.edge_index,force_undirected=True,p=perturb_ratio)
        print("add edge:",noisy_edge_index.shape)
        noisy_edge_index,removed_edges=dropout_adj(data.edge_index,data.edge_weight,force_undirected=True,p=perturb_ratio)
        print("remove edge:",noisy_edge_index.shape)
        noisy_edge_index = torch.cat([noisy_edge_index,added_edges],dim=1).long()
        print("updated graph:",noisy_edge_index.shape)
        noisy_data.edge_index = noisy_edge_index
    return noisy_data

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_adj_1by1(args,edge_index, edge_weight, p,device):
    # update edge_index according to edge_weight
    if(edge_weight!=None):
        edge_index = edge_index[:,edge_weight.long()]
        edge_weight = torch.ones([edge_index.shape[1],]).to(device)

    # rs = np.random.RandomState(args.seed)
    # remain_mask = rs.binomial(1,p,edge_index.shape[1])
    remain_mask = np.random.binomial(1,p,edge_index.shape[1])
    remain_index = remain_mask.nonzero()[0]
    remain_edge_index = edge_index[:,remain_index]
    remain_edge_weight = torch.ones([remain_edge_index.shape[1],]).to(device)
    return remain_edge_index,remain_edge_weight

# def construct_augmentation(data):
#     # graph 1:
#     noisy_edge_index,added_edges=add_random_edge(data.edge_index,force_undirected=True,p=0)
#     noisy_edge_index,noisy_edge_weight=dropout_adj(data.edge_index,data.edge_weight,force_undirected=True,p=0.2)
#     aug_edge_index_1 = torch.cat([noisy_edge_index,added_edges],dim=1)
#     aug_x_1 = drop_feature(data.x,drop_prob=0.3)
#     # graph 2:
#     noisy_edge_index,added_edges=add_random_edge(data.edge_index,force_undirected=True,p=0)
#     noisy_edge_index,noisy_edge_weight=dropout_adj(data.edge_index,data.edge_weight,force_undirected=True,p=0.4)
#     aug_edge_index_2 = torch.cat([noisy_edge_index,added_edges],dim=1)
#     aug_x_2 = drop_feature(data.x,drop_prob=0.4)
#     return aug_edge_index_1,aug_x_1,aug_edge_index_2,aug_x_2

# def construct_augmentation_std(data1,data2):
#     # graph 1:
#     aug_edge_index_1,aug_x_1 = data1.edge_index,data1.x
#     # graph 2:
#     aug_edge_index_2,aug_x_2 = data2.edge_index,data2.x
#     return aug_edge_index_1,aug_x_1,aug_edge_index_2,aug_x_2

def construct_augmentation_1(args, x, edge_index, edge_weight=None):
    #

    # graph 1:
    noisy_edge_index,added_edges=add_random_edge(edge_index,force_undirected=True,p=args.add_edge_rate_1)
    noisy_edge_index,noisy_edge_weight=dropout_adj(edge_index,edge_weight,force_undirected=True,p=args.drop_edge_rate_1)
    if(len(added_edges)>0):
        aug_edge_index_1 = torch.cat([noisy_edge_index,added_edges],dim=1).long()
    else:
        aug_edge_index_1 = noisy_edge_index.long()
    aug_x_1 = drop_feature(x,drop_prob=args.drop_feat_rate_1)
    aug_edge_weight_1 = edge_weight
    # graph 2:
    noisy_edge_index,added_edges=add_random_edge(edge_index,force_undirected=True,p=args.add_edge_rate_2)
    noisy_edge_index,noisy_edge_weight=dropout_adj(edge_index,edge_weight,force_undirected=True,p=args.drop_edge_rate_2)
    if(len(added_edges)>0):
        aug_edge_index_2 = torch.cat([noisy_edge_index,added_edges],dim=1).long()
    else:
        aug_edge_index_2 = noisy_edge_index.long()
    aug_x_2 = drop_feature(x,drop_prob=args.drop_feat_rate_2)
    aug_edge_weight_2 = edge_weight
    return aug_edge_index_1,aug_x_1,aug_edge_weight_1,aug_edge_index_2,aug_x_2,aug_edge_weight_2
def construct_augmentation_1(args, x, edge_index, edge_weight=None):
    #

    # graph 1:
    noisy_edge_index,added_edges=add_random_edge(edge_index,force_undirected=True,p=args.add_edge_rate_1)
    noisy_edge_index,noisy_edge_weight=dropout_adj(edge_index,edge_weight,force_undirected=True,p=args.drop_edge_rate_1)
    if(len(added_edges)>0):
        aug_edge_index_1 = torch.cat([noisy_edge_index,added_edges],dim=1).long()
    else:
        aug_edge_index_1 = noisy_edge_index.long()
    aug_x_1 = drop_feature(x,drop_prob=args.drop_feat_rate_1)
    aug_edge_weight_1 = noisy_edge_weight
    # graph 2:
    noisy_edge_index,added_edges=add_random_edge(edge_index,force_undirected=True,p=args.add_edge_rate_2)
    noisy_edge_index,noisy_edge_weight=dropout_adj(edge_index,edge_weight,force_undirected=True,p=args.drop_edge_rate_2)
    if(len(added_edges)>0):
        aug_edge_index_2 = torch.cat([noisy_edge_index,added_edges],dim=1).long()
    else:
        aug_edge_index_2 = noisy_edge_index.long()
    aug_x_2 = drop_feature(x,drop_prob=args.drop_feat_rate_2)
    aug_edge_weight_2 = noisy_edge_weight
    return aug_edge_index_1,aug_x_1,aug_edge_weight_1,aug_edge_index_2,aug_x_2,aug_edge_weight_2
# aug_edge_index_1,aug_x_1,aug_edge_index_2,aug_x_2 = construct_augmentation(noisy_data)

def construct_augmentation(args, x, edge_index, edge_weight=None, device=None):
    # graph 1:
    aug_edge_index_1,aug_edge_weight_1 = drop_adj_1by1(args,edge_index, edge_weight, 1-args.drop_edge_rate_1,device)
    aug_edge_index_1 = aug_edge_index_1.long()

    aug_x_1 = drop_feature(x,drop_prob=args.drop_feat_rate_1)

    # graph 2:
    aug_edge_index_2,aug_edge_weight_2 = drop_adj_1by1(args,edge_index, edge_weight, 1-args.drop_edge_rate_2,device)
    aug_edge_index_2 = aug_edge_index_2.long()

    aug_x_2 = drop_feature(x,drop_prob=args.drop_feat_rate_2)
    return aug_edge_index_1,aug_x_1,aug_edge_weight_1,aug_edge_index_2,aug_x_2,aug_edge_weight_2

def _sample_structure_noise(args,x,edge_index, edge_weight, idx, device):
    # update edge_index according to edge_weight
    if(edge_weight!=None):
        edge_index = edge_index[:,edge_weight.long()]
        edge_weight = torch.ones([edge_index.shape[1],]).to(device)

    # select edge_index according to idx
    

    # rs = np.random.RandomState(args.seed)
    # remain_mask = rs.binomial(1,p,edge_index.shape[1])
    remain_mask = np.random.binomial(1,p,edge_index.shape[1])
    remain_index = remain_mask.nonzero()[0]
    remain_edge_index = edge_index[:,remain_index]
    remain_edge_weight = torch.ones([remain_edge_index.shape[1],]).to(device)
    return remain_edge_index,remain_edge_weight