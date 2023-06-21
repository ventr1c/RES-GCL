from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import argparse
import torch
import deeprobust.graph.utils as utils
from deeprobust.graph.global_attack import PRBCD
import copy

def generate_prbcd_attack(data, ptb_rate,device):
    ptb_data = copy.deepcopy(data)
    agent = PRBCD(data, device=device)
    edge_index, edge_weight = agent.attack(ptb_rate=ptb_rate)
    ptb_data.edge_index, ptb_data.edge_weight = edge_index, edge_weight
    return ptb_data