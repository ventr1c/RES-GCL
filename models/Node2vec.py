#%%
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI

import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import Node2Vec

from torch import Tensor
from torch.nn import Embedding
from torch_geometric.typing import OptTensor, SparseTensor

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

def train(model, optimizer, loader, device):
    model.train()
    # optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model,data):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
    return acc

class Node2vec(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(Node2vec, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.num_workers = 0 if sys.platform.startswith('win') else 4

    def fit(self, features, edge_index, edge_weight, labels, train_iters=200,seen_node_idx=None,verbose=True):    
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx

        self.model = Node2Vec(edge_index, embedding_dim=self.nhid, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)
        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.lr)
        loader = self.model.loader(batch_size=128, shuffle=True,
                          num_workers=self.num_workers)
        
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            loss = train(self.model, optimizer, loader, self.device)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss))

    # def test(self,x, edge_index, edge_weight, y):
    #     return test(self.encoder_model,x, edge_index, edge_weight, y)
    
    def forward(self, x, edge_index, edge_weight=None):
        z = self.model()
        return z

    