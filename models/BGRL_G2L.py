import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import BootstrapContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS,Planetoid
import models.random_smooth as random_smooth
import models.random_smooth_graph as random_smooth_graph

import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import SVMEvaluator, get_split
from GCL.models import BootstrapContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, args, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch', device=None):
        super(Encoder, self).__init__()
        self.args = args
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.device = device
    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = x.float()
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        batch_1, batch_2 = batch, batch
        if(self.args.if_smoothed==True):
            if(self.args.if_keep_structure1 == True): # or ignore_structure or random_drop
                edge_index1,edge_weight1,batch_1 = edge_index1,edge_weight1, batch
            elif(self.args.if_keep_structure1 == False):
                edge_index1,edge_weight1, batch_1 = random_smooth_graph.sample_noise_all_graph(self.args,edge_index1,edge_weight1,batch, self.device)
            if(self.args.if_ignore_structure2 == True):
                edge_index2,edge_weight2, batch_2 = random_smooth_graph.sample_noise_all_graph(self.args,edge_index2,edge_weight2,batch, self.device)
        h1, h1_online = self.online_encoder(x1, edge_index1, None)
        h2, h2_online = self.online_encoder(x2, edge_index2, None)

        g1 = global_add_pool(h1, batch_1)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch_2)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, None)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, None)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target


class BGRL_G2L(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(BGRL_G2L, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.aug1 = A.Compose([A.EdgeRemoving(pe=self.args.drop_edge_rate_1), A.FeatureMasking(pf=self.args.drop_feat_rate_1)])
        self.aug2 = A.Compose([A.EdgeRemoving(pe=self.args.drop_edge_rate_2), A.FeatureMasking(pf=self.args.drop_feat_rate_2)])
        self.gconv = GConv(input_dim=nfeat, hidden_dim=nhid, num_layers=layer, dropout= dropout).to(device)
        self.encoder_model = Encoder(self.args, encoder=self.gconv, augmentor=(self.aug1, self.aug2), hidden_dim=nhid, dropout=dropout, device = device).to(device)
        self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='G2L').to(device)
    def fit(self, dataloader, train_iters=200,seen_node_idx=None,verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            loss = train(self.encoder_model, self.contrast_model, dataloader, optimizer, self.device)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss))
            
    def test(self, dataloader):
        return test(self.encoder_model, dataloader)
    
    def forward(self, x, edge_index, edge_weight=None,batch=None):
        x = x.float()
        h1, h2, _, _, _, _ = self.encoder_model(x, edge_index,edge_weight,batch=batch)
        z = torch.cat([h1, h2], dim=1)
        return z
    
    
def train(encoder_model, contrast_model, dataloader, optimizer, device):
    encoder_model.train()
    total_loss = 0

    for data in dataloader:
        # data = data.to('cuda')
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, g1_target, g2_target = encoder_model(data.x, data.edge_index, batch=data.batch)

        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_target=g1_target.detach(), g2_target=g2_target.detach(), batch=data.batch)
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)

        total_loss += loss.item()

    return total_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)
        x.append(z)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result
