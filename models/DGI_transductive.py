import torch
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
import models.random_smooth as random_smooth

# class GConv(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers):
#         super(GConv, self).__init__()
#         self.layers = torch.nn.ModuleList()
#         self.activations = torch.nn.ModuleList()
#         self.lns = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.layers.append(GCNConv(input_dim, hidden_dim))
#                 self.lns.append(torch.nn.LayerNorm(input_dim))
#             else:
#                 self.layers.append(GCNConv(hidden_dim, hidden_dim))
#                 self.lns.append(torch.nn.LayerNorm(hidden_dim))
#             self.activations.append(nn.PReLU(hidden_dim))

#     def forward(self, x, edge_index, edge_weight=None):
#         if(self.layer_norm_first):
#             x = self.lns[0](x)
#         z = x
#         for conv, act in zip(self.layers, self.activations):
#             z = conv(z, edge_index, edge_weight)
#             z = act(z)
#             if self.use_ln: 
#                 x = self.lns[i+1](x)
#         return z

class GCN_body(nn.Module):
    def __init__(self,nfeat, nhid, dropout=0.5, layer=2,device=None,layer_norm_first=False,use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(1, layer-1):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
            
        self.convs.append(GCNConv(nhid,nhid))
        self.lns.append(nn.LayerNorm(nhid))

        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
    def forward(self,x, edge_index,edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln: 
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, args, encoder, hidden_dim, device=None):
        super(Encoder, self).__init__()
        self.args = args
        self.device = device
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        if(self.args.if_smoothed==True):
            edge_index1, edge_weight1 = random_smooth.sample_noise_all(self.args,edge_index,None,self.device)
            z = self.encoder(x, edge_index)
            z1 = self.encoder(x, edge_index1)
            g = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
            zn = self.encoder(*self.corruption(x, edge_index))
        else:
            z = self.encoder(x, edge_index)
            g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
            zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn

class DGI(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None,use_ln=False,layer_norm_first=False):
        super(DGI, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.dropout = 0
        # self.gconv = GConv(input_dim=nfeat, hidden_dim=nhid, num_layers=layer).to(device)
        self.gconv = GCN_body(nfeat, nhid, self.dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        self.encoder_model = Encoder(args=self.args,encoder=self.gconv, hidden_dim=self.nhid, device=self.device).to(device)
        self.contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    def fit(self, features, edge_index, edge_weight, labels, train_iters=200,seen_node_idx=None,verbose=True):
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            loss = train(self.encoder_model, self.contrast_model, self.features, self.edge_index, self.edge_weight, optimizer)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss))

    def test(self,x, edge_index, edge_weight, y):
        return test(self.encoder_model,x, edge_index, edge_weight, y)
    
    def forward(self, x, edge_index, edge_weight=None):
        z, _, _ = self.encoder_model(x, edge_index)
        return z

def train(encoder_model, contrast_model, x, edge_index, edge_weight, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(x, edge_index)
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, x, edge_index, edge_weight, y):
    encoder_model.eval()
    z, _, _ = encoder_model(x, edge_index, edge_weight)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=300, desc='(T)') as pbar:
        for epoch in range(1, 301):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data.x, data.edge_index, data.edge_weight, data.y)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()