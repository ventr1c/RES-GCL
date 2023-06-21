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


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

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
        self.device = device
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

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

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        if(self.args.if_smoothed==True):
            if(self.args.if_keep_structure1 == True): # or ignore_structure or random_drop
                edge_index1,edge_weight1 = edge_index1,edge_weight1
            elif(self.args.if_keep_structure1 == False):
                edge_index1,edge_weight1 = random_smooth.sample_noise_all(self.args,edge_index1,edge_weight1,self.device)
            if(self.args.if_ignore_structure2 == True):
                edge_index2,edge_weight2 = random_smooth.sample_noise_all(self.args,edge_index2,edge_weight2,self.device)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

class BGRL(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(BGRL, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.aug1 = A.Compose([A.EdgeRemoving(pe=self.args.drop_edge_rate_1), A.FeatureMasking(pf=self.args.drop_edge_rate_1)])
        self.aug2 = A.Compose([A.EdgeRemoving(pe=self.args.drop_edge_rate_2), A.FeatureMasking(pf=self.args.drop_edge_rate_2)])
        self.gconv = GConv(input_dim=nfeat, hidden_dim=nhid, num_layers=layer, dropout= 0.1).to(device)
        self.encoder_model = Encoder(args, encoder=self.gconv, augmentor=(self.aug1, self.aug2), hidden_dim=nhid,device=device).to(device)
        self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)
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
        h1, h2, _, _, _, _ = self.encoder_model(x, edge_index,edge_weight)
        z = torch.cat([h1, h2], dim=1)
        return z
    
def train(encoder_model, contrast_model, x, edge_index, edge_attr, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(x, edge_index, edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def test(encoder_model, x, edge_index, edge_weight, y):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(x, edge_index,edge_weight)
    z = torch.cat([h1, h2], dim=1)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    print(split)
    result = LREvaluator()(z, y, split)
    return result


def main():
    transform = T.Compose([T.NormalizeFeatures()])
    device = torch.device('cuda')
    # path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    # dataset = WikiCS(path, transform=T.NormalizeFeatures())
    dataset = Planetoid(root='../data/', name='Cora', transform=transform)
    data = dataset[0].to(device)
    print(data)
    model = BGRL(None, data.x.shape[1], nhid=128, nproj=None, nclass=None, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=0.2, layer=2,if_smoothed=True,device=device)
    model.fit(data.x,data.edge_index,data.edge_attr,data.y, train_iters=500)
    test_result = model.test(data.x,data.edge_index,data.edge_attr,data.y)
    # aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    # aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    # gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
    # encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
    # contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

    # optimizer = Adam(encoder_model.parameters(), lr=0.01)

    # with tqdm(total=100, desc='(T)') as pbar:
    #     for epoch in range(1, 101):
    #         loss = train(encoder_model, contrast_model, data.x, data.edge_index, data.edge_attr, optimizer)
    #         pbar.set_postfix({'loss': loss})
    #         pbar.update()

    # test_result = test(encoder_model,  data.x, data.edge_index, data.attr, data.y)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()