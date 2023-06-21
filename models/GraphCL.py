from models.random_smooth import sample_noise,sample_noise_1by1,sample_noise_all,sample_noise_all_dense, sample_noise_all_graph

import copy
import numpy as np
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch.optim as optim

from eval import label_evaluation,linear_evaluation
import models.random_smooth as random_smooth

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,dropout):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim),
            nn.Dropout(dropout))

    def forward(self, x, edge_index, batch, edge_weight=None):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index,edge_weight)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g
    

class Encoder(torch.nn.Module):
    def __init__(self, args, encoder, augmentor, input_dim, hidden_dim, lr, tau, num_epoch, if_smoothed, device):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.proj_dim = proj_dim
        self.lr = lr 
        self.tau = tau
        self.num_epoch = num_epoch
        self.device = device
        self.if_smoothed = if_smoothed

        self._train_flag = False

    def forward(self, x, edge_index, batch, edge_weight=None):
        # print(x,edge_index,batch,edge_weight)
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index,)
        # print(edge_index1.shape,edge_weight1)
        batch_1, batch_2 = batch, batch
        if(self.args.if_smoothed==True):
            if(self.args.if_keep_structure1 == True): # or ignore_structure or random_drop
                edge_index1,edge_weight1,batch_1 = edge_index1,edge_weight1, batch
            elif(self.args.if_keep_structure1 == False):
                edge_index1,edge_weight1, batch_1 = random_smooth.sample_noise_all_graph(self.args,edge_index1,edge_weight1,batch, self.device)
            if(self.args.if_ignore_structure2 == True):
                edge_index2,edge_weight2, batch_2 = random_smooth.sample_noise_all_graph(self.args,edge_index2,edge_weight2,batch, self.device)

        # if(self.if_smoothed):
        #     edge_index1, edge_weight1 = sample_noise_all(self.args, edge_index1, edge_weight1, self.device)
            # edge_index1, edge_weight1, edge_attr1 = sample_noise_all_graph(self.args, edge_index1, edge_weight1, edge_attr1, self.device)
            # edge_index1, edge_weight1 = sample_noise_all_dense(self.args,edge_index1,edge_weight1, self.device)
        # print(edge_index1.shape,edge_weight1)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch_1)
        z2, g2 = self.encoder(x2, edge_index2, batch_2)
        return z, g, z1, z2, g1, g2

    def fit(self,dataloader):
        self._train_flag = True
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='G2G').to(self.device)

        optimizer = Adam(self.parameters(), lr=self.lr)

        with tqdm(total=self.num_epoch, desc='(T)') as pbar:
            for epoch in range(1, self.num_epoch+1):
                loss = self._train(contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
        self._train_flag = False

    def _train(self, contrast_model, dataloader, optimizer):
        self.train()
        epoch_loss = 0
        for data in dataloader:
            data = data.to(self.device)
            # print(data)
            optimizer.zero_grad()

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

            _, _, _, _, g1, g2 = self.forward(data.x, data.edge_index, data.batch, data.edge_weight)
            g1, g2 = [self.encoder.project(g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss
        

    def smooth_test(self, num, dataset,split):
        self.eval()
        predictions = []
        for _ in range(num):
            x = []
            y = []
            rs_dataset = copy.deepcopy(dataset)
            if(self.if_smoothed == True): 
                for rs_data in rs_dataset:
                    # rs_edge_index, rs_edge_weight = sample_noise_all_dense(self.args,data.edge_index, data.edge_weight, self.device)
                    rs_data.edge_index, _ = sample_noise_all(self.args, rs_data.edge_index, rs_data.edge_weight, self.device)
                    # rs_data.edge_index, _, rs_data.edge_attr = sample_noise_all_graph(self.args, rs_data.edge_index, rs_data.edge_weight, rs_data.edge_attr, self.device)
            rs_dataloader = DataLoader(rs_dataset, batch_size=self.args.batch_size)
            for data in rs_dataloader:
                data = data.to(self.device)
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device).to(self.device)
                _, g, _, _, _, _ = self.forward(data.x, data.edge_index, data.batch,)
                # if(self.if_smoothed == True): 
                #     # print(data.edge_index.shape)
                #     # sample_noise_1by1(self.args, data.x, data.edge_index, data.edge_weight,idxs,device)
                #     # rs_edge_index, rs_edge_weight = sample_noise_all_dense(self.args,data.edge_index, data.edge_weight, self.device)
                #     rs_edge_index, _ = sample_noise_all(self.args, data.edge_index, data.edge_weight, self.device)
                #     _, g, _, _, _, _ = self.forward(data.x, rs_edge_index, data.batch,)
                # else:
                #     _, g, _, _, _, _ = self.forward(data.x, data.edge_index, data.batch,)
                x.append(g)
                y.append(data.y)
                # print(g,data.y)
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            acc, prediction = linear_evaluation(x, y, split['train'], split['test'])
            predictions.append(prediction)
        num_class = int(y.max()+1)
        prediction_distribution = np.array([np.bincount(prediction_list, minlength=num_class) for prediction_list in zip(*predictions)])
        final_prediction = prediction_distribution.argmax(1)
        acc = (((torch.tensor(final_prediction).cpu()==y[split['test']].cpu()).sum())/len(final_prediction))
        # print((torch.tensor(final_prediction)==y[split['test']]))
        return acc
    def test(self, dataloader,split):
        self.eval()
        x = []
        y = []
        for data in dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
            if(self.if_smoothed == True): 
                # print(data.edge_index.shape)
                # sample_noise_1by1(self.args, data.x, data.edge_index, data.edge_weight,idxs,device)
                # rs_edge_index, rs_edge_weight = sample_noise_all_dense(self.args,data.edge_index, data.edge_weight, self.device)
                rs_edge_index, rs_edge_weight = sample_noise_all(self.args, data.edge_index, data.edge_weight, self.device)
                # print(data.x,rs_edge_index.shape)
                _, g, _, _, _, _ = self.forward(data.x, rs_edge_index, data.batch,)
            else:
                _, g, _, _, _, _ = self.forward(data.x, data.edge_index, data.batch,)
            x.append(g)
            y.append(data.y)
            # print(g,data.y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)


        # split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
        acc = label_evaluation(x, y, split['train'], split['test'])
        
        # print("Accuracy:",acc)
        return acc
        # result = SVMEvaluator(linear=True)(x, y, split)
        # return result

class GraphCL(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(GraphCL, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.aug2 = A.Identity()
        # self.aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=args.walk_length_2),
        #                    A.NodeDropping(pn=args.drop_node_rate_2),
        #                    A.FeatureMasking(pf=args.drop_feat_rate_2),
        #                    A.EdgeRemoving(pe=args.drop_edge_rate_2)], 1)
        self.aug1 = A.RandomChoice([A.NodeDropping(pn=args.drop_node_rate_2),
                           A.FeatureMasking(pf=args.drop_feat_rate_2),
                           A.EdgeRemoving(pe=args.drop_edge_rate_2)], 1)
        A.Compose([A.EdgeRemoving(pe=self.args.drop_edge_rate_2), A.FeatureMasking(pf=self.args.drop_edge_rate_2)])
        self.gconv = GConv(input_dim=nfeat, hidden_dim=nhid, num_layers=layer, dropout= dropout).to(device)
        
        self.input_dim = nfeat
        self.encoder_model = Encoder(args, encoder=self.gconv, augmentor=(self.aug1, self.aug2), input_dim=self.input_dim, hidden_dim = self.nhid, lr = lr, tau = tau, num_epoch = None, if_smoothed = if_smoothed, device = device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=tau), mode='G2G').to(device)
    def fit(self, dataloader, train_iters=200,seen_node_idx=None,verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            loss = train(self.encoder_model, self.contrast_model, dataloader, optimizer, self.device)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss))
            
    def test(self, dataloader):
        return test(self.encoder_model, dataloader)
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        x = x.float()
        _, g, _, _, _, _ = self.encoder_model(x, edge_index,batch=batch, edge_weight=edge_weight)
        # z = torch.cat([h1, h2], dim=1)
        return g
    
def train(encoder_model, contrast_model, dataloader, optimizer, device):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        data.x = data.x.float()
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name='PTC_MR')
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
