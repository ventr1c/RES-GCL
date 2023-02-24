from models.random_smooth import sample_noise,sample_noise_1by1,sample_noise_all,sample_noise_all_dense

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
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


from eval import label_evaluation

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
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
            nn.Linear(project_dim, project_dim))

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
        if(self.if_smoothed and self._train_flag):
            edge_index1, edge_weight1 = sample_noise_all(self.args, edge_index1, edge_weight1, self.device)
            # edge_index1, edge_weight1 = sample_noise_all_dense(self.args,edge_index1,edge_weight1, self.device)
        # print(edge_index1.shape,edge_weight1)
        z, g = self.encoder(x, edge_index, batch, edge_weight)
        z1, g1 = self.encoder(x1, edge_index1, batch,)
        z2, g2 = self.encoder(x2, edge_index2, batch,)
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
        print("Accuracy:",acc)
        result = SVMEvaluator(linear=True)(x, y, split)
        return result

def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

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
