import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from models.random_smooth import sample_noise,sample_noise_1by1,sample_noise_all


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, args, encoder, augmentor, nfeat, hidden_dim, proj_dim, lr, tau, num_epoch, if_smoothed, device):
        super(Encoder, self).__init__()
        self.encoder = encoder.to(device)
        self.augmentor = augmentor

        self.args = args
        self.nfeat = nfeat
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.lr = lr 
        self.tau = tau
        self.num_epoch = num_epoch
        self.device = device
        self.if_smoothed = if_smoothed

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim).to(device)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim).to(device)

        # Training Flag
        self._train_flag = False

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        if(self.args.if_smoothed and self._train_flag):
            edge_index1, edge_weight1 = sample_noise_all(self.args, edge_index1, edge_weight1, self.device)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def fit(self,data):
        self._train_flag = True
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=self.tau), mode='L2L', intraview_negs=True).to(self.device)

        optimizer = Adam(self.parameters(), lr=self.lr)

        with tqdm(total=self.num_epoch, desc='(T)') as pbar:
            for epoch in range(1, self.num_epoch+1):
                loss = self._train(contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
        self._train_flag = False
    def _train(self, contrast_model, data, optimizer):
        self.train()
        optimizer.zero_grad()
        z, z1, z2 = self.forward(data.x, data.edge_index, data.edge_weight)
        h1, h2 = [self.project(x) for x in [z1, z2]]
        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        return loss.item()
    def test(self, data):
        self.eval()
        split = get_split(num_samples=data.x.shape[0], train_ratio=0.1, test_ratio=0.8)
        if(self.if_smoothed == True):
            rs_edge_index, rs_edge_weight = sample_noise(self.args,data.edge_index, data.edge_weight,split['valid'], self.device)
            # rs_edge_index, rs_edge_weight = sample_noise_1by1(self.args,data.edge_index, data.edge_weight,split['valid'].to(self.device), self.device)
            z, _, _ = self.forward(data.x, rs_edge_index, rs_edge_weight)
        else:
            z, _, _ = self.forward(data.x, data.edge_index, data.edge_weight)
        result = LREvaluator()(z, data.y, split)
        return result

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()