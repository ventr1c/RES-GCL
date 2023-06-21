import torch
import torch_geometric.nn as pyg_nn
from typing import Optional, Tuple
from models._GCN import GCN
import torch.optim as optim


class GAE_model(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: Optional[torch.nn.Module] = None):
        super(GAE_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gae = pyg_nn.GAE(encoder,decoder)
        self.gae.reset_parameters()

    def encode(self, x, edge_index):
        return self.gae.encode(x, edge_index)
    def decode(self, z, edge_index):
        return self.gae.decode(z, edge_index)
    
    def forward(self, x, edge_index):
        z = self.gae.encode(x, edge_index)
        # return self.decode(z, edge_index)
        return z

    def recon_loss(self, z, edge_index):
        loss = self.gae.recon_loss(z, edge_index)
        return loss

    def test(self,z, pos_edge_index, neg_edge_index):
        return self.gae.test(z, pos_edge_index, neg_edge_index)

def train(model, x, train_pos_edge_index,optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z,train_pos_edge_index)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model,data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
    val_auc, val_ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    test_auc, test_ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    return val_auc, val_ap, test_auc, test_ap
    
class GAE(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(GAE, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device
        self.encoder = GCN(nfeat, nhid*2, nhid, num_layers=layer, dropout=args.dropout, bn=False).to(device)
        self.gae = GAE_model(self.encoder)

    def fit(self, features, edge_index, edge_weight, labels, train_iters=200,seen_node_idx=None,verbose=True):
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            loss = train(self.gae, features, edge_index, optimizer)
            # loss = train(self.encoder_model, self.contrast_model, self.features, self.edge_index, self.edge_weight, optimizer)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss))

    def test(self,x, edge_index, edge_weight, y):
        return test(self.encoder_model,x, edge_index, edge_weight, y)
    
    def forward(self, x, edge_index, edge_weight=None):
        z = self.gae(x, edge_index)
        return z