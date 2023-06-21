import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCNConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj
import numpy as np
import networkx as nx
import torch.optim as optim
from torch_geometric.utils import dropout_adj, get_laplacian, add_self_loops
from torch_scatter import scatter_add

# from models.pgd_attack import PGD_attack_graph
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_class: int, activation,
                 base_model=GCNConv, dropout: float=0.5):
        super(GCN, self).__init__()
        self.base_model = base_model

        self.conv1 = base_model(in_channels, out_channels)
        self.head = base_model(out_channels, n_class)
        self.dropout = dropout
        self.activation = activation
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.head(x, edge_index), dim=1)
    
class GCN_body(nn.Module):
    def __init__(self,nfeat, nhid, dropout=0.5, layer=2,device=None,layer_norm_first=False,use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat,2 * nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(1, layer-1):
            self.convs.append(GCNConv(2 * nhid,2 * nhid))
            self.lns.append(nn.LayerNorm(2 * nhid))
            
        self.convs.append(GCNConv(2 * nhid,nhid))
        self.lns.append(nn.LayerNorm(2 * nhid))

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

    def inference(self, x_all, subgraph_loader):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                x = conv(x, edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_class: int, activation,
                 base_model=GCNConv, dropout: float=0.5):
        super(GCN, self).__init__()
        self.base_model = base_model

        self.conv1 = base_model(in_channels, out_channels)
        self.head = base_model(out_channels, n_class)
        self.dropout = dropout
        self.activation = activation
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.head(x, edge_index), dim=1)
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_class: int, activation,
                 base_model=GATConv, input_dropout: float=0.5, coef_dropout: float=0.5):
        super(GAT, self).__init__()
        self.base_model = base_model
        self.conv1 = base_model(in_channels, out_channels, 8, dropout=coef_dropout)
        self.head = base_model(out_channels*8, n_class, 1, dropout=coef_dropout)
        self.dropout = input_dropout
        self.activation = activation
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(self.head(x, edge_index), dim=1)


class Ariel(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None,use_ln=False,layer_norm_first=False):
        super(Ariel, self).__init__()
        self.encoder = Encoder(nfeat, nhid, F.relu, GCNConv, k=2).to(device)
        # self.encoder: Encoder = encoder
        # self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        self.tau: float = tau

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln

        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay
        self.tau = tau
        self.args = args
    
        self.if_smoothed=args.if_smoothed

        self.layer_norm_first = layer_norm_first
        # self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        # linear evaluation layer
        self.fc = nn.Linear(nhid,nclass).to(device)
        # projection layer
        self.fc1 = torch.nn.Linear(nhid, nproj).to(device)
        self.fc2 = torch.nn.Linear(nproj, nhid).to(device)

        # self.fc1 = torch.nn.Linear(nhid, nproj)
        # self.fc2 = torch.nn.Linear(nproj, nhid)
        self.cos = nn.CosineSimilarity()
        self.sample_size = 500
        self.eps = self.args.eps
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.lamb = self.args.lamb

        # if(self.args.dataset == 'Cora'):
        #     self.eps = 0.5
        #     self.alpha = 200
        #     self.lamb = 0
        # elif(self.args.dataset == 'Physics'):
        #     self.eps = 1
        #     self.alpha = 50
        #     self.lamb = 0
        # elif(self.args.dataset == 'Pubmed'):
        #     self.eps = 1
        
    def forward(self, x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    # def forward(self, x, edge_index, edge_weight=None):
    #     x = self.body(x, edge_index,edge_weight)
    #     return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        simi = torch.exp(self.cos(h1,h2)/self.tau)
            
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        #ret = ret.mean() if mean else ret.sum()

        return ret, simi

    def fit(self, features, edge_index, edge_weight, labels, train_iters=200,seen_node_idx=None,idx_train=None,idx_val=None, idx_test=None, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        # self._train_with_val(self.labels, train_iters, verbose)
        self.G = nx.Graph()
        self.G.add_edges_from(list(zip(self.edge_index.cpu().numpy()[0],self.edge_index.cpu().numpy()[1])))

        self._train_without_val(self.labels, train_iters, verbose)
    
    def _train_without_val(self, labels, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            edge_index, edge_weight = self.edge_index, self.edge_weight

            S = self.G.subgraph(np.random.permutation(self.G.number_of_nodes())[:self.sample_size])
            x = self.features[np.array(S.nodes())].to(self.device)
            S = nx.relabel.convert_node_labels_to_integers(S, first_label=0, ordering='default')
            edge_index = np.array(S.edges()).T
            edge_index = torch.LongTensor(np.hstack([edge_index,edge_index[::-1]])).to(self.device)

            # optimizer.zero_grad()
            adj = edge2adj(x, edge_index)
            edge_index_1 = dropout_adj(edge_index, p=self.args.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=self.args.drop_edge_rate_2)[0]

            x_1 = drop_feature(x, self.args.drop_feat_rate_1)
            x_2 = drop_feature(x, self.args.drop_feat_rate_2)  

            adj_1 = edge2adj(x_1, edge_index_1)
            adj_2 = edge2adj(x_2, edge_index_2)
            
            steps, node_ratio = 5, 0.2
            if self.eps > 0:
                adj_3, x_3 = self.PGD_attack_graph(edge_index_1, edge_index, x_1, x, steps, node_ratio, self.alpha, self.beta)
            z = self.forward(x, adj)
            z_1 = self.forward(x_1, adj_1)
            z_2 = self.forward(x_2, adj_2)
            loss1, simi1 = self.loss(z_1,z_2,batch_size=0)
            loss2, simi2 = self.loss(z_1,z,batch_size=0)
            loss3, simi3 = self.loss(z_2,z,batch_size=0)
            print(loss1)
            loss1 = loss1.mean() + self.lamb*torch.clamp(simi1*2 - simi2.detach()-simi3.detach(), 0).mean()
            if self.eps > 0:
                z_3 = self.forward(x_3,adj_3)
                loss2, _ = self.loss(z_1,z_3)
                loss2 = loss2.mean()
                loss = (loss1 + self.eps*loss2)
            else: 
                loss = loss1
                loss2 = loss1

            loss.backward()
            optimizer.step()

            # loss1, loss2 = train(model, x, edge_index, eps, lamb, alpha, beta, 5, 0.2)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))

    def PGD_attack_graph(self, edge_index_1, edge_index_2, x_1, x_2, steps, node_ratio, alpha, beta):
        """ PGD attack on both features and edges"""
        for param in  self.parameters():
            param.requires_grad = False
        self.eval()
        device = x_1.device
        total_edges = edge_index_2.shape[1]
        n_node = x_2.shape[0]
        eps = total_edges * node_ratio/2
        xi = 1e-3
        
        A_ = torch.sparse.FloatTensor(edge_index_2, torch.ones(total_edges,device=device), torch.Size((n_node, n_node))).to_dense() 
        C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
        S_ = torch.zeros_like(A_, requires_grad= True)
        mask = torch.ones_like(A_)
        mask = mask - torch.tril(mask)
        delta = torch.zeros_like(x_2, device=device, requires_grad=True)
        adj_1 = edge2adj(x_1, edge_index_1)
        self.to(device)
        for epoch in range(steps):
            S = (S_ * mask)
            S = S + S.T
            A_prime = A_ + (S * C_)
            adj_hat = normalize_adj_tensor(A_prime + torch.eye(n_node,device=device))
            z1 = self.forward(x_1, adj_1)
            z2 = self.forward(x_2 + delta, adj_hat) 
            loss, _ = self.loss(z1, z2, batch_size=0) 
            attack_loss = loss.mean()
            attack_loss.backward()
            S_.data = (S_.data + alpha/np.sqrt(epoch+1)*S_.grad.detach()) # annealing
            S_.data = bisection(S_.data, eps, xi) # clip S
            S_.grad.zero_()
            
            delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
            delta.grad.zero_()

        randm = torch.rand(n_node, n_node,device=device)
        discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
        discretized_S = discretized_S + discretized_S.T
        A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
            
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        x_hat = x_2 + delta.data.to(device)
        assert torch.equal(A_hat, A_hat.transpose(0,1))
        return normalize_adj_tensor(A_hat), x_hat
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def edge2adj(x, edge_index):
    """Convert edge index to adjacency matrix"""
    num_nodes = x.shape[0]
    tmp, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(tmp.size(1), dtype=None,
                                     device=edge_index.device)

    row, col = tmp[0], tmp[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return torch.sparse.FloatTensor(tmp, edge_weight,torch.Size((num_nodes, num_nodes)))

def bisection(a,eps,xi,ub=1):
    pa = torch.clamp(a, 0, ub)
    if torch.sum(pa) <= eps:
        upper_S_update = pa
    else:
        mu_l = torch.min(a-1)
        mu_u = torch.max(a)
        mu_a = (mu_u + mu_l)/2
        while torch.abs(mu_u - mu_l)>xi:
            mu_a = (mu_u + mu_l)/2
            gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
            gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
            if gu == 0:
                break
            if torch.sign(gu) == torch.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = torch.clamp(a-mu_a, 0, ub)
    return upper_S_update


def normalize_adj_tensor(adj):
    """Symmetrically normalize adjacency tensor."""
    rowsum = torch.sum(adj,1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def normalize_adj_tensor_sp(adj):
    """Symmetrically normalize sparse adjacency tensor."""
    device = adj.device
    adj = adj.to("cpu")
    rowsum = torch.spmm(adj, torch.ones((adj.size(0),1))).reshape(-1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = torch.mm(torch.smm(adj.transpose(0,1),d_mat_inv_sqrt.transpose(0,1)),d_mat_inv_sqrt)
    return adj.to(device)

def PGD_attack_graph(model, edge_index_1, edge_index_2, x_1, x_2, steps, node_ratio, alpha, beta):
    """ PGD attack on both features and edges"""
    for param in  model.parameters():
        param.requires_grad = False
    model.eval()
    device = x_1.device
    total_edges = edge_index_2.shape[1]
    n_node = x_2.shape[0]
    eps = total_edges * node_ratio/2
    xi = 1e-3
    
    A_ = torch.sparse.FloatTensor(edge_index_2, torch.ones(total_edges,device=device), torch.Size((n_node, n_node))).to_dense() 
    C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
    S_ = torch.zeros_like(A_, requires_grad= True)
    mask = torch.ones_like(A_)
    mask = mask - torch.tril(mask)
    delta = torch.zeros_like(x_2, device=device, requires_grad=True)
    adj_1 = edge2adj(x_1, edge_index_1)
    model.to(device)
    for epoch in range(steps):
        S = (S_ * mask)
        S = S + S.T
        A_prime = A_ + (S * C_)
        adj_hat = normalize_adj_tensor(A_prime + torch.eye(n_node,device=device))
        z1 = model(x_1,edge_index_1)
        # z1 = model(x_1, adj_1)
        z2 = model(x_2 + delta, adj_hat) 
        loss, _ = model.loss(z1, z2, batch_size=0) 
        attack_loss = loss.mean()
        attack_loss.backward()
        S_.data = (S_.data + alpha/np.sqrt(epoch+1)*S_.grad.detach()) # annealing
        S_.data = bisection(S_.data, eps, xi) # clip S
        S_.grad.zero_()
        
        delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
        delta.grad.zero_()

    randm = torch.rand(n_node, n_node,device=device)
    discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
    discretized_S = discretized_S + discretized_S.T
    A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
        
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    x_hat = x_2 + delta.data.to(device)
    assert torch.equal(A_hat, A_hat.transpose(0,1))
    return normalize_adj_tensor(A_hat), x_hat