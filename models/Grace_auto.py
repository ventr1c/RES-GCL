import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

from construct_graph import construct_augmentation_1by1,construct_augmentation_overall
from copy import deepcopy
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import subgraph
from models.random_smooth import sample_noise_all_dense,sample_noise_1by1_dense
from torch_geometric.utils import to_undirected, to_dense_adj,to_torch_coo_tensor,dense_to_sparse,degree
import torch_geometric.utils as pyg_utils

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
        
class Grace_auto(nn.Module):

    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None,use_ln=False,layer_norm_first=False):

        super(Grace_auto, self).__init__()

        assert device is not None, "Please specify 'device'!"
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
        self.cont_weight = args.cont_weight
        self.clf_weight = args.clf_weight
        self.inv_weight = args.inv_weight
        # self.unlabeled_idx = unlabeled_idx
        self.args = args
        self.if_smoothed=args.if_smoothed

        self.layer_norm_first = layer_norm_first
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(nfeat, nhid))
        # for _ in range(layer-2):
        #     self.convs.append(GCNConv(nhid,nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        
        # linear evaluation layer
        self.fc = nn.Linear(nhid,nclass).to(device)

        # projection layer
        self.fc1 = torch.nn.Linear(nhid, nproj).to(device)
        self.fc2 = torch.nn.Linear(nproj, nhid).to(device)

    
    def forward(self, x, edge_index, edge_weight=None):
        # for conv in self.convs:
        #     x = F.relu(conv(x, edge_index,edge_weight))
        #     x = F.dropout(x, self.dropout, training=self.training)
        x = self.body(x, edge_index,edge_weight)
        # x = self.fc(x)
        # return F.log_softmax(x,dim=1)
        return x
    def get_h(self, x, edge_index,edge_weight=None):
        self.eval()
        x = self.body(x, edge_index,edge_weight)
        # for conv in self.convs:
        #     x = F.relu(conv(x, edge_index))
        return x
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

        
    def fit(self, features, edge_index, edge_weight, labels, train_iters=200, drop_edge_rate_1=None, drop_edge_rate_2=None, drop_feat_rate_1=None, drop_feat_rate_2=None, seen_node_idx=None,verbose=False):
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
        self.args.drop_edge_rate_1, self.args.drop_edge_rate_2, self.args.drop_feat_rate_1, self.args.drop_feat_rate_2 = drop_edge_rate_1, drop_edge_rate_2, drop_feat_rate_1, drop_feat_rate_2,
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.seen_node_idx = seen_node_idx
        self._train_with_val(self.labels, train_iters, verbose)
        # if idx_val is None:
        #     self._train_without_val(self.labels, idx_train, train_iters, verbose)
        # else:
        #     self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # edge_index, edge_weight = self.sample_noise_all(self.edge_index,self.edge_weight,idx_train)
            edge_index, edge_weight = self.edge_index, self.edge_weight
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1by1(self.args, self.features, edge_index, edge_weight)
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_overall(self.args, self.features, edge_index, edge_weight, device= self.device)
            if(self.if_smoothed==True):
                # edge_index_2,edge_weight_2 = self.sample_noise_all_sparse(self.args,edge_index_1,edge_weight_1,self.features)
                # print(edge_index_1,edge_weight_1)
                if(self.args.if_keep_structure1 == True): # or ignore_structure or random_drop
                    edge_index_1,edge_weight_1 = edge_index_1,edge_weight_1
                elif(self.args.if_keep_structure1 == False):
                    edge_index_1,edge_weight_1 = self.sample_noise_all(edge_index_1,edge_weight_1)
                if(self.args.if_ignore_structure2 == True):
                    edge_index_2,edge_weight_2 = self.sample_noise_all(edge_index_2,edge_weight_2)
                # edge_index_1,edge_weight_1 = self.sample_noise_all(edge_index_1,edge_weight_1,idx_train)
                # edge_index_2,edge_weight_2 = self.sample_noise_all(edge_index_2,edge_weight_2,idx_train)
                # idx_overall = torch.tensor(range(self.features.shape[0])).to(self.device)
                # edge_index_1,edge_weight_1 = self.sample_noise_1by1(edge_index_1, edge_weight_1,idx_overall)
                # print(edge_index_1,edge_weight_1)
            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2

            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)

            loss =  cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}
        

    def linear_evaluation(self):
        pass

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            output = self.clf_head(output)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def sim(self, z1: torch.Tensor, z2: torch.Tensor):
    #     z1 = F.normalize(z1)
    #     z2 = F.normalize(z2)
    #     return z1@z2.T/(z1.norm(dim=1, keepdim=True)@z2.norm(dim=1, keepdim=True).T)

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
            refl_sim = f(self.sim(z1[mask], z1[mask]))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2[mask]))  # [B, N]

            return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
            # losses.append(-torch.log(
            #     between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            #     / (refl_sim.sum(1) + between_sim.sum(1)
            #        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        # h1 = z1
        # h2 = z2

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def clf_loss(self, z: torch.Tensor, labels, idx):
        # h = self.projection(z)
        h = z
        # print(z,labels,idx)
        output = self.clf_head(h)
        
        clf_loss = F.nll_loss(output[idx],labels[idx])
        return clf_loss
    
    def clf_head(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        return F.log_softmax(z,dim=1)
        # z = F.elu(self.fc1_c(z))
        # return self.fc2_c(z)
        # return z
    
    def sample_noise(self,edge_index, edge_weight,idxs):
        noisy_edge_index = edge_index.clone().detach()
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()

        # # rand_noise_data = copy.deepcopy(data)
        # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
        # m = Bernoulli(torch.tensor([args.prob]).to(device))
        # # sample edge_index of train node
        # train_edge_index, _, edge_mask = subgraph(data.train_mask,data.edge_index,relabel_nodes=False)
        # train_edge_weights = torch.ones([1,train_edge_index.shape[1]]).to(device)
        # # generate random noise 
        # mask = m.sample(train_edge_weights.shape).squeeze(-1).int()
        # rand_inputs = torch.randint_like(train_edge_weights, low=0, high=2, device='cuda').squeeze().int().to(device)
        # noisy_train_edge_weights = train_edge_weights * mask + rand_inputs * (1-mask)
        # noisy_train_edge_weights
        for idx in idxs:
            idx_s = (noisy_edge_index[0] == idx).nonzero().flatten()
            m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
            # print(rand_noise_data.edge_weight[idx_s])
            # print(rand_noise_data.edge_weight)
            # break
            mask = m.sample(noisy_edge_weight[idx_s].shape).squeeze(-1).int()
            # print(mask)
            rand_inputs = torch.randint_like(noisy_edge_weight[idx_s], low=0, high=2).squeeze().int().to(self.device)
            # print(rand_noise_data.edge_weight.shape,mask.shape)
            noisy_edge_weight[idx_s] = noisy_edge_weight[idx_s] * mask #+ rand_inputs * (1-mask)
            # print(rand_noise_data.edge_weight.shape)
            # break
        noisy_edge_weight = noisy_edge_weight.float()
        # if(noisy_edge_weight!=None):
        #     noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        #     noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        return noisy_edge_index, noisy_edge_weight

    def sample_noise_all(self,edge_index, edge_weight,idxs=None):
        noisy_edge_index = edge_index.clone().detach()
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()
        # # rand_noise_data = copy.deepcopy(data)
        # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
        m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
        mask = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
        rand_inputs = torch.randint_like(noisy_edge_weight, low=0, high=2).squeeze().int().to(self.device)
        # print(rand_noise_data.edge_weight.shape,mask.shape)
        noisy_edge_weight = noisy_edge_weight * mask # + rand_inputs * (1-mask)
            
        if(noisy_edge_weight!=None):
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        return noisy_edge_index, noisy_edge_weight
    def sample_noise_1by1(self,edge_index, edge_weight,idxs):
        noisy_edge_index = edge_index.clone().detach()
        idx_overall = torch.tensor(range(self.features.shape[0])).to(self.device)
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()
        for idx in idxs:
            idx_s = (noisy_edge_index[0] == idx).nonzero().flatten()
            # idx_edge_index = (data.edge_index[0] == idx_target).nonzero().flatten()
            idx_connected_nodes = noisy_edge_index[1][idx_s]
            idx_nonconnected_nodes = torch.tensor(list(set(np.array(idx_overall.cpu())) - set(np.array(idx_connected_nodes.cpu())))).to(self.device)
            idx_candidate_nodes = torch.cat([idx_connected_nodes,idx_nonconnected_nodes],dim=0)
            edge_weights_non_connected = torch.zeros([idx_nonconnected_nodes.shape[0],]).to(self.device)
            edge_weights_cur_nodes = torch.cat([noisy_edge_weight[idx_s],edge_weights_non_connected],dim=0)
            m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
            mask = m.sample(idx_candidate_nodes.shape).squeeze(-1).int()
            update_edge_weights_cur_nodes = torch.bitwise_not(torch.bitwise_xor(edge_weights_cur_nodes.int(),mask).to(torch.bool)).to(torch.float)
            # update the status of existing edges
            noisy_edge_weight[idx_s] = update_edge_weights_cur_nodes[:len(idx_s)]
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
            # add new edges
            # idx update_edge_weights_cur_nodes[len(idx_s):]
            idx_add_nodes = torch.nonzero(idx_nonconnected_nodes * update_edge_weights_cur_nodes[len(idx_s):]).flatten()
            add_edge_index = utils.single_add_random_edges(idx,idx_add_nodes,self.device)
            update_edge_index = torch.cat([noisy_edge_index,add_edge_index],dim=1)
            noisy_edge_index = update_edge_index
            if(noisy_edge_weight!=None):
                # noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
                noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        return noisy_edge_index, noisy_edge_weight
    def sample_noise_all_dense(self,args,edge_index,edge_weight,device):
        adj = to_dense_adj(edge_index,edge_attr=edge_weight)[0]
        row_idx, col_idx = np.triu_indices(adj.shape[1])
        rand_inputs = torch.randint_like(adj[row_idx,col_idx], low=0, high=2, device=device)
        adj_noise = torch.zeros(adj.shape, device=device)
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        mask = m.sample(adj[row_idx,col_idx].shape).squeeze(-1).int()
        adj_noise[row_idx,col_idx] = adj[row_idx,col_idx] * mask + rand_inputs * (1 - mask)
        adj_noise = adj_noise + adj_noise.t()
        ## diagonal elements set to be 0
        ind = np.diag_indices(adj_noise.shape[0]) 
        adj_noise[ind[0],ind[1]] = adj[ind[0], ind[1]]
        adj_noise = adj_noise.requires_grad_(True)
        edge_index, edge_weight = dense_to_sparse(adj_noise)
        return edge_index,edge_weight

    def sample_noise_all_sparse(self,args,edge_index,edge_weight,x):
        noisy_edge_index = edge_index.clone().detach()
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()
        # # rand_noise_data = copy.deepcopy(data)
        # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
        m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
        mask_connected = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
        num_edgeRemove = len((1-mask_connected).nonzero().flatten())
        num_totalEdgeState = x.shape[0] ** 2
        num_unconnectEdge = num_totalEdgeState - edge_index.shape[1]
        mask_unconnected = m.sample([num_unconnectEdge]).squeeze(-1).int()
        num_edgeAdd = len((1-mask_unconnected).nonzero().flatten())

        # rs = np.random.RandomState(args.seed)
        edge_index_to_add = np.random.randint(0, x.shape[0], (2, num_edgeAdd))
        edge_index_to_add = torch.tensor(edge_index_to_add).to(self.device)
        row = torch.cat([edge_index_to_add[0], edge_index_to_add[1]])
        col = torch.cat([edge_index_to_add[1],edge_index_to_add[0]])
        edge_index_to_add = torch.stack([row,col])
        updated_edge_index = torch.cat([edge_index,edge_index_to_add],dim=1)

        unconnected_edge_weight = torch.zeros([edge_index_to_add.shape[1],]).to(self.device)
        updated_edge_weight = torch.cat([noisy_edge_weight,unconnected_edge_weight],dim=0)
        rand_inputs = torch.randint_like(updated_edge_weight, low=0, high=2).squeeze().int().to(self.device)
        mask = m.sample(updated_edge_weight.shape).squeeze(-1).int()
        updated_edge_weight = updated_edge_weight * mask + rand_inputs * (1-mask)

        if(updated_edge_weight!=None):
            updated_edge_index = updated_edge_index[:,updated_edge_weight.nonzero().flatten().long()]
            updated_edge_weight = torch.ones([updated_edge_index.shape[1],]).to(self.device)
        return updated_edge_index, updated_edge_weight
    
    # def sample_noise_overall(self,edge_index, edge_weight,idxs):
    #     noisy_edge_index = edge_index.clone().detach()
    #     idx_overall = torch.tensor(range(self.data.num_nodes)).to(self.device)
    #     if(edge_weight == None):
    #         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
    #     else:
    #         noisy_edge_weight = edge_weight.clone().detach()
    #     # for idx in idxs:
    #     # idx_s = (noisy_edge_index[0] == idx).nonzero().flatten()
    #     # idx_edge_index = (data.edge_index[0] == idx_target).nonzero().flatten()
    #     idx_connected_nodes = noisy_edge_index.edge_index[1]
    #     idx_nonconnected_nodes = torch.tensor(list(set(np.array(idx_overall.cpu())) - set(np.array(idx_connected_nodes.cpu())))).to(self.device)
    #     idx_candidate_nodes = torch.cat([idx_connected_nodes,idx_nonconnected_nodes],dim=1)
    #     edge_weights_non_connected = torch.zeros([idx_nonconnected_nodes.shape[0],]).to(self.device)
    #     edge_weights_cur_nodes = torch.cat([noisy_edge_weight,edge_weights_non_connected],dim=1)
    #     m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
    #     mask = m.sample(idx_candidate_nodes.shape).squeeze(-1).int()
    #     update_edge_weights_cur_nodes = torch.bitwise_not(torch.bitwise_xor(edge_weights_cur_nodes,mask).to(torch.bool)).to(torch.int)
    #     # update the status of existing edges
    #     noisy_edge_weight[idx_s] = update_edge_weights_cur_nodes[:len(idx_s)]
    #     noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]

    #     # add new edges
    #     # idx update_edge_weights_cur_nodes[len(idx_s):]
    #     idx_add_nodes = torch.nonzero(idx_nonconnected_nodes * update_edge_weights_cur_nodes[len(idx_s):]).flatten()
    #     add_edge_index = utils.single_add_random_edges(idx,idx_add_nodes,self.device)
    #     update_edge_index = torch.cat([noisy_edge_index,add_edge_index],dim=1)
    #     noisy_edge_index = update_edge_index
    

    #     if(noisy_edge_weight!=None):
    #         noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
    #         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
    #     return noisy_edge_index, noisy_edge_weight

# %%
