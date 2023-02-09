#%%
# from msilib.schema import Class
# from turtle import forward
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

from construct_graph import construct_augmentation_1,construct_augmentation 
from copy import deepcopy
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import subgraph

class GCN_Encoder(nn.Module):

    def __init__(self, args, nfeat, nhid, nclass, unlabeled_idx, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(GCN_Encoder, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(nfeat, nhid))
        # for _ in range(layer-2):
        #     self.convs.append(GCNConv(nhid,nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        self.fc = nn.Linear(nhid,nclass).to(device)

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
        self.unlabeled_idx = unlabeled_idx
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

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200,seen_node_idx=None,verbose=False):
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
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.features, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation(self.args, self.features, self.edge_index, self.edge_weight, device= self.device)
            # print(edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2)
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_weight_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device),edge_weight_2.to(self.device)
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            # print(x_1, edge_index_1,edge_weight_1)
            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2

            # inv_loss_1 = self.inv_loss(h1,self.seen_node_idx)
            # inv_loss_2 = self.inv_loss(h2,self.seen_node_idx)
            # inv_loss = inv_loss_1 + inv_loss_2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            
            '''generate pseudo label'''
            select_unlabel_index1,select_label1,select_unlabel_index2,select_label2 = self.pseudo_label_generator(h1,h2,self.args.select_thrh,self.unlabeled_idx)
            idx_train1 = torch.concat([idx_train,select_unlabel_index1])
            pseudo_labels1 = labels.clone()
            pseudo_labels1[select_unlabel_index1] = select_label1

            idx_train2 = torch.concat([idx_train,select_unlabel_index2])
            pseudo_labels2 = labels.clone()
            pseudo_labels2[select_unlabel_index2] = select_label2

            inv_loss = self.inv_loss(h1,h2,pseudo_labels1,idx_train1,pseudo_labels2,idx_train2)

            '''calculate classfication loss'''
            # cont_embds = self.forward(self.features, self.edge_index, self.edge_weight)
            cont_embds = self.forward(x_1, edge_index_1,edge_weight_1)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            
            # clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            clf_loss = self.clf_loss(cont_embds,pseudo_labels1,idx_train1)

            cont_embds2 = self.forward(x_2, edge_index_2,edge_weight_2)
            clf_loss_2 = self.clf_loss(cont_embds2,pseudo_labels2,idx_train2)
            # clf_loss_2 = self.clf_loss(cont_embds2,labels,idx_train)
            # gap_loss = self.gap_loss(cont_embds,self.seen_node_idx)
            # print(inv_loss)
            loss =  self.cont_weight * cont_loss  + clf_loss + self.clf_weight * clf_loss_2 
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            cont_embds = self.forward(self.features, self.edge_index,self.edge_weight)
            # cont_embds = self.forward(x_1, edge_index_1,edge_weight_1)
            clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val
            output = self.clf_head(cont_embds)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                print('Epoch {}, val acc: {}'.format(i, acc_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            # if loss_val < best_loss_val:
            #     best_loss_val = loss_val
            #     # self.output = output
            #     weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                # self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        
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
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        h1 = z1
        h2 = z2

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
    
    def gap_loss(self, z, idx):
        # test_embds = test_model(poison_x,poison_edge_index,poison_edge_weights)
        output = F.softmax(self.clf_head(z))
        sorted_pred = torch.argsort(output,dim=1,descending=True)
        fst_prob = output[:,sorted_pred[:,0]][:,0]
        sec_prob = output[:,sorted_pred[:,1]][:,0]
        gap_loss = (fst_prob - sec_prob)[idx].mean()
        # if(inv_loss<0):
        #     inv_loss.data = torch.tensor(0).to(self.device)
        return gap_loss
    
    def inv_loss(self,z1,z2,label1,idx1,label2,idx2):
        output1 = self.clf_head(z1)
        output2 = self.clf_head(z2)
        ce_loss1 = F.nll_loss(output1[idx1],label1[idx1])
        ce_loss2 = F.nll_loss(output2[idx2],label2[idx2])
        ce_losses = torch.FloatTensor([ce_loss1,ce_loss2])
        inv_loss = torch.var(ce_losses,dim=0)
        return inv_loss
    
    def inv_loss1(self,z1,z2,idx):
        output1 = self.clf_head(z1)
        output2 = self.clf_head(z2)
        ce_losses = torch.tensor([],requires_grad=True,device=self.device)
        for i in idx:
            ce_loss1 = F.nll_loss(output1[i],self.labels[i])
            ce_loss2 = F.nll_loss(output2[i],self.labels[i])
            ce_loss = torch.FloatTensor([[ce_loss1,ce_loss2]]).to(self.device)
            ce_losses = torch.concat([ce_losses,ce_loss],dim=0)
        # print(ce_losses)
        inv_loss = torch.var(ce_losses,dim=1).sum()
        return inv_loss
        # pred1 = torch.argmax(output1,dim=1)
        # pred2 = torch.argmax(output2,dim=2)
        # fst_prob = output1[:,sorted_pred[:,0]][:,0].unsqueeze(dim=1)
        # print(fst_prob)
        # print(torch.concat([fst_prob,fst_prob],dim=1))
    def pseudo_label_generator(self,z1,z2,select_thrh,unlabel_idx):
        output1 = self.clf_head(z1[unlabel_idx])
        output2 = self.clf_head(z2[unlabel_idx])
        pred_prob1 = F.softmax(output1)
        pred_prob2 = F.softmax(output2)
        

        select_index1 = (pred_prob2.max(dim=1).values>select_thrh).nonzero().flatten()
        select_label1 = pred_prob2.max(dim=1).indices[select_index1]
        # print(unlabel_idx.shape,pred_prob2.max(dim=1).indices.shape)

        select_index2 = (pred_prob1.max(dim=1).values>select_thrh).nonzero().flatten()
        select_label2 = pred_prob1.max(dim=1).indices[select_index2]

        select_unlabel_index1 = unlabel_idx[select_index1]
        select_unlabel_index2 = unlabel_idx[select_index2]
        # sharpen
        # mean_pred_prob1 = torch.mean(F.softmax(output),dim=0)
        # target_pred_prob1 = torch.pow(mean_pred_prob,1./T)
        # target_pred_prob1 /= torch.mean(target_pred_prob)
        return select_unlabel_index1,select_label1,select_unlabel_index2,select_label2

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
        for _ in range(layer-1):
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
# %%
class BRGNN(nn.Module):

    def __init__(self, args, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, tau=None, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(BRGNN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        # self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(nfeat, nhid))
        # for _ in range(layer-2):
        #     self.convs.append(GCNConv(nhid,nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        self.body = GCN_body(nfeat, nhid, dropout, layer,device=device,use_ln=use_ln,layer_norm_first=layer_norm_first).to(device)
        self.fc = nn.Linear(nhid,nclass).to(device)

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

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200,seen_node_idx=None,verbose=False):
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
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.features, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation(self.args, self.features, self.edge_index, self.edge_weight, device= self.device)
            # print(edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2)
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_weight_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device),edge_weight_2.to(self.device)
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            # print(x_1, edge_index_1,edge_weight_1)
            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2

            # inv_loss_1 = self.inv_loss(h1,self.seen_node_idx)
            # inv_loss_2 = self.inv_loss(h2,self.seen_node_idx)
            # inv_loss = inv_loss_1 + inv_loss_2
            inv_loss = self.inv_loss(h1,h2,idx_train)
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            # cont_embds = self.forward(self.features, self.edge_index, self.edge_weight)
            cont_embds = self.forward(x_1, edge_index_1,edge_weight_1)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)

            cont_embds2 = self.forward(x_2, edge_index_2,edge_weight_2)
            clf_loss_2 = self.clf_loss(cont_embds2,labels,idx_train)
            # gap_loss = self.gap_loss(cont_embds,self.seen_node_idx)
            # print(inv_loss)
            loss =  self.cont_weight * cont_loss  + clf_loss + self.clf_weight * clf_loss_2 + inv_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            # cont_embds = self.forward(self.features, self.edge_index,self.edge_weight)
            cont_embds = self.forward(x_1, edge_index_1,edge_weight_1)
            clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val
            output = self.clf_head(cont_embds)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                print('Epoch {}, val acc: {}'.format(i, acc_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            # if loss_val < best_loss_val:
            #     best_loss_val = loss_val
            #     # self.output = output
            #     weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                # self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

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
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        h1 = z1
        h2 = z2

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
    
    def gap_loss(self, z, idx):
        # test_embds = test_model(poison_x,poison_edge_index,poison_edge_weights)
        output = F.softmax(self.clf_head(z))
        sorted_pred = torch.argsort(output,dim=1,descending=True)
        fst_prob = output[:,sorted_pred[:,0]][:,0]
        sec_prob = output[:,sorted_pred[:,1]][:,0]
        gap_loss = (fst_prob - sec_prob)[idx].mean()
        # if(inv_loss<0):
        #     inv_loss.data = torch.tensor(0).to(self.device)
        return gap_loss
    
    def inv_loss(self,z1,z2,idx):
        output1 = self.clf_head(z1)
        output2 = self.clf_head(z2)
        ce_loss1 = F.nll_loss(output1[idx],self.labels[idx])
        ce_loss2 = F.nll_loss(output2[idx],self.labels[idx])
        ce_losses = torch.FloatTensor([ce_loss1,ce_loss2])
        inv_loss = torch.var(ce_losses,dim=0)
        return inv_loss
    
    def inv_loss1(self,z1,z2,idx):
        output1 = self.clf_head(z1)
        output2 = self.clf_head(z2)
        ce_losses = torch.tensor([],requires_grad=True,device=self.device)
        for i in idx:
            ce_loss1 = F.nll_loss(output1[i],self.labels[i])
            ce_loss2 = F.nll_loss(output2[i],self.labels[i])
            ce_loss = torch.FloatTensor([[ce_loss1,ce_loss2]]).to(self.device)
            ce_losses = torch.concat([ce_losses,ce_loss],dim=0)
        # print(ce_losses)
        inv_loss = torch.var(ce_losses,dim=1).sum()
        return inv_loss

class Grace_Encoder(nn.Module):

    def __init__(self, args, nfeat, nhid, nclass, unlabeled_idx, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(Grace_Encoder, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
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
        self.fc1 = torch.nn.Linear(nhid, 128).to(device)
        self.fc2 = torch.nn.Linear(128, nhid).to(device)

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
        self.unlabeled_idx = unlabeled_idx
        self.args = args
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
        z = F.relu(self.fc1(z))
        return self.fc2(z)

        
    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200,seen_node_idx=None,verbose=False):
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
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
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
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.features, edge_index, edge_weight)
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation(self.args, self.features, edge_index, edge_weight, device= self.device)

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
        # print("----Training DT Classifier----")
        # # freeze parameter
        # for param in self.body.parameters():
        #     param.requires_grad = False
        
        # for i in range(train_iters):
        #     h = self.forward(self.features, edge_index, edge_weight)
        #     clf_loss = self.clf_loss(h,labels,idx_train)
        #     if verbose and i % 10 == 0:
        #         print('Epoch {}, classification loss: {}'.format(i, clf_loss.item()))
        #     clf_loss.backward()
        #     optimizer.step()
            
        #     self.eval()
        #     clf_loss_val = self.clf_loss(h,labels,idx_val)
        #     # loss_val = clf_loss_val + self.cont_weight * cont_loss
        #     loss_val = clf_loss_val
        #     output = self.clf_head(h)
        #     acc_val = utils.accuracy(output[idx_val], labels[idx_val])
        #     if verbose and i % 10 == 0:
        #         print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
        #         print('Epoch {}, val acc: {}'.format(i, acc_val.item()))
        #     if acc_val > best_acc_val:
        #         best_acc_val = acc_val
        #         # self.output = output
        #         weights = deepcopy(self.state_dict())
        # if verbose:
        #     print('=== picking the best model according to the performance on validation ===')
        # self.load_state_dict(weights)

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

class Smoothed_Encoder(nn.Module):

    def __init__(self, args, nfeat, nhid, nclass, unlabeled_idx, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,device=None,use_ln=False,layer_norm_first=False):

        super(Smoothed_Encoder, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
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
        self.fc1 = torch.nn.Linear(nhid, 128).to(device)
        self.fc2 = torch.nn.Linear(128, nhid).to(device)

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
        self.unlabeled_idx = unlabeled_idx
        self.args = args
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

        
    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200,seen_node_idx=None,verbose=False):
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
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
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
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.features, edge_index, edge_weight)
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation(self.args, self.features, edge_index, edge_weight, device= self.device)

            edge_index_1,edge_weight_1 = self.sample_noise_all(edge_index_1,edge_weight_1,idx_train)
    
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
        # print("----Training DT Classifier----")
        # # freeze parameter
        # for param in self.body.parameters():
        #     param.requires_grad = False
        
        # for i in range(train_iters):
        #     h = self.forward(self.features, edge_index, edge_weight)
        #     clf_loss = self.clf_loss(h,labels,idx_train)
        #     if verbose and i % 10 == 0:
        #         print('Epoch {}, classification loss: {}'.format(i, clf_loss.item()))
        #     clf_loss.backward()
        #     optimizer.step()
            
        #     self.eval()
        #     clf_loss_val = self.clf_loss(h,labels,idx_val)
        #     # loss_val = clf_loss_val + self.cont_weight * cont_loss
        #     loss_val = clf_loss_val
        #     output = self.clf_head(h)
        #     acc_val = utils.accuracy(output[idx_val], labels[idx_val])
        #     if verbose and i % 10 == 0:
        #         print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
        #         print('Epoch {}, val acc: {}'.format(i, acc_val.item()))
        #     if acc_val > best_acc_val:
        #         best_acc_val = acc_val
        #         # self.output = output
        #         weights = deepcopy(self.state_dict())
        # if verbose:
        #     print('=== picking the best model according to the performance on validation ===')
        # self.load_state_dict(weights)

    def _train_with_val_withtwohead(self, labels, idx_train, idx_val, train_iters, verbose):
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
            # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.features, edge_index, edge_weight)
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation(self.args, self.features, edge_index, edge_weight, device= self.device)

            edge_index_1_rs,edge_weight_1_rs = self.sample_noise_all(edge_index_1,edge_weight_1,idx_train)
    
            z1 = self.forward(x_1, edge_index_1,edge_weight_1)
            z2 = self.forward(x_2, edge_index_2,edge_weight_2)

            z1_rs = self.forward(x_1,edge_index_1_rs,edge_weight_1_rs)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2

            h1_rs = z1_rs

            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)

            if(self.seen_node_idx!=None):
                cont_loss_rs = self.loss(h1[self.seen_node_idx], h1_rs[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss_rs = self.loss(h1, h1_rs, batch_size=self.args.cont_batch_size)

            loss =  (cont_loss + cont_loss_rs)/2
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
            loss.backward()
            optimizer.step()

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
            noisy_edge_weight[idx_s] = noisy_edge_weight[idx_s] * mask + rand_inputs * (1-mask)
            # print(rand_noise_data.edge_weight.shape)
            # break

        if(noisy_edge_weight!=None):
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        return noisy_edge_index, noisy_edge_weight

    def sample_noise_all(self,edge_index, edge_weight,idxs):
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
        noisy_edge_weight = noisy_edge_weight * mask + rand_inputs * (1-mask)
            
        if(noisy_edge_weight!=None):
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(self.device)
        return noisy_edge_index, noisy_edge_weight

# %%
