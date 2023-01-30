import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from construct_graph import construct_augmentation_1,construct_augmentation
from torch import optim
from copy import deepcopy
import utils
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


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        # self.base_model = base_model
        self.base_model = ({'GCNConv': GCNConv})[base_model]
        assert k >= 2
        self.k = k
        self.conv = [self.base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(self.base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(self.base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[activation]
        # self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights=None):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index,edge_weights))
        return x


class Encoder_new(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, mlp_hidden = 0, gcn_hidden = 0):
        super(Encoder_new, self).__init__()
        # self.base_model = base_model
        self.fc1 = torch.nn.Linear(in_channels, mlp_hidden)
        self.fc2 = torch.nn.Linear(mlp_hidden, gcn_hidden)

        self.base_model = ({'GCNConv': GCNConv})[base_model]
        assert k >= 2
        self.k = k
        self.conv = [self.base_model(gcn_hidden, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(self.base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(self.base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[activation]

        # self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights=None):
        z = F.elu(self.fc1(x))
        z = F.elu(self.fc2(z))
        z_origin = z.clone()
        for i in range(self.k):
            z = self.activation(self.conv[i](z, edge_index, edge_weights))
        z = z+ z_origin
        return z

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weights=None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weights)
        # return x

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

        # between_sim.diag() + (between_sim.sum(1)-between_sim.diag()) + (refl_sim.sum(1) - refl_sim.diag())
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

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class UnifyModel(torch.nn.Module):
    def __init__(self, args, encoder: Encoder, num_hidden: int, num_proj_hidden: int, cl_num_proj_hidden: int, num_class: int,
                 tau: float = 0.5, cont_lr=0.0001, cont_weight_decay=0.00001, lr=0.01, weight_decay=5e-4, device=None,data1=None,data2=None):
        super(UnifyModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, cl_num_proj_hidden)
        self.fc2 = torch.nn.Linear(cl_num_proj_hidden, num_hidden)

        self.fc1_c = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2_c = torch.nn.Linear(num_proj_hidden, num_class)

        self.args = args
        self.cont_lr = cont_lr
        self.cont_weight_decay = cont_weight_decay
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.data1 = data1
        self.data2 = data2
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weights=None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weights)
        # return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def clf_head(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1_c(z))
        return self.fc2_c(z)
        # return z

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
    
    def fit(self, args, x, edge_index,edge_weight,labels,idx_train,idx_val=None,train_iters=200,cont_iters=None,seen_node_idx = None):
        self.args = args
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.x = x
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seen_node_idx = seen_node_idx
        self.cont_weight = args.cont_weight
        

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters,verbose=True)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters)
            # self._train_with_val_duo(self.labels, idx_train, idx_val, train_iters)
            # self._train_with_val_2(self.labels, idx_train, idx_val, train_iters,verbose=True)

    def fit_1(self, args, x, edge_index,edge_weight,labels,idx_train,idx_val=None,train_iters=200,cont_iters=None,seen_node_idx = None):
        self.args = args
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.x = x
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seen_node_idx = seen_node_idx
        self.cont_weight = args.cont_weight
        

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters,verbose=True)
        else:
            # self._train_with_val(self.labels, idx_train, idx_val, train_iters)
            self._train_with_val_2(self.labels, idx_train, idx_val, train_iters,verbose=True)
        
    def clf_loss(self, z: torch.Tensor, labels, idx):
        # h = self.projection(z)
        h = z
        output = self.clf_head(h)
        clf_loss = F.cross_entropy(output[idx],labels[idx])
        return clf_loss
    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=0)
            else:
                cont_loss = self.loss(h1, h2, batch_size=0)
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.clf_head(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss + self.cont_weight * cont_loss
            loss.backward() 
            optimizer.step()
            # return loss.item()
            if verbose and i % 10 == 0:
                # print('Epoch {}, training loss: {}'.format(i, loss.item()))
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            # self.eval()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
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
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss  + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
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
        
    def _train_with_val_2(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_std(self.data1,self.data2)
        edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
            clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_val_1(self, labels, idx_train, idx_val, cont_iters, train_iters, verbose=True):
        if verbose:
            print('=== training contrastive ===')
        optimizer = optim.Adam(self.parameters(), lr=self.cont_lr, weight_decay=self.cont_weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(cont_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(z1[self.seen_node_idx], z2[self.seen_node_idx], batch_size=0)
            else:
                cont_loss = self.loss(z1, z2, batch_size=0)
            if verbose and i % 10 == 0:
                print('Epoch {}, contrastive loss: {}'.format(i, cont_loss.item()))
                
            # cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.nll_loss(output[idx_train],labels[idx_train])
            # loss = cont_loss + self.clf_weight * clf_loss
            cont_loss.backward()
            optimizer.step()
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if verbose:
            print('=== training classification ===')
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # loss = F.cross_entropy(output[idx_train],labels[idx_train])
            loss = self.clf_loss(cont_embds,labels,idx_train)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
            # loss = cont_loss + self.clf_weight * clf_loss
            loss.backward()
            optimizer.step()

            self.eval()
            cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

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
        cont_embds = self.forward(features, edge_index, edge_weight)
        output = self.clf_head(cont_embds)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def _train_with_val_duo(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            # cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            clf_loss_1 = self.clf_loss(h1,labels,idx_train)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss_2 = self.clf_loss(h2,labels,idx_train)
            loss = clf_loss_1+ clf_loss_2 + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                # print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss_1.item()+clf_loss_2.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            clf_loss_val_1 = self.clf_loss(h1,labels,idx_val)
            clf_loss_val_2 = self.clf_loss(h2,labels,idx_val)
            # cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
            # clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val_1 + clf_loss_val_2
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

class SmoothModel(torch.nn.Module):
    def __init__(self, args, encoder: Encoder, num_hidden: int, num_proj_hidden: int, cl_num_proj_hidden: int, num_class: int,
                 tau: float = 0.5, cont_lr=0.0001, cont_weight_decay=0.00001, lr=0.01, weight_decay=5e-4, device=None,data1=None,data2=None):
        super(SmoothModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, cl_num_proj_hidden)
        self.fc2 = torch.nn.Linear(cl_num_proj_hidden, num_hidden)

        self.fc1_c = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2_c = torch.nn.Linear(num_proj_hidden, num_class)

        self.args = args
        self.cont_lr = cont_lr
        self.cont_weight_decay = cont_weight_decay
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.data1 = data1
        self.data2 = data2
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weights=None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weights)
        # return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def clf_head(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1_c(z))
        return self.fc2_c(z)
        # return z

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
    
    def fit(self, args, x, edge_index,edge_weight,labels,idx_train,idx_val=None,train_iters=200,cont_iters=None,seen_node_idx = None):
        self.args = args
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.x = x
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seen_node_idx = seen_node_idx
        self.cont_weight = args.cont_weight
        

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters,verbose=True)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters)
            # self._train_with_val_duo(self.labels, idx_train, idx_val, train_iters)
            # self._train_with_val_2(self.labels, idx_train, idx_val, train_iters,verbose=True)

    def fit_1(self, args, x, edge_index,edge_weight,labels,idx_train,idx_val=None,train_iters=200,cont_iters=None,seen_node_idx = None):
        self.args = args
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.x = x
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seen_node_idx = seen_node_idx
        self.cont_weight = args.cont_weight
        

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters,verbose=True)
        else:
            # self._train_with_val(self.labels, idx_train, idx_val, train_iters)
            self._train_with_val_2(self.labels, idx_train, idx_val, train_iters,verbose=True)
        
    def clf_loss(self, z: torch.Tensor, labels, idx):
        # h = self.projection(z)
        h = z
        output = self.clf_head(h)
        clf_loss = F.cross_entropy(output[idx],labels[idx])
        return clf_loss
    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=0)
            else:
                cont_loss = self.loss(h1, h2, batch_size=0)
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.clf_head(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss + self.cont_weight * cont_loss
            loss.backward() 
            optimizer.step()
            # return loss.item()
            if verbose and i % 10 == 0:
                # print('Epoch {}, training loss: {}'.format(i, loss.item()))
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            # self.eval()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            edge_index_1,x_1,edge_weight_1,edge_index_2,x_2,edge_weight_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
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
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss  + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
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
        
    def _train_with_val_2(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_std(self.data1,self.data2)
        edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss = self.clf_loss(cont_embds,labels,idx_train)
            loss = clf_loss + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
            clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_val_1(self, labels, idx_train, idx_val, cont_iters, train_iters, verbose=True):
        if verbose:
            print('=== training contrastive ===')
        optimizer = optim.Adam(self.parameters(), lr=self.cont_lr, weight_decay=self.cont_weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(cont_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(z1[self.seen_node_idx], z2[self.seen_node_idx], batch_size=0)
            else:
                cont_loss = self.loss(z1, z2, batch_size=0)
            if verbose and i % 10 == 0:
                print('Epoch {}, contrastive loss: {}'.format(i, cont_loss.item()))
                
            # cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # clf_loss = F.nll_loss(output[idx_train],labels[idx_train])
            # loss = cont_loss + self.clf_weight * clf_loss
            cont_loss.backward()
            optimizer.step()
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if verbose:
            print('=== training classification ===')
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # loss = F.cross_entropy(output[idx_train],labels[idx_train])
            loss = self.clf_loss(cont_embds,labels,idx_train)
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
            # loss = cont_loss + self.clf_weight * clf_loss
            loss.backward()
            optimizer.step()

            self.eval()
            cont_embds = self(self.x, self.edge_index, self.edge_weight)
            # output = self.projection(cont_embds)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

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
        cont_embds = self.forward(features, edge_index, edge_weight)
        output = self.clf_head(cont_embds)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)

    def _train_with_val_duo(self, labels, idx_train, idx_val, train_iters, verbose=True):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.data1,self.data2)
        # edge_index_1,x_1,edge_index_2,x_2 = edge_index_1.to(self.device),x_1.to(self.device),edge_index_2.to(self.device),x_2.to(self.device)
        edge_index_1,x_1,edge_index_2,x_2 = construct_augmentation_1(self.args, self.x, self.edge_index, self.edge_weight)
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            # output = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train.backward()
            # optimizer.step()
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            # h1 = self.projection(z1)
            # h2 = self.projection(z2)
            h1 = z1
            h2 = z2
            if(self.seen_node_idx!=None):
                cont_loss = self.loss(h1[self.seen_node_idx], h2[self.seen_node_idx], batch_size=self.args.cont_batch_size)
            else:
                cont_loss = self.loss(h1, h2, batch_size=self.args.cont_batch_size)
            # cont_embds = self.forward(self.x, self.edge_index, self.edge_weight)
            clf_loss_1 = self.clf_loss(h1,labels,idx_train)
            # output = self.projection(cont_embds)
            # clf_loss = F.cross_entropy(output[idx_train],labels[idx_train])
            clf_loss_2 = self.clf_loss(h2,labels,idx_train)
            loss = clf_loss_1+ clf_loss_2 + self.cont_weight * cont_loss
            if verbose and i % 10 == 0:
                # print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss.item(),self.cont_weight,cont_loss.item()))
                print('Epoch {}, training loss: {} = {} + {} * {}'.format(i, loss.item(),clf_loss_1.item()+clf_loss_2.item(),self.cont_weight,cont_loss.item()))
            loss.backward()
            optimizer.step()


            self.eval()
            clf_loss_val_1 = self.clf_loss(h1,labels,idx_val)
            clf_loss_val_2 = self.clf_loss(h2,labels,idx_val)
            # cont_embds = self.forward(self.x, self.edge_index,self.edge_weight)
            # clf_loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = clf_loss_val + self.cont_weight * cont_loss
            loss_val = clf_loss_val_1 + clf_loss_val_2
            if verbose and i % 10 == 0:
                print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
            # loss_val = self.clf_loss(cont_embds,labels,idx_val)
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            #     print('Epoch {}, val loss: {}'.format(i, loss_val.item()))
                # print("acc_val: {:.4f}".format(acc_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # self.output = output
                weights = deepcopy(self.state_dict())

            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)