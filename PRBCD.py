"""
Robustness of Graph Neural Networks at Scale. NeurIPS 2021.

Modified from https://github.com/sigeisler/robustness_of_gnns_at_scale/blob/main/rgnn_at_scale/attacks/prbcd.py
"""
import numpy as np
from deeprobust.graph.defense_pyg import GCN
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

from models.Grace import Grace

class PRBCD:

    def __init__(self, data, model=None,
            make_undirected=True,
            eps=1e-7, search_space_size=10_000_000,
            max_final_samples=20,
            fine_tune_epochs=100,
            epochs=400, lr_adj=0.1,
            with_early_stopping=True,
            do_synchronize=True,
            device='cuda:2',
            **kwargs
            ):
        """
        Parameters
        ----------
        data : pyg format data
        model : the model to be attacked, should be models in deeprobust.graph.defense_pyg
        """
        self.device = device
        self.data = data

        if model is None:
            model = self.pretrain_model()

        self.model = model
        nnodes = data.x.shape[0]
        d = data.x.shape[1]

        self.n, self.d = nnodes, nnodes
        self.make_undirected = make_undirected
        self.max_final_samples = max_final_samples
        self.search_space_size = search_space_size
        self.eps = eps
        self.lr_adj = lr_adj

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        # lr_factor = 0.1
        # self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs = epochs
        self.epochs_resampling = epochs - fine_tune_epochs # TODO

        self.with_early_stopping = with_early_stopping
        self.do_synchronize = do_synchronize

    def pretrain_model(self, model=None):
        data = self.data
        device = self.device
        feat, labels = data.x, data.y
        nclass = max(labels).item()+1

        if model is None:
            model = GCN(nfeat=feat.shape[1], nhid=256, dropout=0,
                    nlayers=3, with_bn=True, weight_decay=5e-4, nclass=nclass,
                    device=device).to(device)
            print(model)
        print(model(data.x,data.edge_index).shape)
        model.fit(data, train_iters=1000, patience=200, verbose=True)
        model.eval()
        model.data = data.to(self.device)
        output = model.predict()
        labels = labels.to(device)
        print(f"{model.name} Test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
        self.clean_node_mask = (output.argmax(1) == labels)
        return model


    def sample_random_block(self, n_perturbations):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations):
        best_loss = -float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        _, feat, labels = self.edge_index, self.data.x, self.data.y
        for i in range(self.max_final_samples):
            if best_loss == float('Inf') or best_loss == -float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                print(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                output = self.model.forward(feat, edge_index, edge_weight)
                loss = F.nll_loss(output[self.data.val_mask], labels[self.data.val_mask]).item()

            if best_loss < loss:
                best_loss = loss
                print('best_loss:', best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def resample_random_block(self, n_perturbations: int):
        self.keep_heuristic = 'WeightOnly'
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.search_space_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
                ] = perturbed_edge_weight_old # unique_idx: the indices for the old edges

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')


    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = -gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def _update_edge_weights(self, n_perturbations, epoch, gradient):
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(lr * gradient)
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps
        return None

    def attack(self, edge_index=None, edge_weight=None, ptb_rate=0.1):
        data = self.data
        epochs, lr_adj = self.epochs, self.lr_adj
        model = self.model
        model.eval() # should set to eval

        self.edge_index, feat, labels = data.edge_index, data.x, data.y
        with torch.no_grad():
            output = model.forward(feat, self.edge_index)
            pred = output.argmax(1)
        gt_labels = labels
        labels = labels.clone() # to avoid shallow copy
        labels[~data.train_mask] = pred[~data.train_mask]

        if edge_index is not None:
            self.edge_index = edge_index

        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(ptb_rate * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=lr_adj)
        best_loss_val = -float('Inf')
        for it in tqdm(range(epochs)):
            self.perturbed_edge_weight.requires_grad = True
            edge_index, edge_weight  = self.get_modified_adj()
            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            output = model.forward(feat, edge_index, edge_weight)
            loss = self.loss_attack(output, labels, type='tanhMargin')
            gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if it % 10 == 0:
                print(f'Epoch {it}: {loss}')

            with torch.no_grad():
                self.update_edge_weights(n_perturbations, it, gradient)
                self.perturbed_edge_weight = self.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)

                del edge_index, edge_weight #, logits

                if it < self.epochs_resampling - 1:
                    self.resample_random_block(n_perturbations)

                edge_index, edge_weight = self.get_modified_adj()
                output = model.predict(feat, edge_index, edge_weight)
                loss_val = F.nll_loss(output[data.val_mask], labels[data.val_mask])

            self.perturbed_edge_weight.requires_grad = True
            self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=lr_adj)

        # Sample final discrete graph
        edge_index, edge_weight = self.sample_final_edges(n_perturbations)
        output = model.predict(feat, edge_index, edge_weight)
        print('Test:')
        self.get_perf(output, gt_labels, data.test_mask)
        print('Validatoin:')
        self.get_perf(output, gt_labels, data.val_mask)
        return edge_index, edge_weight

    def loss_attack(self, logits, labels, type='CE'):
        self.loss_type = type
        if self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def get_perf(self, output, labels, mask, verbose=True):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        if verbose:
            print("loss= {:.4f}".format(loss.item()),
                  "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)

def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = lin_idx // n
    col_idx = lin_idx % n
    return torch.stack((row_idx, col_idx))

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))

def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu
# if __name__ == "__main__":
#     from torch_geometric.datasets import Planetoid
#     from ogb.nodeproppred import PygNodePropPredDataset
#     from torch_geometric.utils import to_undirected
#     import torch_geometric.transforms as T
#     # dataset = PygNodePropPredDataset(name='cora')
#     dataset = Planetoid(root='./data', name='Cora')
#     dataset.transform = T.NormalizeFeatures()
#     data = dataset[0]
#     if not hasattr(data, 'train_mask'):
#         utils.add_mask(data, dataset)
#     data.edge_index = to_undirected(data.edge_index, data.num_nodes)
#     # model = model_construct(args,args.encoder_model,data,device)
#     # model.fit(data.x, data.edge_index,data.edge_weight,data.y,train_iters=500,seen_node_idx=None,verbose=True)
#     agent = PRBCD(data)
#     edge_index, edge_weight = agent.attack()

if __name__ == "__main__":
    import yaml
    from yaml import SafeLoader
    import imp
    import time
    import argparse
    import numpy as np
    import torch

    from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Amazon, Coauthor, WikiCS


    # from torch_geometric.loader import DataLoader
    # from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
    import scipy.sparse as sp

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            default=True, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='GCN', help='propagation model for encoder',
                        choices=['GCN','GAT','GraphSage','GIN'])
    parser.add_argument('--if_smoothed', action='store_true', default=False)
    parser.add_argument('--encoder_model', type=str, default='Grace', help='propagation model for encoder',
                        choices=['Grace','GraphCL','BGRL','DGI','GAE','Node2vec'])
    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='Dataset',
                        choices=['Cora','Citeseer','Pubmed','Flickr','ogbn-arxiv','Computers','Photo','CS','Physics','WikiCS'])
    parser.add_argument('--train_lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units of backdoor model.')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--num_proj_hidden', type=int, default=128,
                        help='Number of hidden units in MLP.')
    # parser.add_argument('--thrd', type=float, default=0.5)
    # parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--test_model', type=str, default='GCN',
                        choices=['GCN','GAT','GraphSage','GIN'],
                        help='Model used to attack')
    # GPU setting
    parser.add_argument('--device_id', type=int, default=2,
                        help="Threshold of prunning edges")
    # Contrastive Learning setting
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--cl_lr', type=float, default=0.0005)
    # parser.add_argument('--cl_num_proj_hidden', type=int, default=128)
    parser.add_argument('--cl_num_layers', type=int, default=2)
    parser.add_argument('--cl_activation', type=str, default='relu')
    parser.add_argument('--cl_base_model', type=str, default='GCNConv')
    parser.add_argument('--cont_weight', type=float, default=1)
    parser.add_argument('--add_edge_rate_1', type=float, default=0)
    parser.add_argument('--add_edge_rate_2', type=float, default=0)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0)
    parser.add_argument('--drop_feat_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_feat_rate_2', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cl_num_epochs', type=int, default=200)
    parser.add_argument('--cl_weight_decay', type=float, default=1e-5)
    parser.add_argument('--cont_batch_size', type=int, default=0)
    parser.add_argument('--noisy_level', type=float, default=0.3)
    parser.add_argument('--clf_weight', type=float, default=1)
    parser.add_argument('--inv_weight', type=float, default=1)
    # parser.add_argument('--select_thrh', type=float, default=0.8)

    # Attack
    parser.add_argument('--attack', type=str, default='none',
                        choices=['nettack','random','none', 'PRBCD'],)
    parser.add_argument('--select_target_ratio', type=float, default=0.15,
                        help="The number of selected target test nodes for targeted attack")
    # Randomized Smoothing
    parser.add_argument('--prob', default=0.8, type=float,
                        help="probability to keep the status for each binary entry")
    parser.add_argument('--num_sample', default=20, type=int,
                        help="the number of noisy samples to calculate the bounded probability")
    parser.add_argument('--if_keep_structure1', action='store_true', default=False)
    parser.add_argument('--if_ignore_structure2', action='store_true', default=False)
    parser.add_argument('--sample_way', type=str, default='random_drop',
                        choices=['random_drop','keep_structure'])
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    args.cuda =  not args.no_cuda and torch.cuda.is_available()
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

    np.random.seed(args.seed)


    from torch_geometric.utils import to_undirected
    import torch_geometric.transforms as T
    transform = T.Compose([T.NormalizeFeatures()])

    if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/',                         name=args.dataset,                        transform=transform)
    elif(args.dataset == 'Computers' or args.dataset == 'Photo'):
        dataset = Amazon(root='./data/', name=args.dataset,                    transform=transform)
    elif(args.dataset == 'CS' or args.dataset == 'Physics'):
        dataset = Coauthor(root='./data/', name=args.dataset,                    transform=transform)  
    elif(args.dataset == 'WikiCS'):
        dataset = WikiCS(root='./data/WikiCS', transform=transform)    
    elif(args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/',                     transform=transform)
    elif(args.dataset == 'Reddit2'):
        dataset = Reddit2(root='./data/Reddit2/',                     transform=transform)
    elif(args.dataset == 'ogbn-arxiv'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
        split_idx = dataset.get_idx_split() 

    data = dataset[0].to(device)


    # from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
    # dr_data = Dataset(root='/tmp/', name='pubmed') # load clean graph
    # data = Dpr2Pyg(dr_data)
    # data = data[0].to(device)

    # we build our own train test split 
    if(args.dataset == 'ogbn-arxiv'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.y = data.y.squeeze(1)
    elif(args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'CS' or args.dataset == 'Physics'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    # elif(args.dataset == 'WikiCS'):
    #     data.train_mask = data.train_mask[:,0]
    #     data.val_mask = data.val_mask[:,0]
    #     data.stopping_mask = data.stopping_mask[:,0]

    # In[14]:

    print(data)
    from utils import get_split
    if(args.dataset == 'Computers' or args.dataset == 'Photo'):
        data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
    # data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
    idx_train = data.train_mask.nonzero().flatten()
    idx_val = data.val_mask.nonzero().flatten()
    idx_clean_test = data.test_mask.nonzero().flatten()

    from torch_geometric.utils import to_undirected
    from utils import subgraph
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
    mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
    # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
    # unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()

    # In[28]:
    if(args.if_smoothed == True):
        config_path = "./config/config_Smooth_{}.yaml".format(args.encoder_model)    
    else:
        config_path = "./config/config_{}.yaml".format(args.encoder_model)
    config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset]

    args.drop_edge_rate_1 = config['drop_edge_rate_1']
    args.drop_edge_rate_2 = config['drop_edge_rate_2']
    args.drop_feat_rate_1 = config['drop_feat_rate_1']
    args.drop_feat_rate_2 = config['drop_feat_rate_2']
    args.tau = config['tau']
    args.cl_lr = config['cl_lr']
    args.weight_decay = config['weight_decay']
    args.cl_num_epochs = config['cl_num_epochs']
    # args.cl_num_epochs = 1
    args.num_hidden = config['num_hidden']
    args.num_proj_hidden = config['num_proj_hidden']

    print(args)
    from models.construct import model_construct
    from eval import label_evaluation
    model = model_construct(args,args.encoder_model,data,device)
    model.fit(data.x, data.edge_index,data.edge_weight,data.y,train_iters=args.cl_num_epochs,seen_node_idx=None,verbose=True)
    model.eval()
    # data.x = model(data.x, data.edge_index, data.edge_weight).detach()
    # print(data.x.shape)
    print(data)
    z = model(data.x, data.edge_index,data.edge_weight)
    acc = label_evaluation(z, data.y, idx_train, idx_clean_test)
    print(acc)
    agent = PRBCD(data,model = None, device = device)
    edge_index, edge_weight = agent.attack(ptb_rate=0.25)
    z = model(data.x, edge_index,edge_weight)
    acc = label_evaluation(z, data.y, idx_train, idx_clean_test)
    print(acc)