#%%
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI

import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import Node2Vec

def main(data,lr,device):
    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    # dataset = Planetoid(path, dataset)
    # data = dataset[0]

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    # Training and test
    for epoch in range(1, 101):
        best_acc = 0
        loss = train()
        acc = test()
        if(acc>best_acc):
            best_acc = acc
        if(epoch%10==0):
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    print(f'Best Acc: {best_acc:.4f}')
    return best_acc
    # @torch.no_grad()
    # def plot_points(colors):
    #     model.eval()
    #     z = model(torch.arange(data.num_nodes, device=device))
    #     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    #     y = data.y.cpu().numpy()

    #     plt.figure(figsize=(8, 8))
    #     for i in range(dataset.num_classes):
    #         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    #     plt.axis('off')
    #     plt.show()

    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700'
    # ]
    # plot_points(colors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            default=True, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--num_repeat', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='Dataset',
                        choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
    parser.add_argument('--train_lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units of backdoor model.')
    # parser.add_argument('--thrd', type=float, default=0.5)
    # parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    # GPU setting
    parser.add_argument('--device_id', type=int, default=2,
                        help="Threshold of prunning edges")

    # Attack
    parser.add_argument('--attack', type=str, default='none',
                        choices=['nettack','random','none'],)
    parser.add_argument('--select_target_ratio', type=float, default=0.1,
                        help="The number of selected target test nodes for targeted attack")

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    args.cuda =  not args.no_cuda and torch.cuda.is_available()
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    print(args)


    # In[13]:


    from torch_geometric.utils import to_undirected
    import torch_geometric.transforms as T
    transform = T.Compose([T.NormalizeFeatures()])

    if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/',                         name=args.dataset,                        transform=transform)
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

    if(args.dataset == 'ogbn-arxiv'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.y = data.y.squeeze(1)
    # we build our own train test split 


    # In[14]:


    # from utils import get_split
    # # data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
    idx_train = data.train_mask.nonzero().flatten()
    idx_val = data.val_mask.nonzero().flatten()
    idx_clean_test = data.test_mask.nonzero().flatten()
    if(args.attack == 'random'):
        perturbation_sizes = list(range(0,21))
        accuracys = {}
        for n_perturbation in perturbation_sizes:
            accuracys[n_perturbation] = []
    
    
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=args.num_repeat)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("seed {}".format(seed))
        if(args.attack == 'random'):
            import construct_graph
            import copy
            perturbation_sizes = list(range(0,21))
            for n_perturbation in perturbation_sizes:
                print("Perturbation Size:{}".format(n_perturbation))
                noisy_data = copy.deepcopy(data)
                if(n_perturbation > 0):
                    for idx in idx_clean_test:
                        noisy_data = construct_graph.generate_node_noisy(args,noisy_data,idx,n_perturbation,device)
                        noisy_data = noisy_data.to(device)
                # z = model(noisy_data.x, noisy_data.edge_index,noisy_data.edge_weight)
                # acc = label_evaluation(z, noisy_data.y, idx_train, idx_clean_test)
                # print("Accuracy:",acc)
                acc = main(noisy_data,args.train_lr,device)
                accuracys[n_perturbation].append(acc)
    if(args.attack == 'random'):
        for n_perturbation in perturbation_sizes:
            mean_acc =  np.mean(accuracys[n_perturbation])  
            std_acc =  np.std(accuracys[n_perturbation])     
            print("Ptb size:{} Accuracy:{}+-{}".format(n_perturbation,mean_acc,std_acc))
