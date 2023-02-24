import torch 
from torch.distributions.bernoulli import Bernoulli
import utils
import numpy as np
from torch_geometric.utils import to_undirected, to_dense_adj,to_torch_coo_tensor,dense_to_sparse

def sample_noise(args,edge_index, edge_weight,idxs, device):
    noisy_edge_index = edge_index.clone().detach()
    if(edge_weight == None):
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
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
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        # print(rand_noise_data.edge_weight[idx_s])
        # print(rand_noise_data.edge_weight)
        # break
        mask = m.sample(noisy_edge_weight[idx_s].shape).squeeze(-1).int()
        # print(mask)
        rand_inputs = torch.randint_like(noisy_edge_weight[idx_s], low=0, high=2).squeeze().int().to(device)
        # print(rand_noise_data.edge_weight.shape,mask.shape)
        noisy_edge_weight[idx_s] = noisy_edge_weight[idx_s] * mask + rand_inputs * (1-mask)
        # print(rand_noise_data.edge_weight.shape)
        # break

    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight
def sample_noise_all(args,edge_index, edge_weight,device):
        noisy_edge_index = edge_index.clone().detach()
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()
        # # rand_noise_data = copy.deepcopy(data)
        # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        mask = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
        rand_inputs = torch.randint_like(noisy_edge_weight, low=0, high=2).squeeze().int().to(device)
        # print(rand_noise_data.edge_weight.shape,mask.shape)
        noisy_edge_weight = noisy_edge_weight * mask + rand_inputs * (1-mask)
            
        if(noisy_edge_weight!=None):
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
        return noisy_edge_index, noisy_edge_weight
def sample_noise_1by1(args, x, edge_index, edge_weight,idxs,device):
    noisy_edge_index = edge_index.clone().detach()
    idx_overall = torch.tensor(range(x.shape[0])).to(device)
    if(edge_weight == None):
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    else:
        noisy_edge_weight = edge_weight.clone().detach()
    for idx in idxs:
        idx_s = (noisy_edge_index[0] == idx).nonzero().flatten()
        # idx_edge_index = (data.edge_index[0] == idx_target).nonzero().flatten()
        idx_connected_nodes = noisy_edge_index[1][idx_s]
        idx_nonconnected_nodes = torch.tensor(list(set(np.array(idx_overall.cpu())) - set(np.array(idx_connected_nodes.cpu())))).to(device)
        idx_candidate_nodes = torch.cat([idx_connected_nodes,idx_nonconnected_nodes],dim=0)
        edge_weights_non_connected = torch.zeros([idx_nonconnected_nodes.shape[0],]).to(device)
        edge_weights_cur_nodes = torch.cat([noisy_edge_weight[idx_s],edge_weights_non_connected],dim=0)
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        mask = m.sample(idx_candidate_nodes.shape).squeeze(-1).int()
        update_edge_weights_cur_nodes = torch.bitwise_not(torch.bitwise_xor(edge_weights_cur_nodes.int(),mask).to(torch.bool)).to(torch.float)
        # update the status of existing edges
        noisy_edge_weight[idx_s] = update_edge_weights_cur_nodes[:len(idx_s)]
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
        # add new edges
        # idx update_edge_weights_cur_nodes[len(idx_s):]
        idx_add_nodes = torch.nonzero(idx_nonconnected_nodes * update_edge_weights_cur_nodes[len(idx_s):]).flatten()
        add_edge_index = utils.single_add_random_edges(idx,idx_add_nodes,device)
        update_edge_index = torch.cat([noisy_edge_index,add_edge_index],dim=1)
        noisy_edge_index = update_edge_index
        if(noisy_edge_weight!=None):
            # noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight
    
def sample_noise_all_dense(args,edge_index,edge_weight,device):
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
    edge_index, edge_weight = dense_to_sparse(adj_noise)
    return edge_index,edge_weight

def sample_noise_1by1_dense(args,edge_index,edge_weight,idx,device):
    adj = to_dense_adj(edge_index,edge_attr=edge_weight)[0]
    adj_noise = adj.clone().detach()
    # row_idx, col_idx = np.triu_indices(adj.shape[1])
    rand_inputs = torch.randint_like(adj[idx], low=0, high=2, device=device)
    # adj_noise = torch.zeros(adj.shape, device=device)
    m = Bernoulli(torch.tensor([args.prob]).to(device))
    mask = m.sample(adj[idx].shape).squeeze(-1).int()
    adj_noise[idx] = adj[idx] * mask + rand_inputs * (1 - mask)
    adj_noise[idx,idx] = adj[idx,idx]
    # print(adj_noise)
    adj_noise[:,idx] = adj_noise[idx].t()
    # adj_noise = adj_noise + adj_noise.t()
    # ## diagonal elements set to be 0
    # ind = np.diag_indices(adj_noise.shape[0]) 
    # adj_noise[ind[0],ind[1]] = adj[ind[0], ind[1]]
    edge_index, edge_weight = dense_to_sparse(adj_noise)
    return edge_index,edge_weight