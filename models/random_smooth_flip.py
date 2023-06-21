import torch 
from torch.distributions.bernoulli import Bernoulli
import utils
import numpy as np
from torch_geometric.utils import to_undirected, to_dense_adj,to_torch_coo_tensor,dense_to_sparse
import scipy.special as sp_special
import torch_geometric.utils as pyg_utils

def sample_noise_all_sparse(args,edge_index,edge_weight,x,device):
        noisy_edge_index = edge_index.clone().detach()
        if(edge_weight == None):
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
        else:
            noisy_edge_weight = edge_weight.clone().detach()
        # # rand_noise_data = copy.deepcopy(data)
        # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        mask_connected = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
        num_edgeRemove = len((1-mask_connected).nonzero().flatten())
        num_totalEdgeState = x.shape[0] ** 2
        num_unconnectEdge = num_totalEdgeState - edge_index.shape[1]
        mask_unconnected = m.sample([num_unconnectEdge]).squeeze(-1).int()
        num_edgeAdd = len((1-mask_unconnected).nonzero().flatten())

        # rs = np.random.RandomState(args.seed)
        edge_index_to_add = np.random.randint(0, x.shape[0], (2, num_edgeAdd))
        edge_index_to_add = torch.tensor(edge_index_to_add).to(device)
        row = torch.cat([edge_index_to_add[0], edge_index_to_add[1]])
        col = torch.cat([edge_index_to_add[1],edge_index_to_add[0]])
        edge_index_to_add = torch.stack([row,col])
        updated_edge_index = torch.cat([edge_index,edge_index_to_add],dim=1)

        unconnected_edge_weight = torch.zeros([edge_index_to_add.shape[1],]).to(device)
        updated_edge_weight = torch.cat([noisy_edge_weight,unconnected_edge_weight],dim=0)
        rand_inputs = torch.randint_like(updated_edge_weight, low=0, high=2).squeeze().int().to(device)
        mask = m.sample(updated_edge_weight.shape).squeeze(-1).int()
        updated_edge_weight = updated_edge_weight * mask + rand_inputs * (1-mask)

        if(updated_edge_weight!=None):
            updated_edge_index = updated_edge_index[:,updated_edge_weight.nonzero().flatten().long()]
            updated_edge_weight = torch.ones([updated_edge_index.shape[1],]).to(device)
        return updated_edge_index, updated_edge_weight

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
        noisy_edge_weight[idx_s] = noisy_edge_weight[idx_s] * mask #+ rand_inputs * (1-mask)


    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight

def sample_noise_mask(args,edge_index, edge_weight,idxs, device):
    noisy_edge_index = edge_index.clone().detach()
    if(edge_weight == None):
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    else:
        noisy_edge_weight = edge_weight.clone().detach()
    num_remain_edges = []
    for idx in idxs:
        idx_s = (noisy_edge_index[0] == idx).nonzero().flatten()
        m = Bernoulli(torch.tensor([args.prob]).to(device))
        mask = m.sample(noisy_edge_weight[idx_s].shape).squeeze(-1).int()
        rand_inputs = torch.randint_like(noisy_edge_weight[idx_s], low=0, high=2).squeeze().int().to(device)
        noisy_edge_weight[idx_s] = noisy_edge_weight[idx_s] * mask #+ rand_inputs * (1-mask)
        # calculate the remaining edges
        e = noisy_edge_weight[idx_s].count_nonzero().item()
        num_remain_edges.append(e)
    num_remain_edges = np.array(num_remain_edges)
    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight, num_remain_edges

def sample_noise_all_graph(args,edge_index, edge_weight,batch,device):
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
    noisy_edge_weight = noisy_edge_weight * mask #+ rand_inputs * (1-mask)
        
    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        # batch = batch[noisy_edge_index[0].unique()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight, batch

# def sample_noise_all_graph(args,edge_index, edge_weight, edge_attr, device):
#     noisy_edge_index = edge_index.clone().detach()
#     if(edge_attr!=None):
#         noisy_edg_attr = edge_attr.clone().detach()
#     else:
#         noisy_edg_attr = edge_attr
#     if(edge_weight == None):
#         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
#     else:
#         noisy_edge_weight = edge_weight.clone().detach()
#     # # rand_noise_data = copy.deepcopy(data)
#     # rand_noise_data.edge_weight = torch.ones([rand_noise_data.edge_index.shape[1],]).to(device)
#     m = Bernoulli(torch.tensor([args.prob]).to(device))
#     mask = m.sample(noisy_edge_weight.shape).squeeze(-1).int()
#     rand_inputs = torch.randint_like(noisy_edge_weight, low=0, high=2).squeeze().int().to(device)
#     # print(rand_noise_data.edge_weight.shape,mask.shape)
#     noisy_edge_weight = noisy_edge_weight * mask #+ rand_inputs * (1-mask)
        
#     if(noisy_edge_weight!=None):
#         noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
#         if(edge_attr != None):
#             print(noisy_edge_index.shape,noisy_edg_attr.shape)
#             noisy_edg_attr = noisy_edg_attr[noisy_edge_weight.nonzero().flatten().long(),:]
#         noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
#     return noisy_edge_index, noisy_edge_weight, noisy_edg_attr

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
    noisy_edge_weight = noisy_edge_weight * mask #+ rand_inputs * (1-mask)
        
    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
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

def sample_noise_sparse(args, x, edge_index, edge_weight,idxs,device):
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
        mask = m.sample(edge_weights_cur_nodes.shape).squeeze(-1).int()
        rand_inputs = torch.randint_like(edge_weights_cur_nodes, low=0, high=2, device=device).squeeze().int()
        
        noisy_edge_weight = edge_weights_cur_nodes * mask + rand_inputs * (1-mask)
        edge_index_to_add = torch.LongTensor([[idx]*len(idx_nonconnected_nodes),idx_nonconnected_nodes]).to(device)
        noisy_edge_index = torch.cat([noisy_edge_index,edge_index_to_add],dim=1)
        if(noisy_edge_weight!=None):
            noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
            noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight

from statsmodels.stats.proportion import proportion_confint

def _lower_confidence_bound(NA: int, N: int, alpha: float):
    """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
    This function uses the Clopper-Pearson method.
    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

def calculate_p_bound(args, model, x, edge_index, edge_weight, num, y, idx_train, idx_test, device):
    '''
    num: the number of samples used to 
    '''
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    num_class = int(y.max()+1)

    predictions = []
    for _ in range(num):
        rs_edge_index, rs_edge_weight = sample_noise(args, edge_index, edge_weight, idx_test, device)
        embeddings = model(x, rs_edge_index, rs_edge_weight)

        X = embeddings.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        Y = Y.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
        Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

        X = normalize(X, norm='l2')

        # X_train, X_test, y_train, y_test = train_test_split(X, Y,
        #                                                     test_size=1 - ratio)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 10)

        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                        param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                        verbose=0)
        clf.fit(X[idx_train], Y[idx_train])

        y_pred = clf.predict_proba(X[idx_test])
        y_pred = prob_to_one_hot(y_pred)
        y_test = Y[idx_test]
        prediction = y_pred.argmax(1)
        predictions.append(prediction)
        # print(prediction)
    prediction_distribution = np.array([np.bincount(prediction_list, minlength=num_class) for prediction_list in zip(*predictions)])
    final_prediction = prediction_distribution.argmax(1)
    acc = (((final_prediction==y_test.argmax(1)).sum())/len(final_prediction))
    return acc, final_prediction

import eval
from scipy.special import comb
from certify_K import certify_K
class Smooth_Ber(object):
    ABSTAIN = -1
    def __init__(self, args, base_encoder: torch.nn.Module, num_class, prob, x, if_node_level,device):
        self.args = args
        self.base_encoder = base_encoder
        self.num_class = num_class
        self.prob = prob
        self.x = x
        # self.edge_index = edge_index
        # self.edge_weight = edge_weight
        # self.y = y
        self.device = device
        self.if_node_level = if_node_level
        self.BASE = 100
        self.num_nodes = self.x.shape[0]
    
    def _sample_noise_ber(self, num, edge_index, edge_weight, y, if_node_level = True, idx_test = None, idx_train = None):
        with torch.no_grad():
            predictions = []
            for _ in range(num):
                if(if_node_level):
                    # if(self.args.sample_way == 'keep_structure'): # or ignore_structure or random_drop
                    #     rs_edge_index, rs_edge_weight = edge_index, edge_weight
                    # elif(self.args.sample_way == 'random_drop'):
                    #     # rs_edge_index, rs_edge_weight = sample_noise_all(self.args, edge_index, edge_weight, self.device)
                    #     rs_edge_index, rs_edge_weight = sample_noise_all_sparse(self.args, edge_index, edge_weight, self.x, self.device)
                    # z = self.base_encoder(self.x, rs_edge_index, rs_edge_weight)

                    adj = pyg_utils.to_dense_adj(edge_index)[0]
                    adj_noise = adj.clone().detach()
                    m = Bernoulli(torch.tensor([self.args.prob]).to(self.device))
                    
                    mask = m.sample(adj.shape).squeeze(-1).int()    
                    rand_inputs = torch.randint_like(adj, low=0, high=2).squeeze().int().to(self.device)
                    
                    adj_noise = adj * mask + rand_inputs * (1 - mask)
                    for idx in range(adj.shape[0]):
                        adj_noise[idx, idx] = adj[idx, idx]
                        adj_noise[:,idx] = adj_noise[idx]
                    z = self.base_encoder(self.x, adj_noise)

                    # if(self.args.sample_way == 'keep_structure'): # or ignore_structure or random_drop
                    #     rs_edge_index, rs_edge_weight = edge_index, edge_weight
                    # elif(self.args.sample_way == 'random_drop'):
                    #     # rs_edge_index, rs_edge_weight = sample_noise_all(self.args, edge_index, edge_weight, self.device)
                    #     rs_edge_index, rs_edge_weight = sample_noise_all_sparse(self.args, edge_index, edge_weight, self.x, self.device)
                    # z = self.base_encoder(self.x, rs_edge_index, rs_edge_weight)

                    if(self.args.dataset == 'ogbn-arxiv'):
                        _, prediction = eval.linear_evaluation_log(z, y, idx_train, idx_test)
                    else:
                        _, prediction = eval.linear_evaluation(z, y, idx_train, idx_test)
                    predictions.append(prediction)
            prediction_distribution = np.array([np.bincount(prediction_list, minlength=self.num_class) for prediction_list in zip(*predictions)])
            final_prediction = prediction_distribution.argmax(1)
            acc = (final_prediction==(y[idx_test].cpu().numpy())).sum()/len(final_prediction)
            return prediction_distribution, acc

    def certify_Ber(self,edge_index, edge_weight, y, n0, n, alpha, idx_train, idx_test):
        self.base_encoder.eval()
        # draw samples of h(x+ epsilon)
        counts_selection, acc_selection = self._sample_noise_ber(n0,edge_index, edge_weight, y, if_node_level = self.if_node_level, idx_test = idx_test, idx_train = idx_train)
        # use these samples to take a guess at the top class of the idx_test samples
        cAHats = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation, acc_estimation = self._sample_noise_ber(n,edge_index, edge_weight, y, if_node_level = self.if_node_level, idx_test = idx_test, idx_train = idx_train)
        
        classes = []
        probs = []
        for i, count_est in enumerate(counts_estimation):
            cAHat = cAHats[i]
            # use these samples to estimate a lower bound on pA
            nA = count_est[cAHat].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
        # print('ppf:', norm.ppf(pABar))
            if(cAHat != y[idx_test[i]]):
                classes.append(Smooth_Ber.ABSTAIN)
                probs.append(0.0)
            else:
                classes.append(cAHat)
                probs.append(pABar)
            # if pABar < 0.5:
            #     classes.append(Smooth_Ber.ABSTAIN)
            #     probs.append(0.0)
            # else:
            #     classes.append(cAHat)
            #     probs.append(pABar)
        return classes, probs

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float):
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    def calculate_certify_K(self, edge_index, edge_weight, y,n0, n, alpha, idx_train, idx_test, num_nodes, fn):
        v_range = [0,21]
        Ks = []
        classes, probs = self.certify_Ber(edge_index, edge_weight, y, n0=n0,n=n,alpha=alpha,idx_train=idx_train, idx_test = idx_test)
        for idx in range(len(probs)):
            if(probs[idx] > 0):
                v = certify_K(probs[idx], frac_alpha= 0.2, global_d=num_nodes,v_range=v_range, fn=fn)
                print('Idx_test',idx_test[idx].item(),'pAHat:', probs[idx], 'Certified K:', v)
            else:
                v = -1
                print('Idx_test',idx_test[idx].item(),'pAHat:', probs[idx], 'Certified K:', v)
            Ks.append(v)
        return probs, Ks, classes
    
    def _sample_noise_ber_approx(self, num, edge_index, edge_weight, y, if_node_level = True, idx_test = None, idx_train = None):
        with torch.no_grad():
            predictions = []
            total_num_remain_edges = []
            for _ in range(num):
                if(if_node_level):
                    '''remain edges are type List'''
                    rs_edge_index, rs_edge_weight, num_remain_edges = sample_noise_mask(self.args, edge_index, edge_weight, idx_test, self.device)
                    total_num_remain_edges.append(num_remain_edges)
                    # rs_edge_index, rs_edge_weight = sample_noise_sparse(self.args, self.x, edge_index, edge_weight, idx_test, self.device)
                    z = self.base_encoder(self.x, rs_edge_index, rs_edge_weight)
                    if(self.args.dataset == 'ogbn-arxiv'):
                        _, prediction = eval.linear_evaluation_log(z, y, idx_train, idx_test)
                    else:
                        _, prediction = eval.linear_evaluation(z, y, idx_train, idx_test)
                    predictions.append(prediction)
                else:
                    rs_edge_index, rs_edge_weight = sample_noise_all(self.args, edge_index, edge_weight, self.device)
                    # TODO
            total_num_remain_edges = np.array(total_num_remain_edges)
            prediction_distribution = np.array([np.bincount(prediction_list, minlength=self.num_class) for prediction_list in zip(*predictions)])
            final_prediction = prediction_distribution.argmax(1)
            acc = (final_prediction==(y[idx_test].cpu().numpy())).sum()/len(final_prediction)
            return prediction_distribution, acc, total_num_remain_edges
        
    def certify_Ber_approx(self,edge_index, edge_weight, y, n0, n, alpha, idx_train, idx_test):
        self.base_encoder.eval()
        # draw samples of h(x+ epsilon)
        counts_selection, acc_selection = self._sample_noise_ber(n0,edge_index, edge_weight, y, if_node_level = self.if_node_level, idx_test = idx_test, idx_train = idx_train)
        # use these samples to take a guess at the top class of the idx_test samples
        cAHats = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation, acc_estimation, total_num_remain_edges  = self._sample_noise_ber_approx(n,edge_index, edge_weight, y, if_node_level = self.if_node_level, idx_test = idx_test, idx_train = idx_train)
        # the lower bound probability of the larg
        classes = []
        probs = []
        for i, count_est in enumerate(counts_estimation):
            cAHat = cAHats[i]
            # use these samples to estimate a lower bound on pA
            nA = count_est[cAHat].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
        # print('ppf:', norm.ppf(pABar))
            if(cAHat != y[idx_test[i]]):
                classes.append(Smooth_Ber.ABSTAIN)
                probs.append(0.0)
            else:
                classes.append(cAHat)
                probs.append(pABar)
            # if pABar < 0.5:
            #     classes.append(Smooth_Ber.ABSTAIN)
            #     probs.append(0.0)
            # else:
            #     classes.append(cAHat)
            #     probs.append(pABar)
        return classes, probs, total_num_remain_edges
    
    def calculate_certify_K_approx(self, edge_index, edge_weight, y,n0, n, alpha, idx_train, idx_test, num_nodes, fn, k_range):
        # v_range = [0,21]
        k_range = list(range(0,21))
        Ks = []
        total_Deltas = []
        classes, probs, total_num_remain_edges = self.certify_Ber_approx(edge_index, edge_weight, y, n0=n0,n=n,alpha=alpha,idx_train=idx_train, idx_test = idx_test)
        # statistic the remaining edges of the clean nodes: total_num_remain_edges
        # stattistic the degree of the clean nodes
        degrees = pyg_utils.degree(edge_index[0]).cpu().numpy()
        # calculate the Delta for each node by using Monte Carlo Sampling
        
        # for i in range(n): 
        #     '''num_remain_edges'''
        #     num_remain_edges = total_num_remain_edges[i]
        #     for k in v_range:
        #         # calculate the degrees of perturbed version
        #         degrees_perturb = degrees + k
        #         Deltas = 1 - sp_special.comb(degrees,num_remain_edges)/(sp_special.comb(degrees_perturb,num_remain_edges) * (1-self.args.prob)**k)
        #         total_Deltas.append(Deltas)
        for k in k_range:
        # for k in range(v_range[0],v_range[1]):
            print("K={}".format(k))
            Deltas_list = []
            for i in range(n): 
                num_remain_edges = (total_num_remain_edges[i])
                degrees_perturb = (degrees + k)
                Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges)/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges)) * (1-self.args.prob)**(k) 
                # Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges)/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges)) * (1-self.args.prob)**(degrees[idx_test.cpu()] - num_remain_edges + k)  * (self.args.prob)**(num_remain_edges) 
                
                
                # Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges)/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges)) * (self.args.prob**(2*num_remain_edges) * (1-self.args.prob)**(2*degrees[idx_test.cpu()] - 2*num_remain_edges+k))
                Deltas_list.append(Deltas)
            mean_Deltas = np.mean(Deltas_list,axis=0)
            # print(mean_Deltas)
            total_Deltas.append(mean_Deltas)
        return total_Deltas, probs, classes
       
       
    # def certify_K(self, K: int):
    #     pr = 0
    #     p = 0
    #     sorted_ratio = self.sort_ratio(K)

    #     for ele in sorted_ratio:
    #         u = ele[1] # 1
    #         v = ele[2] # 9
    #         p_orig, p_pertub = self.cal_prob(u, v)
            
    #         p_orig = p_orig * self.cal_L(K, u, v)
    #         p_pertub = p_pertub * self.cal_L(K, u, v)

    #         if pr + p_pertub < self.BASE/2 * np.power(self.BASE, self.num_nodes-1):
    #             pr += p_orig
    #             p += p_orig
    #         else:
    #             p += p_orig * (self.BASE/2 * np.power(self.BASE, self.num_nodes-1) - pr) /  p_pertub
    #             return float(p) / np.power(self.BASE, self.num_nodes)
            
    # def sort_ratio(self, K: int):
        
    #     ratio_list = list()
    #     for u in range(K+1):
    #         for v in list(reversed(range(u, K+1))):
    #             if u + v >= K and np.mod(u + v - K, 2) == 0:
    #                 ratio_list.append((v-u,u,v))
    #     sorted_ratio = sorted(ratio_list, key=lambda tup: tup[0], reverse=True)
    #     return sorted_ratio
    
    # def cal_prob(self, u: int, v: int):
    #     p_orig = np.power(int(self.prob * self.BASE), self.num_nodes-u) * np.power(int((1-self.prob) * self.BASE), u)
    #     p_pertub = np.power(int(self.prob * self.BASE), self.num_nodes-v) * np.power(int((1-self.prob) * self.BASE), v)
    #     return p_orig, p_pertub

    # def cal_L(self, K: int, u: int, v: int):
        
    #     i = int((u + v - K) / 2)
    #     return comb(self.num_nodes-K, i) * comb(K, u-i)
