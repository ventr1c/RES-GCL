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
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
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
                    rs_edge_index, rs_edge_weight = sample_noise(self.args, edge_index, edge_weight, idx_test, self.device)
                    z = self.base_encoder(self.x, rs_edge_index, rs_edge_weight)
                    _, prediction = eval.linear_evaluation(z, y, idx_train, idx_test)
                    predictions.append(prediction)
                else:
                    rs_edge_index, rs_edge_weight = sample_noise_all(self.args, edge_index, edge_weight, self.device)
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
    
    def certify_K(self, K: int):
        pr = 0
        p = 0
        sorted_ratio = self.sort_ratio(K)

        for ele in sorted_ratio:
            u = ele[1] # 1
            v = ele[2] # 9
            p_orig, p_pertub = self.cal_prob(u, v)
            
            p_orig = p_orig * self.cal_L(K, u, v)
            p_pertub = p_pertub * self.cal_L(K, u, v)

            if pr + p_pertub < self.BASE/2 * np.power(self.BASE, self.num_nodes-1):
                pr += p_orig
                p += p_orig
            else:
                p += p_orig * (self.BASE/2 * np.power(self.BASE, self.num_nodes-1) - pr) /  p_pertub
                return float(p) / np.power(self.BASE, self.num_nodes)
            
    def sort_ratio(self, K: int):
        
        ratio_list = list()
        for u in range(K+1):
            for v in list(reversed(range(u, K+1))):
                if u + v >= K and np.mod(u + v - K, 2) == 0:
                    ratio_list.append((v-u,u,v))
        sorted_ratio = sorted(ratio_list, key=lambda tup: tup[0], reverse=True)
        return sorted_ratio
    
    def cal_prob(self, u: int, v: int):
        p_orig = np.power(int(self.prob * self.BASE), self.num_nodes-u) * np.power(int((1-self.prob) * self.BASE), u)
        p_pertub = np.power(int(self.prob * self.BASE), self.num_nodes-v) * np.power(int((1-self.prob) * self.BASE), v)
        return p_orig, p_pertub

    def cal_L(self, K: int, u: int, v: int):
        
        i = int((u + v - K) / 2)
        return comb(self.num_nodes-K, i) * comb(K, u-i)
