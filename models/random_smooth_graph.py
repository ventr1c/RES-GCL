import torch 
from torch.distributions.bernoulli import Bernoulli
import utils
import numpy as np
from torch_geometric.utils import to_undirected, to_dense_adj,to_torch_coo_tensor,dense_to_sparse
import scipy.special as sp_special
import torch_geometric.utils as pyg_utils

import eval
from scipy.special import comb
from certify_K import certify_K
from statsmodels.stats.proportion import proportion_confint

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


def sample_noise_all_mask(args,edge_index, edge_weight, batch, device):
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
    # e = noisy_edge_weight.count_nonzero().item()
    num_remain_edges = torch.zeros([batch.unique().shape[0]])
    for i in range(noisy_edge_weight.shape[0]):
        if(noisy_edge_weight[i]>0):
            idx = edge_index[0,i]
            num_remain_edges[batch[idx]]+=1
    
    if(noisy_edge_weight!=None):
        noisy_edge_index = noisy_edge_index[:,noisy_edge_weight.nonzero().flatten().long()]
        # batch = batch[noisy_edge_index[0].unique()]
        noisy_edge_weight = torch.ones([noisy_edge_index.shape[1],]).to(device)
    return noisy_edge_index, noisy_edge_weight, batch, num_remain_edges

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
    
    def _sample_noise_ber(self, num, dataloader, idx_test = None, idx_train = None):
        with torch.no_grad():
            predictions = []
            for _ in range(num):
                x = []
                y = []
                for data in dataloader:
                    data = data.to(self.device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.zeros((num_nodes, 1), dtype=torch.float32, device=self.device)
                    rs_edge_index, rs_edge_weight, rs_batch = sample_noise_all_graph(self.args, data.edge_index, data.edge_weight, data.batch, self.device)
                   
                    # _, g, _, _, _, _ = self.base_encoder.forward(data.x, rs_edge_index, batch=rs_batch)
                    g = self.base_encoder.forward(data.x, rs_edge_index, batch=rs_batch)
                    x.append(g)
                    y.append(data.y)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                _, prediction = eval.linear_evaluation(x, y, idx_train, idx_test)
                predictions.append(prediction)
            # print(predictions)
            prediction_distribution = np.array([np.bincount(prediction_list, minlength=self.num_class+1) for prediction_list in zip(*predictions)])
            # print(prediction_distribution)
            final_prediction = prediction_distribution.argmax(1)
            acc = (final_prediction==(y[idx_test].cpu().numpy())).sum()/len(final_prediction)
            return prediction_distribution, acc
        
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float):
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    def _sample_noise_ber_approx(self, num, dataloader, idx_test = None, idx_train = None):
        with torch.no_grad():
            total_num_remain_edges = []
            predictions = []
            num_remain_edges = []
            for _ in range(num):
                x = []
                y = []
                num_remain_edges = torch.tensor([])
                for data in dataloader:
                    data = data.to(self.device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.zeros((num_nodes, 1), dtype=torch.float32, device=self.device)
                    # rs_edge_index, rs_edge_weight = sample_noise_all(self.args, data.edge_index, data.edge_weight, self.device)
                    rs_edge_index, rs_edge_weight, rs_batch, batch_num_remain_edges = sample_noise_all_mask(self.args, data.edge_index, data.edge_weight, data.batch, self.device)
                    
                    num_remain_edges = torch.cat((num_remain_edges,batch_num_remain_edges))
                    # _, g, _, _, _, _ = self.forward(data.x, rs_edge_index, rs_batch,)
                    g = self.base_encoder.forward(data.x, rs_edge_index, batch=rs_batch)
                    x.append(g)
                    y.append(data.y)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                num_remain_edges = np.array(num_remain_edges)
                total_num_remain_edges.append(num_remain_edges)
                _, prediction = eval.linear_evaluation(x, y, idx_train, idx_test)
                predictions.append(prediction)
                
            total_num_remain_edges = np.array(total_num_remain_edges)

            prediction_distribution = np.array([np.bincount(prediction_list, minlength=self.num_class) for prediction_list in zip(*predictions)])
            final_prediction = prediction_distribution.argmax(1)
            acc = (final_prediction==(y[idx_test].cpu().numpy())).sum()/len(final_prediction)
            return prediction_distribution, acc, total_num_remain_edges, y 
        
    def certify_Ber_approx(self,dataloader, n0, n, alpha, idx_train, idx_test):
        self.base_encoder.eval()
        # draw samples of h(x+ epsilon)
        counts_selection, acc_selection = self._sample_noise_ber(n0,dataloader, idx_test = idx_test, idx_train = idx_train)
        # print(counts_selection)
        # use these samples to take a guess at the top class of the idx_test samples
        cAHats = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation, acc_estimation, total_num_remain_edges, y  = self._sample_noise_ber_approx(n,dataloader, idx_test = idx_test, idx_train = idx_train)
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
    
    def calculate_certify_K_approx(self, dataset, dataloader,n0, n, alpha, idx_train, idx_test, k_range):
        # v_range = [0,21]
        k_range = list(range(0,21))
        Ks = []
        total_Deltas = []
        # total_num_remain_edges -> shape: [n, test_graph_num]
        classes, probs, total_num_remain_edges = self.certify_Ber_approx(dataloader, n0=n0,n=n,alpha=alpha,idx_train=idx_train, idx_test = idx_test)
        # statistic the remaining edges of the clean nodes: total_num_remain_edges
        # stattistic the degree of the clean nodes
        degrees = []
        for data in dataset:
            degree_single = data.edge_index.shape[1]
            degrees.append(degree_single)
        degrees = torch.tensor(degrees)
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
                Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges[idx_test.cpu()])/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges[idx_test.cpu()])) * (1-self.args.prob)**(k) 
                # Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges)/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges)) * (1-self.args.prob)**(degrees[idx_test.cpu()] - num_remain_edges + k)  * (self.args.prob)**(num_remain_edges) 
                
                
                # Deltas = 1 - sp_special.comb(degrees[idx_test.cpu()],num_remain_edges)/(sp_special.comb(degrees_perturb[idx_test.cpu()],num_remain_edges)) * (self.args.prob**(2*num_remain_edges) * (1-self.args.prob)**(2*degrees[idx_test.cpu()] - 2*num_remain_edges+k))
                Deltas_list.append(Deltas)
            mean_Deltas = np.mean(Deltas_list,axis=0)
            # print(mean_Deltas)
            total_Deltas.append(mean_Deltas)
        return total_Deltas, probs, classes
       
       