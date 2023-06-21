# %%
from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.GCN_Encoder import GCN_Encoder
from models._GNNGuard import GNNGuard
from models.Grace import Grace
# from models._GRACE import GRACE
from models.RobustGCN import RobustGCN
# import models.GraphCL as GraphCL
from models.GraphCL import GraphCL
from models.BGRL import BGRL
from models.DGI_transductive import DGI
from models.GAE import GAE
from models.LR import LogisticReg
from models.Node2vec import Node2vec
from models.Grace_Jaccard import Grace_Jaccard

from models.Grace_auto import Grace_auto
from models.BGRL_G2L import BGRL_G2L
from models.Grace_flip import Grace_flip
# from models._Ariel import Ariel
from models._ARIEL.Ariel_1 import Ariel
from models.Grace_batch import Grace_batch

# from GNNGuard.GCN import GuardGCN
def model_construct(args,model_name,data,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    elif(args.dataset == 'ogbn-arxiv'):
        use_ln = True
        layer_norm_first = True
    else:
        use_ln = False
        layer_norm_first = False
    if(model_name == 'Grace'):
        model = Grace(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device,\
                      use_ln=use_ln,\
                      layer_norm_first=layer_norm_first)
    elif(model_name == 'Grace_flip'):
        model = Grace_flip(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device,\
                      use_ln=use_ln,\
                      layer_norm_first=layer_norm_first)
    elif(model_name == 'Grace_batch'):
        model = Grace_flip(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device,\
                      use_ln=use_ln,\
                      layer_norm_first=layer_norm_first)
    elif(model_name == 'Grace_auto'):
        model = Grace_auto(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device,\
                      use_ln=False,\
                      layer_norm_first=False)
        # model = GRACE(args, 
        #         data.x.shape[1], 
        #         nhid=args.num_hidden, 
        #         nproj=args.num_proj_hidden, 
        #         nclass=int(data.y.max()+1), 
        #         dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device)
    elif(model_name == 'BGRL'):
        model = BGRL(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device)
    elif(model_name == 'DGI'):
        model = DGI(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device,
                use_ln=use_ln,
                layer_norm_first=layer_norm_first)
    elif(model_name == 'GraphCL'):
        model = GraphCL.Encoder(
                      args = args, 
                      encoder=gconv, 
                      augmentor=(aug1, aug2), 
                      input_dim=input_dim, 
                      hidden_dim=args.num_hidden, 
                      lr=args.cl_lr, 
                      tau=args.tau,
                      num_epoch = args.cl_num_epochs, 
                      if_smoothed = args.if_smoothed,
                      device = device)
    elif(model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
        
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.train_lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.train_lr,\
                weight_decay=args.weight_decay,\
                device=device)
    elif(model_name == 'GNNGuard'):
        model = GNNGuard(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    use_ln=use_ln,\
                    device=device)
    elif(model_name == 'RobustGCN'):
        model = RobustGCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    elif(model_name == 'GAE'):
        model = GAE(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device)
    elif(model_name == 'LR'):
        model = LogisticReg(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device)
    elif(model_name == 'Node2vec'):
        model = Node2vec(args, 
                data.x.shape[1], 
                nhid=args.num_hidden, 
                nproj=args.num_proj_hidden, 
                nclass=int(data.y.max()+1), 
                dropout=args.dropout, lr=args.cl_lr, weight_decay=args.cl_weight_decay,tau=args.tau, layer=2,if_smoothed=args.if_smoothed,device=device)
    elif(model_name == 'Grace-Jaccard'):
        model = Grace_Jaccard(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      threshold = args.prune_thrh,\
                      device=device,\
                      use_ln=False,\
                      layer_norm_first=False)
    elif(model_name == 'Ariel'):
        model = Ariel(args = args, \
                      nfeat = data.x.shape[1], \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      nclass = int(data.y.max()+1),\
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device,\
                      use_ln=False,\
                      layer_norm_first=False)
    else:
        print("Not implement {}".format(model_name))
    return model.to(device)

def encoder_construct(args,model_name,data,base_encoder,aug1,aug2,input_dim,device):
    if(model_name == 'GraphCL'):
        model = GraphCL.Encoder(
                      args = args, 
                      encoder=base_encoder, 
                      augmentor=(aug1, aug2), 
                      input_dim=input_dim, 
                      hidden_dim=args.num_hidden, 
                      lr=args.cl_lr, 
                      tau=args.tau,
                      num_epoch = args.cl_num_epochs, 
                      if_smoothed = args.if_smoothed,
                      device = device)
    return model

def model_construct_global(args,model_name,dataset,device):
    if(model_name == 'BGRL-G2L'):
        model = BGRL_G2L(args = args, \
                      nfeat = dataset.num_features, \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device)
    elif(model_name == 'GraphCL'):
        model = GraphCL(args = args, \
                      nfeat = dataset.num_features, \
                      nhid = args.num_hidden, \
                      nproj = args.num_proj_hidden, \
                      dropout=args.dropout, \
                      lr=args.cl_lr, \
                      weight_decay=args.cl_weight_decay, \
                      tau=args.tau, \
                      layer=2,\
                      if_smoothed=args.if_smoothed,\
                      device=device)
    return model.to(device)
def model_construct_old(args,model_name,data,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    if (model_name == 'GCN'):

        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
        
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.train_lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.train_lr,\
                weight_decay=args.weight_decay,\
                device=device)
    elif(model_name == 'GCN_Encoder'):
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device,
                            use_ln=use_ln,
                            layer_norm_first=layer_norm_first)
    elif(model_name == 'guard'):
        model = GNNGuard(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    use_ln=use_ln,\
                    device=device)
    else:
        print("Not implement {}".format(model_name))
    return model.to(device)
