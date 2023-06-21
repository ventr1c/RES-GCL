import torch
import torch.optim as optim


from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, OneHotEncoder

import numpy as np

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def test(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

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
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc, prediction

class LogisticReg(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nproj, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,tau=None, layer=2,if_smoothed=True,device=None):
        super(LogisticReg, self).__init__()
        self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.lr = lr
        self.weight_decay = 0
        self.device = device

    def fit(self, features, labels, train_iters=200,seen_node_idx=None,verbose=True):
        pass

    def test(self,x, y, idx_train, idx_test):
        return test(x, y, idx_train, idx_test)
    
    def forward(self, x, edge_index, edge_weight=None):
        return x