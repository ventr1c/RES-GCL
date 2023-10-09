import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_evaluation(embeddings, y, idx_train, idx_test):
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

    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc

def linear_evaluation(embeddings, y, idx_train, idx_test):
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

    # prediction results of idx_test
    prediction = y_pred.argmax(1)
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc, prediction

def linear_evaluation_log(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()

    X = normalize(X, norm='l2')


    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X[idx_train], Y[idx_train])

    y_pred = logreg.predict_proba(X[idx_test])
    y_pred = prob_to_one_hot(y_pred)
    y_test = Y[idx_test]

    prediction = y_pred.argmax(1)
    acc = (((y_pred.argmax(1)==y_test).sum())/len(y_pred.argmax(1)))
    return acc, prediction


def lr_evaluation(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    clf = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)


    clf.fit(X[idx_train], Y[idx_train])

    y_pred = clf.predict_proba(X[idx_test])
    y_pred = prob_to_one_hot(y_pred)
    y_test = Y[idx_test]

    acc = (((y_pred.argmax(1)==y_test).sum())/len(y_pred.argmax(1)))
    return acc