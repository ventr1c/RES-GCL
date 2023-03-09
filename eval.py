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
def label_classification(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

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

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    # print(y_pred.argmax(1).shape,y_test.argmax(1))
    # print(((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }

def label_classification_origin(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    print('F1Mi',micro,'F1Ma',macro)
    return {
        'F1Mi': micro,
        'F1Ma': macro
    }

def label_evaluation(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

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

    # micro = f1_score(y_test, y_pred, average="micro")
    # macro = f1_score(y_test, y_pred, average="macro")

    # print(y_pred.argmax(1).shape,y_test.argmax(1))
    # return {
    #     'F1Mi': micro,
    #     'F1Ma': macro
    # }
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def linear_evaluation(embeddings, y, idx_train, idx_test):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

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

    # micro = f1_score(y_test, y_pred, average="micro")
    # macro = f1_score(y_test, y_pred, average="macro")

    # print(y_pred.argmax(1).shape,y_test.argmax(1))
    # return {
    #     'F1Mi': micro,
    #     'F1Ma': macro
    # }
    # prediction results of idx_test
    prediction = y_pred.argmax(1)
    acc = (((y_pred.argmax(1)==y_test.argmax(1)).sum())/len(y_pred.argmax(1)))
    return acc, prediction

import models.random_smooth as random_smooth
def smoothed_linear_evaluation(args, model, x, edge_index, edge_weight, num, y, idx_train, idx_test, device):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    num_class = int(y.max()+1)

    predictions = []
    for _ in range(num):
        rs_edge_index, rs_edge_weight = random_smooth.sample_noise(args, edge_index, edge_weight, idx_test, device)
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

def smoothed_linear_evaluation_all(args, model, x, edge_index, edge_weight, num, y, idx_train, idx_test, device):
    idx_train = idx_train.cpu().numpy()
    idx_test = idx_test.cpu().numpy()

    num_class = int(y.max()+1)

    predictions = []
    for _ in range(num):
        rs_edge_index, rs_edge_weight = random_smooth.sample_noise_all(args, edge_index, edge_weight, device)
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