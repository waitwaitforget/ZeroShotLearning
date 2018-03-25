import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from scipy.linalg import solve_sylvester


def SAE(X, S, lamd):
    A = np.dot(S.T, S)
    B = lamd * np.dot(X.T, X)
    C = (1 + lamd) * np.dot(S.T, X)

    try:
        W = solve_sylvester(A, B, C)
    except:
        print('Sovler failed')
        return None
    return W


def NormalizeFeat(fea, dim):
    # feat = normalize(fea, norm='l2')
    norm = np.linalg.norm(fea, axis=0)
    norm[norm == 0] = 1e-5
    feat = fea / norm[np.newaxis, :]
    return feat


def train(X, S):
    X = NormalizeFeat(X, 1)
    lamd = 500000
    W = SAE(X, S, lamd)
    return W


def evaluate(X_test, S_test, W, **kwargs):
    S_est = np.dot(X_test, NormalizeFeat(W, 1).transpose())

    zsl_acc, Y_hit5 = accuracy(S_est, S_test, **kwargs)
    return zsl_acc, Y_hit5


def accuracy(pred, groundtruth, **kwargs):
    hitk = kwargs['HITK']
    dist = 1 - cosine_distances(pred, NormalizeFeat(groundtruth, 1))

    Y_hit5 = np.zeros((dist.shape[0], hitk), dtype=np.int32)
    for i in range(dist.shape[0]):
        idx = dist[i].argsort()[::-1][:hitk]
        # print(kwargs['testclasses_id'][idx], kwargs['test_labels'][i])
        Y_hit5[i] = kwargs['testclasses_id'][idx]

    n = 0
    for i in range(dist.shape[0]):
        if kwargs['test_labels'][i] in Y_hit5[i]:
            n += 1
    zsl_acc = n * 1.0 / dist.shape[0]
    return zsl_acc, Y_hit5


def awa_demo():
    import scipy.io as sio
    data = sio.loadmat('./data_zsl/awa_demo_data.mat')

    X_tr = data['X_tr']
    S_tr = data['S_tr']
    print('training')
    X_tr = NormalizeFeat(X_tr, 1)
    # if os.path.exists('W.npy'):
    #    W = np.load('W.npy')
    # else:
    W = train(X_tr, S_tr)  # W is k * d
    # W = sio.loadmat('W.mat')
    # W = W['W']
    # np.save('W.npy', W)
    # print(W.shape)
    X_te = data['X_te']
    S_te_gt = data['S_te_gt']
    param = data['param']

    print('evaluate')
    acc, Y_hit5 = evaluate(X_te, S_te_gt, W, HITK=1, testclasses_id=param[0][0][1], test_labels=param[0][0][2])

    print('AWA dataset V >> S Acc :', acc)


if __name__ == '__main__':
    awa_demo()
