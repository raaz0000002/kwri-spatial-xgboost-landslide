"""Fast DeLong test for correlated ROC AUCs (Sun & Xu, 2014)."""
import numpy as np
from scipy import stats


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_cov(y, preds):
    """preds: k x n array. Returns aucs (k,), covariance (k,k)."""
    y = np.asarray(y); order = np.argsort(-y, kind='mergesort')
    m = int(y.sum()); n = len(y) - m
    P = np.asarray(preds)[:, order]
    pos, neg = P[:, :m], P[:, m:]
    k = P.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(P[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(S)


def delong_pair(y, p1, p2):
    a, S = delong_cov(y, [p1, p2])
    d = a[0] - a[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(max(var, 1e-20))
    z = d / se
    p = 2 * stats.norm.sf(abs(z))
    return a[0], a[1], d, d - 1.96 * se, d + 1.96 * se, p


def auc_ci(y, p):
    a, S = delong_cov(y, [p])
    se = np.sqrt(S[0, 0])
    return a[0], a[0] - 1.96 * se, a[0] + 1.96 * se


def holm(pvals):
    p = np.asarray(pvals); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
