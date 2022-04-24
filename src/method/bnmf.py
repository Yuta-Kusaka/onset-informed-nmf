import cupy as cp
import numpy as np
from tqdm import tqdm
from logging import getLogger


log = getLogger(__name__)


EPS = 1.0e-12


def _init_WH(W_priors, H_priors, dims):
    F, T, K = dims
    a, b = W_priors
    c, d = H_priors
    W = cp.random.gamma(a, 1.0 / b, (F, K))
    H = cp.random.gamma(c, 1.0 / d, (K, T))
    return W, H


def _sample_W(X, W, H, W_priors):
    a, b = W_priors
    X_hat = W @ H + EPS
    a_hat = a + (W * ((X / X_hat) @ H.T))
    b_hat = b + cp.sum(H ,axis=1, keepdims=True).T
    return cp.random.gamma(a_hat, 1.0 / b_hat)


def _sample_H(X, W, H, H_priors):
    c, d = H_priors
    X_hat = W @ H + EPS
    c_hat = c + (H * (W.T @ (X / X_hat)))
    d_hat = d + cp.sum(W, axis=0, keepdims=True).T
    return cp.random.gamma(c_hat, 1.0 / d_hat)


def update_expectation(previous_E, current_value, n):
    return (n * previous_E + current_value) / (n + 1)


def gibbs_sampler(X, K, W_priors, H_priors, n_iter, burnin):
    X = cp.asarray(X)
    F, T = X.shape
    W, H = _init_WH(W_priors, H_priors, (F, T, K))
    for i in tqdm(range(burnin), postfix="burnin"):
        W = _sample_W(X, W, H, W_priors)
        H = _sample_H(X, W, H, H_priors)
    EW = cp.copy(W)
    EH = cp.copy(H)
    for i in tqdm(range(n_iter - burnin), postfix="sampling"):
        W = _sample_W(X, W, H, W_priors)
        H = _sample_H(X, W, H, H_priors)
        EW = update_expectation(EW, W, i)
        EH = update_expectation(EH, H, i)
    EW = cp.asnumpy(EW)
    EH = cp.asnumpy(EH)
    return EW, EH
