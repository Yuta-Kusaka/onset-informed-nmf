import cupy as cp
import numpy as np
from tqdm import tqdm
from logging import getLogger


log = getLogger(__name__)


EPS = 1.0e-12


def _init_W(W_priors, W_shape):
    """ Wを初期化．

    Notes
    -----
    すべての基底をガンマ事前分布に基づきランダムに初期化．
    """
    a, b = W_priors
    return cp.random.gamma(a, 1.0 / b, W_shape, dtype=np.float32)


def _init_H(H_priors, H_shape, onset_mat, without_onset):
    """ Hを初期化．

    Notes
    -----
    オンセットが与えられる基底は，オンセット区間で事前分布の期待値c/dで初期化．
    それ以外の基底はガンマ事前分布に基づきランダムに初期化．
    """
    c, d = H_priors
    H = np.random.gamma(c, 1.0 / d, H_shape)
    if not without_onset:
        target_components_num = np.count_nonzero(np.any(onset_mat == 1, axis=1))
        for k in range(target_components_num):
            H[k, :] = c / d * onset_mat[k, :]
    return cp.asarray(H, dtype=np.float32)


def _init_S(onset_mat, without_onset):
    """ Sを初期化

    Notes
    -----
    オンセットが与えられる基底は，オンセット行列と同じように初期化．
    それ以外の基底はすべて1で初期化．
    """
    if without_onset:
        S_temp = np.random.rand(onset_mat.shape[0], onset_mat.shape[1]).astype(np.float32)
        S =  np.where(S_temp > 0.5, 1, 0)
    else:
        S = np.full_like(onset_mat, 1, dtype=np.float32)
        target_components_num = np.count_nonzero(np.any(onset_mat == 1, axis=1))
        for k in range(target_components_num):
            S[k, :] = onset_mat[k, :]
    return cp.asarray(S)


def _sample_W(X, W, H, S, W_priors):
    """ Wをサンプリング
    """
    a, b = W_priors
    X_hat = W @ (H * S) + EPS
    a_hat = a + W * cp.dot(X / X_hat, (H * S).T)
    b_hat = b + cp.sum(H * S, axis=1, keepdims=True).T
    return cp.random.gamma(a_hat, 1.0 / b_hat, dtype=np.float32)


def _sample_H(X, W, H, S, H_priors):
    """ Hをサンプリング
    """
    c, d = H_priors
    X_hat = W @ (H * S) + EPS
    c_hat = c + (H * S) * cp.dot(W.T, X / X_hat)
    d_hat = d + S * cp.sum(W, axis=0, keepdims=True).T
    return cp.random.gamma(c_hat, 1.0 / d_hat, dtype=np.float32)


def _sample_S_unit(X, W, H, S, S_priors, index, onset_mat):
    """ Sの要素をサンプリング
    """
    k, t = index
    p_init, p_1to1, p_0to1 = S_priors

    X_neg_k_t = W @ (H[:, t] * S[:, t]) - W[:, k] * (H[k, t] * S[k, t])
    WH = W[:, k] * H[k, t]
    logP1 = np.sum(X[:, t] * np.log(X_neg_k_t + WH) - WH)
    logP0 = np.sum(X[:, t] * np.log(X_neg_k_t + EPS))
    # t=0のときは初期確率を使う．
    if t == 0:
        logP1 += np.log(p_init)
        logP0 += np.log(1 - p_init)
    else:
        logP1 += S[k, t-1] * np.log(p_1to1) + (1 - S[k, t-1]) * np.log(p_0to1)
        logP0 += S[k, t-1] * np.log(1 - p_1to1) + (1 - S[k, t-1]) * np.log(1 - p_0to1)
    # 確率が[0, 1]になるように正規化し，ベルヌーイ分布からサンプリング．
    P_max = np.maximum(logP1, logP0)
    P1 = np.exp(logP1 - P_max)
    P0 = np.exp(logP0 - P_max)
    ratio = P1 / (P1 + P0)
    S_unit = np.random.rand() <= ratio
    S[k, t] = S_unit | onset_mat[k, t]


def _sample_S(X, W, H, S, onset_mat, S_priors):
    """ Sをサンプリング
    """
    X = cp.asnumpy(X)
    W = cp.asnumpy(W)
    H = cp.asnumpy(H)
    S = cp.asnumpy(S)
    K, T = H.shape
    for k in range(K):
        for t in range(T):
            _sample_S_unit(X, W, H, S, S_priors, (k, t), onset_mat)
    return cp.asarray(S)


def update_expect(current_value, previous_expactation, n):
    """ 期待値を更新する

    サンプリングした値を利用して各変数の期待値をオンラインに更新する．

    Parameters
    ----------
    current_value : ndarray
        更新に利用するサンプルしてきた行列の値．
    previous_expactation : ndarray
        更新する前の入力行列の期待値（1ステップ前の値）．
    n : int
        現在のステップ数．

    Returns
    -------
    ndarray
        更新後の期待値の行列．
    """
    return (n * previous_expactation + current_value) / (n + 1)


def gibbs_sampler(X, onset_mat, K, W_priors, H_priors, S_priors, n_iter, burnin, without_onset):
    """ ギブスサンプリングを行う
    
    Parameters
    ----------
    X : ndarray
        振幅スペクトログラムの行列．
    onset_mat : ndarray
        オンセット行列．
    K : int
        基底数．
    W_priors : (float, float)
        Wの事前分布のハイパーパラメータ．
    H_priors : (float, float)
        Hの事前分布のハイパーパラメータ．
    S_priors : (float, float, float)
        Sの事前分布のハイパーパラメータ．
    n_iter : int
        サンプリングのイテレーション数．
    burnin : int
        サンプリングのバーンイン．
    without_onset : bool
        オンセットを利用しない場合は ``true`` .
    
    Returns
    -------
    ndarray, ndarray, ndarray
        基底スペクトル，アクティベーション，バイナリマスクの期待値の行列．
    """
    # 初期化
    X = cp.asarray(X)
    F, T = X.shape
    W = _init_W(W_priors, (F, K))
    H = _init_H(H_priors, (K, T), onset_mat, without_onset)
    S = _init_S(onset_mat, without_onset)
    # バーンイン
    for i in tqdm(range(burnin), postfix="burnin"):
        W = _sample_W(X, W, H, S, W_priors)
        H = _sample_H(X, W, H, S, H_priors)
        S = _sample_S(X, W, H, S, onset_mat, S_priors)
    # 期待値計算に利用するサンプリング
    EW = cp.copy(W)
    EH = cp.copy(H)
    ES = cp.copy(S)
    for i in tqdm(range(n_iter - burnin), postfix="sampling"):
        W = _sample_W(X, W, H, S, W_priors)
        H = _sample_H(X, W, H, S, H_priors)
        S = _sample_S(X, W, H, S, onset_mat, S_priors)
        ES = update_expect(S, ES, i)
        EW = update_expect(W, EW, i)
        EH = update_expect(H, EH, i)
    EW = cp.asnumpy(EW)
    EH = cp.asnumpy(EH)
    ES = np.round(cp.asnumpy(ES))
    return EW, EH, ES
