import numpy as np
from logging import getLogger

from src.method import oinmf, bnmf


log = getLogger(__name__)


class Trainer:
    """トレーナークラス．

    与えられたNMFモデルを推定する．

    Parameters
    ----------
    model : src.model.Model
        推定したいNMFモデル．
    conf : omegaconf
        設定オブジェクト．
    """
    def __init__(self):
        self.model = None
        self.conf  = None

    def set_model(self, model):
        """NMFモデルをセットする．

        Parameters
        ----------
        model : src.model.Model
            NMFモデルのオブジェクト．
        """
        self.model = model
        log.info("Set an input spectrogram and onset matrix.")
    
    def set_config(self, conf):
        """推定に利用する設定をセットする．

        Parameters
        ----------
        conf : omegaconf
            設定オブジェクト．
        """
        self.conf = conf
        log.info("Set OI-NMF model parameters.")

    def train(self, train_method):
        """モデル推定を行う．

        Parameters
        ----------
        train_method : {'oinmf', 'bnmf'}
            推定に利用する手法．
            ``'oinmf'``: OI-NMF.
             ``'bnmf'``: Bayesian NMF.
        """
        log.info("Start model inference.")
        log.info("Inference method: %s" % train_method)
        if train_method == "oinmf":
            EW, EH, ES = oinmf.gibbs_sampler(self.model.X_mag, self.model.onset_matrix, **self.conf.model_params, without_onset=self.conf.transform_onset.without_onset)
            self.model.W, self.model.H, self.model.S = EW, EH, ES
        if train_method == "bnmf":
            EW, EH     = bnmf.gibbs_sampler(self.model.X_mag, self.conf.model_params.K, self.conf.model_params.W_priors, self.conf.model_params.H_priors, self.conf.model_params.n_iter, self.conf.model_params.burnin)
            self.model.W, self.model.H = EW, EH
        log.info("Finish model inference.")
