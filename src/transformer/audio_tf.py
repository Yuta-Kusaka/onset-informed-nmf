import numpy as np
from librosa import stft, istft
from librosa.effects import hpss
from logging import getLogger

from src.data.audio import Audio


log = getLogger(__name__)


class AudioTransformer:
    """ オーディオ変換クラス．

    :abbr:`STFT (short time Fourier transform)` と :abbr:`HPSS (harmonic/percussive sourse separation)`
    （オプション）を利用して，オーディオをスペクトログラムに変換する．

    .. [1] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.

    Parameters
    ----------
    stft_n_fft : int
        STFTのFFTサイズ．
    stft_hop_length : int
        STFTのホップサイズ．
    stft_window : stf
        STFTの窓関数．
    apply_hpss : bool
        HPSSを適用するか．
    """
    def __init__(self, stft_n_fft, stft_hop_length, stft_window, apply_hpss):
        self.stft_n_fft      = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_window     = stft_window
        self.apply_hpss      = apply_hpss

    def transform(self, y):
        """ オーディオをスペクトログラムに変換する．

        Parameters
        ----------
        y : ndarray
            オーディオ信号の配列（data.audio.Audio）．

        Returns
        -------
        ndarray, ndarray
            振幅，位相スペクトログラムの行列．
        """
        log.info("Start audio transformation.")
        # HPSS
        if self.apply_hpss:
            log.debug("Apply HPSS.")
            y = hpss(y)[0]

        # STFT
        log.debug("Apply STFT.")
        X_complex = stft(y, self.stft_n_fft, self.stft_hop_length, window=self.stft_window)
        return np.abs(X_complex), np.angle(X_complex)

    def inverse_transform(self, X_mag, X_ang, sr=None):
        """ スペクトログラムをオーディオに逆変換する．

        Parameters
        ----------
        X_mag : ndarray
            振幅スペクトログラムの行列．
        X_ang : ndarray
            位相スペクトログラムの行列．

        Returns
        -------
        ndarray
            :abbr:`ISTFT (inverse STFT)` によって得られたオーディオの配列．
        """
        log.info("Start audio inverse transformation.")
        X_complex = X_mag * np.exp(1j * X_ang)
        y = istft(X_complex, self.stft_hop_length)
        if sr is None:
            return y
        else:
            return Audio(y, sr)
