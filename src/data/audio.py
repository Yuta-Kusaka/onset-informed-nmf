import numpy as np
import librosa
import librosa.output
from logging import getLogger


log = getLogger(__name__)


def estimate_bpm(y, sr):
    """ オーディオのBPMを推定する．

    Parameters
    ----------
    y : ndarray
        オーディオ信号の配列．
    sr : float
        オーディオのサンプリングレート．

    Returns
    -------
    float
        推定されたオーディオのBPM．
    """
    onset_envelope = librosa.onset.onset_strength(y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_envelope, sr=sr)[0]
    log.debug("Estimated BPM: %f" % bpm)
    return bpm


class Audio:
    """ オーディオクラス．

    Parameters
    ----------
    y : ndarray
        オーディオ信号の配列．
    sr : float
        オーディオのサンプリングレート．
    """
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
        self.bpm = estimate_bpm(y, sr)

    @classmethod
    def load(cls, fname, sr, duration):
        """ wavファイルからオーディオを読み込む．

        Parameters
        ----------
        fname : str
            読み込むwavファイルのパス．
        sr : float
            読み込み後のサンプリングレート．
        duration : float
            読み込み後のオーディオの長さ．

        Returns
        -------
        data.audio.Audio
            オーディオクラス．
        """
        y, _ = librosa.load(fname, sr=sr, duration=duration)
        log.info("Load audio file: %s" % fname)
        return cls(y, sr)

    def save(self, fname):
        """ オーディオをwavファイルとして保存する．

        Parameters
        ----------
        fname : str
            保存するwavファイルのパス．
        """
        librosa.output.write_wav(fname, self.y, self.sr)
        log.info("Save audio file: %s" % fname)