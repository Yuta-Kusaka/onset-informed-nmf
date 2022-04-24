import numpy as np
from logging import getLogger


log = getLogger(__name__)


def parse_onset_csv(onset_csv, audio_duration):
    """ オンセットcsvファイルをパースする．

    Parameters
    ----------
    onset_csv : ndarray
        オンセットcsvファイルから読み込んだオンセットの行列．オンセットクラスの例を参照．
    audio_duration : float
        オンセット読み込み時の長さ．オーディオと合わせる．

    Returns
    -------
    ndarray, ndarray, list
        ``time_seq``, ``notenum_seq``, ``notenum_set``
    """
    # Truncate onsets to audio_duration range
    seq_len = np.count_nonzero(onset_csv[:, 0] < audio_duration)

    time_seq    = onset_csv[:seq_len, 0]
    notenum_seq = onset_csv[:seq_len, 1]
    notenum_set = np.unique(notenum_seq).astype(int)
    return time_seq, notenum_seq, notenum_set


class Onset:
    """ オンセットクラス．

    オンセットは以下のように表される．

    +------------------------+----------+
    |タイムスタンプ [sec]    |ノート番号|
    +========================+==========+
    |4.005442176870748217e-01|66        |
    +------------------------+----------+
    |6.907936507936508486e-01|66        |
    +------------------------+----------+
    |9.462131519274377123e-01|67        |
    +------------------------+----------+
    |1.393197278911564752e+00|63        |
    +------------------------+----------+
    |...                     |...       |
    +------------------------+----------+

    Parameters
    ----------
    time_seq : ndarray
        オンセットのタイムスタンプの配列（上図左列）．
    notenum_seq : ndarray of int
        タイムスタンプに対応するオンセットのノート番号（上図右列）．
    notenum_set : list of int
        存在するオンセットのノート番号集合（上の例だと[66, 67, 63]）．
    """
    def __init__(self, time_seq, notenum_seq, notenum_set):
        self.time_seq = time_seq
        self.notenum_seq = notenum_seq
        self.notenum_set = notenum_set

    @classmethod
    def load(cls, fname, audio_duration):
        """ csvファイルからオンセットを読み込む．

        Parameters
        ----------
        fname : str
            読み込むオンセットcsvファイルのパス．
        audio_duration : float
            オンセット読み込み時の長さ．オーディオの長さと合わせる．

        Returns
        -------
        data.onset.Onset
            オンセットクラス．
        """
        onset_csv = np.loadtxt(fname)
        log.info("Load onset file: %s" % fname)

        return cls(*parse_onset_csv(onset_csv, audio_duration))