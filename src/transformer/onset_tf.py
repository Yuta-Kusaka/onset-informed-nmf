from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from copy import deepcopy
from scipy import stats
from librosa import time_to_frames, samples_to_frames
from logging import getLogger


log = getLogger(__name__)


def omit_part_of_onset(base_onset, remaining_ratio):
    """ オンセットの一部を省略する．

    オンセットが欠落したときの分離精度を調べる実験に利用．

    Parameters
    ----------
    base_onset : src.data.onset.Onset
        変換前のオンセットクラス.
    remaining_ratio : float
        オンセットを残す割合．(0.0, 1.0]の範囲で与える．

        * 0.0より少しだけ大きい値（0.01など）: 各基底に対してひとつだけオンセットを残す．
        * 1.0: 全てのオンセットを残す．
    """
    assert 0 < remaining_ratio <= 1, "remaining_ratio should be in (0, 1.0]."

    # 各基底についてランダムに残すインデックスを抽出．
    # インデックスはonset全体で共通なので最後にソートする．
    remaining_index = np.array([], dtype=int)
    for num in base_onset.notenum_set:
        num_index = np.where(base_onset.notenum_seq == num)[0]
        remaining_size = np.ceil(len(num_index) * remaining_ratio).astype(int)
        remaining_num_index = np.random.choice(num_index, remaining_size, replace=False)
        remaining_index = np.append(remaining_index, remaining_num_index)
    remaining_index = np.sort(remaining_index)

    base_onset.time_seq = base_onset.time_seq[remaining_index]
    base_onset.notenum_seq = base_onset.notenum_seq[remaining_index]


def add_noise_to_onset(base_onset, noise_mean, noise_std):
    """ オンセットにずれのノイズを加える．

    オンセットに正規分布に従うずれを加える．オンセットがずれた場合の分離精度を調べる実験に利用．

    Parameters
    ----------
    base_onset : src.data.onset.Onset
        変換前のオンセットクラス．
    noise_mean : float
        正規分布の平均．
    noise_std : float
        正規分布の標準偏差．
    """
    for i, time in enumerate(base_onset.time_seq):
        time += stats.norm.rvs(loc=noise_mean, scale=noise_std)
        # if onset added noise is lower than 0, return 0
        base_onset.time_seq[i] = np.max([time, 0])


def calc_onset_width(width_as_note, bpm, sr, hop_length):
    """ オンセット行列におけるオンセットの長さを計算する．

    オンセット行列は，各オンセットのタイムスタンプを開始時刻とし，その後ろに一定の許容フレームを持つ．
    この許容フレームの長さを音源のBPMなどから相対的に算出する．

    Parameters
    ----------
    width_as_note : int
        許容フレームの音符としての長さ（ ``4`` :4分音符，``8`` : 8分音符など）．
    bpm : float
        入力オーディオのBPM．
    sr : float
        入力オーディオのサンプリングレート．
    hop_length : int
        音声変換クラスのSTFTのホップサイズ．

    Returns
    -------
    int
        時間-周波数領域での時間フレーム単位での許容フレームサイズ．
    """
    assert isinstance(width_as_note, int), "width_as_note must be integer."

    coef = 4 / width_as_note
    frames_per_beat = (60 * sr) / (bpm * hop_length)
    return int(round(coef * frames_per_beat))


def build_onset_matrix(onset, shape, onset_width, shift_frame, sr, hop_length):
    """ オンセットクラスをオンセット行列に変換する．

    Parameters
    ----------
    onset : src.data.onset.Onset
        オンセット．
    shape : (int, int)
        オンセット行列のサイズ．アクティベーションと同じサイズ（基底数 x 時間フレーム数）になるように設定する．
    onset_width : int
        計算したオンセットの許容フレームサイズ．
    shift_frame : int
        オンセットのシフト幅（例: ``1`` ... 後ろに1フレームずらす）．
    sr : float
        入力オーディオのサンプリングレート．
    hop_length : int
        音声変換クラスのSTFTホップサイズ．

    Returns
    -------
    ndarray
        オンセット行列．アクティベーションやバイナリマスクと同サイズ．
    """
    onset_matrix = np.zeros(shape, dtype=int)
    for k, notenum in enumerate(onset.notenum_set):
        k_index = np.where(onset.notenum_seq == notenum)[0]
        k_start_frames = time_to_frames(onset.time_seq[k_index], sr, hop_length)
        for start_frame in k_start_frames:
            start_frame += shift_frame
            end_frame = start_frame + onset_width
            onset_matrix[k, start_frame:end_frame] = 1
    return onset_matrix


class OnsetTransformer:
    """オンセット変換クラス

    Parameters
    ----------
    K : int
        基底数
    hop_length : int
        STFTのホップサイズ．AudioTransformerと同じ値に設定する．
    width_as_note : int
        オンセットの許容フレームの音符としての長さ．
    remaining_ratio : float
        オンセットを残す割合．
    includes_noise : bool
        ノイズによるずれを含めるか否か．
    noise_mean : float
        ずれの正規分布の平均．
    noise_std : float
        ずれの正規分布の標準偏差．
    shift_frame : int
        オンセットのシフト．
    without_onset : bool
        Trueにするとオンセットを利用しない．
    """
    def __init__(self, K, hop_length, width_as_note, remaining_ratio, includes_noise, noise_mean, noise_std, shift_frame, without_onset):
        self.K               = K
        self.hop_length      = hop_length
        self.width_as_note   = width_as_note
        self.remaining_ratio = remaining_ratio
        self.includes_noise  = includes_noise
        self.noise_mean      = noise_mean
        self.noise_std       = noise_std
        self.shift_frame     = shift_frame
        self.without_onset   = without_onset
    
    def transform(self, base_onset, audio):
        """オンセットをオンセット行列に変換する

        Parameters
        ----------
        base_onset : src.data.onset.Onset
            変換するオンセット．
        audio : src.data.audio.Audio
            変換するオンセットに付随するオーディオ．

        Returns
        -------
        ndarray
            オンセット行列．
        """
        log.info("Transform onset into matrix form.")
        onset = deepcopy(base_onset)
        if self.remaining_ratio < 1:
            log.debug("Omit part of the onsets.")
            omit_part_of_onset(onset, self.remaining_ratio)
        if self.includes_noise:
            log.debug("Add noise to the onsets.")
            add_noise_to_onset(onset, self.noise_mean, self.noise_std)
        
        onset_width = calc_onset_width(self.width_as_note, audio.bpm, audio.sr, self.hop_length)
        n_col = self.K
        n_row = samples_to_frames(len(audio.y), self.hop_length) + 1
        if self.without_onset:
            return np.zeros((n_col, n_row), dtype=int)
        else:
            return build_onset_matrix(onset, (n_col, n_row), onset_width, self.shift_frame, audio.sr, self.hop_length)
