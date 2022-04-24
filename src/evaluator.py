import os
from glob import glob
from museval import evaluate
import numpy as np
import pandas as pd
from numpy.linalg import norm
import librosa
import museval
from logging import getLogger

from src.data.audio import Audio
from src.transformer.audio_tf import AudioTransformer
from src.model import Model


log = getLogger(__name__)


def is_model_old(model_fname: str) -> bool:
    """モデルが旧モデルか確認する

    Parameters
    ----------
    model_fname : str
        モデルのパス．

    Returns
    -------
    bool
        モデルの判定結果．旧モデルなら`True`．
    """
    m = np.load(model_fname, mmap_mode='r')
    return 'X' in m.files



def fetch_ref_paths(fname):
    """ 混合音源のパスから，ターゲットと伴奏音源のパスを得る．

    Parameters
    ----------
    fname : str
        混合音源のパス．

    Returns
    -------
    str, str, str
        ターゲット，伴奏，混合音源のパス．
    """
    fname_target = fname.replace("raw", "interim").replace("MIX", "TARGET")
    fname_accomp = fname.replace("raw", "interim").replace("MIX", "ACCOMP")
    return fname_target, fname_accomp, fname


def load_ref_audio(fname, sr, duration):
    """混合音源のパスから，参照音源を読み込む．

    Parameters
    ----------
    fname : str
        混合音源のパス．
    sr : float
        オーディオのサンプリングレート．OI-NMFを適用したのと同じパラメータを使用．
    duration : float
        オーディオの長さ．
    Returns
    -------
    ndarray, ndarray, ndarray
        ターゲット，伴奏，混合音の信号の配列．
    """
    ref_fname = fetch_ref_paths(fname)
    x_ref_target, _ = librosa.load(ref_fname[0], sr=sr, duration=duration)
    x_ref_accomp, _ = librosa.load(ref_fname[1], sr=sr, duration=duration)
    x_ref_mix   , _ = librosa.load(ref_fname[2], sr=sr, duration=duration)
    return x_ref_target, x_ref_accomp, x_ref_mix


def preproccess_sources(x_est_tuple, x_ref_tuple):
    """オーディオを前処理して配列に変換する．

    Parameters
    ----------
    x_est_tuple : (ndarray, ndarray)
        推定されたターゲットと伴奏音源の配列を含むタプル．
    x_ref_tuple : (ndarray, ndarray, ndarray)
        正解のターゲット，伴奏，混合根源の配列を含むタプル．

    Returns
    -------
    (3d ndarray, 3d ndarray, 3d ndarray)
        推定，正解，混合音源から構成される3次元配列を含むタプル．
        3次元配列の次元は(ソース番号，サンプル数，チャンネル数)．
    """
    est_array = np.expand_dims(np.vstack(x_est_tuple), 2)
    ref_array = np.expand_dims(np.vstack((x_ref_tuple[0], x_ref_tuple[1])), 2)
    mix_array = np.expand_dims(np.vstack((x_ref_tuple[2], x_ref_tuple[2])), 2)
    # Adjust signals length. The reference signals are not padded or truncated.
    ref_array, est_array = museval.pad_or_truncate(ref_array, est_array)
    _        , mix_array = museval.pad_or_truncate(ref_array, mix_array)
    return est_array, ref_array, mix_array


def compute_metrics(source_est, sources_ref, source_idx):
    """分離精度評価指標を計算する．

    指標はSI-SDR, SI-SIR, SI-SAR．

    Parameters
    ----------
    source_est : ndarray
        推定音源の配列．
    sources_ref : ndarray
        正解音源の配列
    source_idx : int
        計算する音源の番号．

    Returns
    -------
    float, float, float
        SI-{SDR, SIR, SAR}．
    """
    sources_ref = sources_ref.T

    ref_projection = sources_ref.T @ sources_ref
    s = sources_ref[:, source_idx]
    scale = (s @ source_est) / ref_projection[source_idx, source_idx]

    e_target = scale * s
    e_res    = source_est - e_target
    SISDR = 10 * np.log10((e_target ** 2).sum() / (e_res ** 2).sum())

    ref_onto_res = np.dot(sources_ref.T, e_res)
    b = np.linalg.solve(ref_projection, ref_onto_res)

    e_interf = np.dot(sources_ref, b)
    e_artif  = e_res - e_interf
    SISIR = 10 * np.log10((e_target ** 2).sum() / (e_interf ** 2).sum())
    SISAR = 10 * np.log10((e_target ** 2).sum() / (e_artif ** 2).sum())

    return SISDR, SISIR, SISAR


def fetch_mix_fname_from_model_fname(model_fname):
    """モデルのパスから混合音源のパスを得る．

    Parameters
    ----------
    model_fname : str
        モデルのパス．

    Returns
    -------
    str
        混合音源のパス．
    """
    mix_fname = os.path.splitext(os.path.basename(model_fname))[0][:-2]
    mix_fname = glob("data/**/*{}.wav".format(mix_fname), recursive=True)[0]
    return mix_fname


class Evaluator:
    """ 評価用クラス．

    分離音のSI-{SDR, SIR, SAR}を計算して評価を行う．

    Parameters
    ----------
    fname : str
        混合音源のパス．
    sr : float
        混合音源のサンプリングレート．
    duration : float
        混合音源の長さ．
    """
    def __init__(self, fname, sr, duration):
        self.fname    = fname
        self.sr       = sr
        self.duration = duration

    def evaluate(self, x_est_tuple):
        """指標を計算する．

        Parameters
        ----------
        x_est_tuple : tuple of ndarray
            推定されたターゲットと伴奏音源の配列を含むタプル．

        Returns
        -------
        Tuple of (ndarray)
            SI-{SDR, SIR, SAR}の配列を含むタプル．
            各配列は推定音源，混合音源の値と，その差（改善率）を含む．
        """
        x_ref_tuple = load_ref_audio(self.fname, self.sr, self.duration)
        est_array, ref_array, mix_array = preproccess_sources(x_est_tuple, x_ref_tuple)

        SISDR = np.empty(3)
        SISIR = np.empty(3)
        SISAR = np.empty(3)
        source_est  = est_array[0, :, 0]
        sources_ref = ref_array[:, :, 0]
        source_mix  = mix_array[0, :, 0]
        SISDR[0], SISIR[0], SISAR[0] = compute_metrics(source_est, sources_ref, 0)
        SISDR[1], SISIR[1], SISAR[1] = compute_metrics(source_mix, sources_ref, 0)
        SISDR[2], SISIR[2], SISAR[2] = SISDR[0] - SISDR[1], SISIR[0] - SISIR[1], SISAR[0] - SISAR[1]
        return SISDR, SISIR, SISAR

    @classmethod
    def evalutate_dir(cls, dir_name, conf):
        """ディレクトリに含まれるモデル全てを評価する．

        Parameters
        ----------
        dir_name : str
            評価するディレクトリのパス．
        conf : omegaconf
            Hydraで読み込んだ設定ファイル．
        """
        df_result = pd.DataFrame([], columns=["name",
                                              "num",
                                              "SISDR_est",
                                              "SISDR_mix",
                                              "GSISDR",
                                              "SISIR_est",
                                              "SISIR_mix",
                                              "GSISIR",
                                              "SISAR_est",
                                              "SISAR_mix",
                                              "GSISAR"])
        audio_name_list = ["Bebop",
                           "Cool",
                           "Free",
                           "Funk",
                           "Fusion",
                           "Latin",
                           "Modal",
                           "Swing"]

        for audio_name in audio_name_list:
            for i in range(conf.others.experiment_num):
                model_fname = dir_name + "/MusicDelta_{}Jazz_MIX_{}.npz".format(audio_name, i)
                print(model_fname)
                if is_model_old(model_fname):
                    model = Model.load_old(model_fname)
                    print('Load old model')
                else:
                    model = Model.load(model_fname)
                    print('Load new model')
                X_target, X_accomp = model.reconst_spectrogram()

                mix_fname = fetch_mix_fname_from_model_fname(model_fname)
                audio_transformer = AudioTransformer(**conf.transform_audio)
                x_target = audio_transformer.inverse_transform(X_target, model.X_phase)
                x_accomp = audio_transformer.inverse_transform(X_accomp, model.X_phase)
                x_est = (x_target, x_accomp)

                evaluator = cls(mix_fname, **conf.audio)
                temp = evaluator.evaluate(x_est)
                ser = pd.Series([audio_name, i, *np.hstack(temp)], index=df_result.columns)
                df_result = df_result.append(ser, ignore_index=True)
                df_result.to_csv("result.csv")
