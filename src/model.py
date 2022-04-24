import numpy as np
from logging import getLogger

EPSILON = 1.0e-20 # Wiener filterでゼロ除算を防ぐための定数


log = getLogger(__name__)


class Model:
    """ NMFモデルクラス

    Parameters
    ----------
    X_mag : ndarray
        振幅スペクトログラムの行列．
    X_phase : ndarray
        位相スペクトログラムの行列．ISTFTを行うときに利用．
    onset_matrix : ndarray
        オンセット行列．
    W : ndarray
        基底スペクトル行列．
    H : ndarray
        アクティベーション行列．
    S : ndarray
        バイナリマスク行列．
    """
    def __init__(self, X_mag, X_phase, onset_matrix):
        # inputs
        self.X_mag = X_mag
        self.onset_matrix = onset_matrix
        # outputs
        self.W = None
        self.H = None
        self.S = None
        # other
        self.X_phase = X_phase

    @classmethod
    def load(cls, fname):
        """ .npzファイルに保存したモデルを読み込む．

        Parameters
        ----------
        fname : str
            モデルファイルのパス．

        Returns
        -------
        src.model.Model
            読み込んだモデルオブジェクト．
        """
        log.info("Load model: %s" % fname)
        model_np = np.load(fname, allow_pickle=True)
        model = cls(model_np["X_mag"], model_np["X_phase"], model_np["onset_matrix"])
        model.W, model.H, model.S = model_np["W"], model_np["H"], model_np["S"]
        return model

    @classmethod
    def load_old(cls, fname):
        """ .npzファイルに保存したモデルを読み込む（古いバージョン用）．

        以前までのモデルは ``X_mag`` の代わりに ``X``， ``onset_matrix`` の代わりに ``onset_mat`` になっており，
        ``X_phase`` が存在しなかったので，現在のモデルに合うように読み込む．

        Parameters
        ----------
        fname : str
            モデルファイルのパス．

        Returns
        -------
        src.model.Model
            読み込んだモデルオブジェクト．
        """
        log.info("Load model: %s" % fname)
        model_np = np.load(fname)
        # Previous model has X instead of X_mag and onset_mat instead of onset_matrix.
        # And it does not have X_phase.
        model = cls(X_mag=model_np["X"], X_phase=None, onset_matrix=model_np["onset_mat"])
        model.W, model.H, model.S = model_np["W"], model_np["H"], model_np["S"]
        return model

    def save(self, fname):
        """.npzファイルにモデルを保存する．

        Parameters
        ----------
        fname : str
            モデルファイルを保存するパス．
        """
        log.info("Save model: %s" % fname)
        np.savez_compressed(fname, **vars(self))

    def fetch_component_num(self):
        """ モデルの基底数を得る．
        """
        return np.count_nonzero(np.any(self.onset_matrix == 1, axis=1))

    def reconst_spectrogram(self, target_components=None):
        """ 推定されたモデルからスペクトログラムを復元する．

        推定されたOI-NMFの変数（基底スペクトル，アクティベーション，バイナリマスク）から，
        ウィナーフィルタによりターゲット楽器音と伴奏楽器音の振幅スペクトログラムを復元する．　

        Parameters
        ----------
        target_components : list of int, optional
            復元に利用する基底の番号のリスト．デフォルトは ``None`` ．
            ``None`` が与えられた場合はオンセットが与えられた基底を利用して復元を行う．

        Returns
        -------
        ndarray, ndarray
            ターゲット楽器と伴奏楽器の振幅スペクトログラム．
        """
        # If target_components == None, set it as the number given the onsets.
        if target_components is None:
            target_components = range(self.fetch_component_num())
        # If target_components is int (such as 2, 5 ...), convert it to list for setting it as a row vector.
        if type(target_components) == int:
            target_components = [target_components]
        log.info("Recover the target spectrogram from the inferred model using components {}.".format(target_components))

        # Weiner filtering.
        W_target = self.W[:, target_components]
        H_target = self.H[target_components, :]
        if np.all(self.S) is None:
            S_target = np.ones_like(H_target)
            X_target = (W_target @ (H_target * S_target)) / (self.W @ self.H + EPSILON) * self.X_mag
        else:
            S_target = self.S[target_components, :]
            X_target = (W_target @ (H_target * S_target)) / (self.W @ (self.H * self.S) + EPSILON) * self.X_mag
        X_accomp = self.X_mag - X_target
        return X_target, X_accomp


class Model_Sinmf:
    def __init__(self, X_mag, X_phase, W_init, H_init):
        """ SI-NMFのモデルクラス．

        Score-informed NMF (SI-NMF)クラス．
        予め楽譜を用いて初期化された基底スペクトルとアクティベーションを
        用いてモデルを推定する．

        Parameters
        ----------
        X_mag : np.ndarray
            振幅スペクトログラムの行列．
        X_phase : np.ndarray
            位相スペクトログラムの行列．
        W_init : np.ndarray
            初期化された基底スペクトルの行列．
        H_init : np.ndarray
            初期化されたアクティベーションの行列．
        """
        # inputs
        self.X_mag = X_mag
        self.W_init = W_init
        self.H_init = H_init
        # outputs
        self.W = None
        self.H = None
        # other
        self.X_phase = X_phase

    @classmethod
    def load(cls, fname):
        log.info("Load model: %s" % fname)
        model_np = np.load(fname, allow_pickle=True)
        model = cls(model_np["X_mag"], model_np["X_phase"], model_np["W_init"], model_np["H_init"])
        model.W, model.H = model_np["W"], model_np["H"]
        return model
    
    def save(self, fname):
        log.info("Save model: %s" % fname)
        np.savez_compressed(fname, **vars(self))

    def reconst_spectrogram(self, K):
        W_target = self.W[:, :2*K]
        H_target = self.H[:2*K, :]
        X_target = ((W_target @ H_target) / (self.W @ self.H + 1.0e-20)) * self.X_mag
        X_accomp = self.X_mag - X_target
        return X_target, X_accomp


class SINMF_Model:
    def __init__(self, X_mag: np.ndarray, X_phase: np.ndarray, W_init: np.ndarray, H_init: np.ndarray, component_nums):
        self.X_mag   = X_mag
        self.X_phase = X_phase
        self.W_init  = W_init
        self.H_init  = H_init
        self.component_nums = component_nums
        self.W = None
        self.H = None

    @classmethod
    def load(cls, fname):
        log.info("Load model: %s" % fname)
        model_np = np.load(fname, allow_pickle=True)
        model = cls(model_np["X_mag"], model_np["X_phase"], model_np["W_init"], model_np["H_init"], model_np["component_nums"])
        model.W, model.H = model_np["W"], model_np["H"]
        return model
    
    def save(self, fname):
        log.info("Save model: %s" % fname)
        np.savez_compressed(fname, **vars(self))

    def reconst_spectrogram(self, target_inst_num: int=None) -> np.ndarray:
        """ 目的楽器の振幅スペクトログラムを構成

        Parameters
        ----------
        target_inst_num : int
            目的楽器の番号

        Returns
        -------
        np.ndarray
            目的楽器の振幅スペクトログラム
        """
        # 累積和で各楽器の基底本数 -> 基底の範囲に変換
        # 各基底は2本で1セットであることに注意
        target_components_range = 2 * np.cumsum(np.insert(self.component_nums, 0, 0)) 
        # 'target_inst_num'が'None'のときはすべてのスペクトログラムを返す
        if target_inst_num is None:
            X_inst_list = []
            for i in range(len(self.component_nums)):
                k_start = target_components_range[i]
                k_end =   target_components_range[i+1]
                X_inst_list.append(wiener_filter(self.W, self.H, self.X_mag, (k_start, k_end)))
            return X_inst_list
        # 楽器番号が指定されたときはそのスペクトログラムのみを返す
        else:
            k_start = target_components_range[target_inst_num]
            k_end   = target_components_range[target_inst_num+1]
            return wiener_filter(self.W, self.H, self.X_mag, (k_start, k_end))


def wiener_filter(W: np.ndarray, H: np.ndarray, X: np.ndarray, k_pair) -> np.ndarray:
    W_target = W[:, k_pair[0]:k_pair[1]]
    H_target = H[k_pair[0]:k_pair[1], :]
    return ((W_target @ H_target) / (W @ H + 1.0e-20)) * X