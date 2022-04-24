import os
from glob import glob
import logging


log = logging.getLogger(__name__)


def load_list_file(fname):
    """リストファイルを読み込む．

    入力wavファイルやオンセットcsvファイルのパスが示されている.lstファイルを読み込んで，
    パスのリストに変換する．

    Parameters
    ----------
    fname : str
        リストファイルのパス．

    Returns
    -------
    list of str
        リストファイルに含まれるパスからなるリスト．
    """
    # convert to list with eliminating /n
    with open(fname, mode="r") as f:
        path_list = [line.rstrip(os.linesep) for line in f.readlines()]
    log.info("Load list file: %s" % fname)

    return path_list


class URMP:
    def __init__(self, dir: str):
        """URMPデータセットの各種情報を扱うためのクラス．

        Parameters
        ----------
        dir : str
            UMRPデータセットのディレクトリ．
        """
        self.dir = dir

        song_dir_list = sorted([dirname for dirname in os.listdir(self.dir) if not dirname.startswith(".")])
        song_dir_list.remove("Supplementary_Files")
        self.song_dir_list = song_dir_list

    def set_song(self, id: str):
        """情報を読み取るための楽曲IDを設定する．

        Parameters
        ----------
        id : str
            楽曲ID．
        """
        self.id = id
        song_dir_name = self.song_dir_list[id-1]
        self.song_dir = self.dir + song_dir_name + "/"
        
        self.title = song_dir_name.split("_")[1]
        self.instruments = song_dir_name.split("_")[2:]
    
    def get_data_path(self, type: str):
        """楽曲の関連ファイルのパスを得る．

        Parameters
        ----------
        type : {"mix", "sep", "notes"}
            取得したい関連ファイルの種類．
            ``"mix"``: 混合音のwavファイル．
            ``"sep"``: 各分離音のwavファイルのリスト．
            ``"notes"``: 各楽器音の楽譜を示したテキストファイルのリスト．

        Returns
        -------
        list of str
            各種ファイルのパスリスト．
        """
        type_list = ("mix", "sep", "notes")
        assert type in type_list, "'type' must be in ('mix', 'sep', 'notes')"

        data_dir = self.dir + self.song_dir_list[self.id-1] + "/"
        if type == "mix":
            return glob(data_dir + "AuMix_*.wav")[0]
        if type == "sep":
            return sorted(glob(data_dir + "AuSep_*.wav"))
        if type == "notes":
            return sorted(glob(data_dir + "Notes_*.txt"))
