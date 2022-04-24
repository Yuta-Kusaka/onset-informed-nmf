import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import librosa
from librosa.display import specshow, TimeFormatter


def generate_axis_array(n, unit_type, sr, n_fft, hop_length):
    """ Generate an axis array for pcolormesh.

    Parameters
    ----------
    n : int
    unit_type : {"component", "time", "time_frame", "freq", "freq_bin"}
        Type of axis unit.
    sr : float
        Sampling rate of the input audio.
    n_fft : int
        FFT size of STFT.
    hop_length : int
        Hop length of STFT.

    Returns
    -------
    ndarray
        Axis array.
    """
    if unit_type == "component":
        return np.arange(n+1) + 0.5
    if unit_type == "time":
        return librosa.frames_to_time(np.arange(n+1), sr, hop_length)
    if unit_type == "time_frame":
        return np.arange(n+1)
    if unit_type == "freq" or "log_freq":
        return librosa.fft_frequencies(sr, n_fft)
    if unit_type == "freq_bin":
        return np.arange(n+1)


def decolate_axis(ax, unit_type):
    """ Decolate axis by a label and ticks.

    Parameters
    ----------
    ax : matplotlib.axis.Axis
        Target axis.
    unit_type : {"component", "time", "time_frame", "freq", "freq_bin"}
        Type of axis unit.
    """
    if unit_type == "component":
        ax.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_label_text("Component")
    if unit_type == "time":
        ax.set_major_formatter(TimeFormatter(unit="s", lag=False))
        ax.set_major_locator(ticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        ax.set_label_text("Time (sec)")
    if unit_type == "time_frame":
        ax.set_label_text("Time frame")
    if unit_type == "freq":
        ax.set_major_formatter(ticker.ScalarFormatter())
        ax.set_label_text("Frequency (Hz)")
    if unit_type == "freq_bin":
        ax.set_label_text("Frequency bin")
    if unit_type == "log_freq":
        ax.set_major_formatter(ticker.LogFormatter())
        ax.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.set_label_text("Frequency (Hz)")


def plot_heatmap(data, x_axis, y_axis, plot_percentile, cmap, sr, n_fft, hop_length, ax):
    """ Plot any heatmap.

    Parameters
    ----------
    data : 2d ndarray
        Data to plot.
    x_axis : ndarray
        X axis array.
    y_axis : ndarray
        Y axis array.
    plot_percentile : list of int
        Percentile for plotting ([vmin_percentile, vmax_percentile]).
    cmap : str
        Colormap.
    sr : float
        Sampling rate of the input audio.
    n_fft : int
        FFT size of the STFT.
    hop_length : int
        Hop size of the STFT.
    ax : matplotlib.axis.Axis
        Axis object to plot.

    Returns
    -------
    matplotlib.axis.Axis
        Plotted axis object.
    """
    vmin = np.percentile(data, plot_percentile[0])
    vmax = np.percentile(data, plot_percentile[1])
    x_axis_array = generate_axis_array(data.shape[1], x_axis, sr, n_fft, hop_length)
    y_axis_array = generate_axis_array(data.shape[0], y_axis, sr, n_fft, hop_length)
    ax.pcolormesh(x_axis_array, y_axis_array, data, cmap=cmap, vmax=vmax, vmin=vmin, rasterized=True)
    decolate_axis(ax.xaxis, x_axis)
    decolate_axis(ax.yaxis, y_axis)
    return ax
    

class Visualizer:
    def __init__(self, conf):
        self.conf = conf
    
    def spectrogram(self, X, cmap="inferno", ax=None):
        """ Plot spectrogram.

        Parameters
        ----------
        X : 2d ndarray
            A matrix representing the spectrogram.
        cmap : str, optional
            Colormap, by default "inferno"
        ax : matplotlib.axis.Axis, optional
            Axis object to plot, by default None

        Returns
        -------
        matplotlib.axis.Axis
            Plotted axis object.
        """
        if ax is None: ax = plt.gca()

        X_dB = librosa.amplitude_to_db(X, ref=np.max)
        sr = self.conf.audio.sr
        hop_length = self.conf.transform_audio.stft_hop_length
        ax = specshow(X_dB, x_axis="time", y_axis="linear", cmap=cmap, sr=sr, hop_length=hop_length, ax=ax)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Frequency (Hz)")
        return ax

    def dictionary(self, W, onset_K=None, unit="freq", plot_percentile=(0, 99), cmap="OrRd", ax=None):
        """ Plot dictionary matrix W.

        Parameters
        ----------
        W : 2d ndarray
            A dictionary matrix W.
        unit : str, optional
            Y-axis unit, by default "freq"
        onset_K : int
            Number of components given the onset.
            If 'onset_K' is not None, a line splitting the onset components and the other components will be drawn.
        plot_percentile : tuple, optional
            Percentile for plotting ([vmin_percentile, vmax_percentile]), by default (0, 99)
        cmap : str, optional
            Colormap, by default "OrRd"
        ax : matplotlib.axis.Axis, optional
            An axis object to plot, by default None

        Returns
        -------
        matplotlib.axis.Axis
            Plotted axis object.
        """
        if ax is None: ax = plt.gca()
    
        sr         = self.conf.audio.sr
        n_fft      = self.conf.transform_audio.stft_n_fft
        hop_length = self.conf.transform_audio.stft_hop_length

        plot_heatmap(W, "component", unit, plot_percentile, cmap, sr, n_fft, hop_length, ax)
        if onset_K is not None:
            ax.axvline(onset_K + 0.5, color="black")
        return ax

    def activation(self, H, S=None, onset_K=None, unit="time", plot_percentile=(0, 99), cmap="OrRd", ax=None):
        """ Plot activation matrix H.

        Parameters
        ----------
        H : 2d ndarray
            A activation matrix H.
        onset_K : int
            Number of components given the onset.
            If 'onset_K' is not None, a line splitting the onset components and the other components will be drawn.
        unit : str, optional
            X-axis unit, by default "time"
        plot_percentile : tuple, optional
            Percentile for plotting ([vmin_percentile, vmax_percentile]), by default (0, 99)
        cmap : str, optional
            Colormap, by default "OrRd"
        ax : matplotlib.axis.Axis, optional
            An axis object to plot, by default None

        Returns
        -------
        matplotlib.axis.Axis
            Plotted axis object.
        """
        if S is not None: H = H * S
        if ax is None: ax = plt.gca()

        sr         = self.conf.audio.sr
        n_fft      = self.conf.transform_audio.stft_n_fft
        hop_length = self.conf.transform_audio.stft_hop_length

        plot_heatmap(H, unit, "component", plot_percentile, cmap, sr, n_fft, hop_length, ax)
        if onset_K is not None:
            ax.axhline(onset_K + 0.5, color="black")
        return ax
    
    def binary(self, S, onset_K=None, unit="time", cmap="OrRd", ax=None):
        """ Plot binary matrix such as S or an onset matrix.

        Parameters
        ----------
        S : 2d ndarray
            A binary matrix.
        onset_K : int
            Number of components given the onset.
            If 'onset_K' is not None, a line splitting the onset components and the other components will be drawn.
        unit : str, optional
            X-axis unit, by default "time"
        cmap : str, optional
            Colormap, by default "OrRd"
        ax : matplotlib.axis.Axis, optional
            An axis object to plot, by default None

        Returns
        -------
        matplotlib.axis.Axis
            Plotted axis object.
        """
        if ax is None: ax = plt.gca()

        sr         = self.conf.audio.sr
        n_fft      = self.conf.transform_audio.stft_n_fft
        hop_length = self.conf.transform_audio.stft_hop_length

        plot_heatmap(S, unit, "component", (0, 100), cmap, sr, n_fft, hop_length, ax)
        if onset_K is not None:
            ax.axhline(onset_K + 0.5, color="black")
        return ax

    def SImetrics(self, df1, df2, metric="GSISDR", ax=None):
        if ax is None: ax = plt.gca()

        w = 0.3
        x = np.array(range(8))
        m1 = df1.groupby("name").mean()[metric].values
        s1 = df1.groupby("name").std(ddof=0)[metric].values
        m2 = df2.groupby("name").mean()[metric].values
        s2 = df2.groupby("name").std(ddof=0)[metric].values

        ax.bar(x - 0.5*w, m1, w, yerr=s1, edgecolor="black", capsize=2, label="OI-NMF")
        ax.bar(x + 0.5*w, m2, w, yerr=s2, edgecolor="black", capsize=2, label="Bayesian NMF")
        ax.set_xticks(x)
        xlabels = ["Bebop", "Cool", "Free", "Funk", "Fusion", "Latin", "Model", "Swing"]
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("dB")
        ax.legend()
        ax.grid(axis="x")
        return ax

def result_box_and_dot(result_df, metric, hue_key, palette_box=None, palette_dot=None, ax=None):
    if ax is None: ax = plt.gca()
    ax.set_axisbelow(True)
    ax.axhline(0, color='0.0', linewidth=1, zorder=1)
    sns.boxplot(data=result_df, x="name", y=metric, hue=hue_key, showfliers=False, whis=[25, 75], palette=palette_box, linewidth=0.8, ax=ax, boxprops={'zorder': 2})
    sns.stripplot(data=result_df, x="name", y=metric, hue=hue_key, size=4, linewidth=0.8, palette=palette_dot, edgecolor="k", jitter=0., dodge=True, ax=ax, zorder=3)
    # legend設定
    handler, _ = ax.get_legend_handles_labels()
    legend_list = result_df[hue_key].unique()
    ax.legend(handler, legend_list, bbox_to_anchor=(0.5, -0.16), loc="upper center", borderaxespad=0, ncol=len(legend_list))
    # labelとその他設定
    ax.margins(y=0.05)
    ax.grid(axis='y', zorder=-10)
    ax.set_xlabel("")
    y_labels = {"GSISDR": "SI-SDR Improvement (dB)", "SISIR_est": "SI-SIR (dB)", "SISAR_est": "SI-SAR (dB)"}
    ax.set_ylabel(y_labels[metric])
    # boxplotの色設定
    for i, box in enumerate(ax.artists):
        box.set_edgecolor("0.2")
    for line in ax.lines:
        line.set_color("0.2")
