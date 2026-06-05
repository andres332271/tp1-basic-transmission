import numpy as np
from numpy.fft import fftshift
from scipy.signal import welch as _welch
from scipy.special import erfc
import matplotlib.pyplot as plt


def qammod(x, M):
    m = int(np.sqrt(M))
    re = 2*(x % m) - m + 1
    im = 2*(x // m) - m + 1
    return re + 1j*im


def slicer(rx, M):
    const = qammod(np.arange(M), M)
    idx = np.argmin(np.abs(rx[:, None] - const), axis=1)
    return const[idx]


def estimate_delay(tx_symbols, rx_symbols):
    Lcorr = min(len(tx_symbols), len(rx_symbols))
    corr = np.correlate(rx_symbols[:Lcorr], tx_symbols[:Lcorr], mode='full')
    return np.argmax(np.abs(corr)) - (Lcorr - 1)


def ber_mqam(EbNo_db, M):
    k = np.log2(M)
    EbNo = 10**(EbNo_db / 10)
    return (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt(3*k*EbNo / (2*(M - 1))))


def welch_psd(sig, sample_rate, nperseg):
    """Devuelve (f, Pxx) con fftshift aplicado, sin normalizar."""
    f_w, Pxx = _welch(sig, sample_rate, nperseg=nperseg, return_onesided=False)
    return fftshift(f_w), fftshift(Pxx)


def eye_diagram(sig, sps, title, num_traces=200, levels=None, save_path=None):
    span = sps * 2 + 1
    traces = [np.real(sig[i*sps : i*sps + span])
              for i in range(num_traces)
              if i*sps + span <= len(sig)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for trace in traces:
        ax.plot(trace, alpha=0.3, color='steelblue')
    for x in range(0, 2*sps + 1, sps):
        ax.axvline(x, color='tomato', linestyle='--', linewidth=1.2,
                   label='Instante de muestreo' if x == 0 else None)
    if levels is not None:
        for y in levels:
            ax.axhline(y, color='black', linestyle='-.', linewidth=1.0,
                       label='Nivel de símbolo' if y == levels[0] else None)
    ax.set_title(title)
    ax.set_xlabel("Muestra")
    ax.set_xlim(-0.2, span - 1 + 0.2)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
