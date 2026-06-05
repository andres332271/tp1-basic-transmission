import os
import sys
import numpy as np
from numpy.fft import fftshift
from scipy.signal import welch as _welch, lfilter
from scipy.special import erfc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import root_raised_cosine as _rrc


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


def simulate_txrx(M, L, N, rolloff, h_taps, EbNo_db, BR=32e9):
    """Pipeline completo TX→AWGN→MF→slicer. Devuelve dict con señales y métricas."""
    # QAM modulator + upsampling
    ak  = qammod(np.random.randint(0, M, L), M)
    xup = np.zeros(L * N, dtype=complex)
    xup[::N] = ak * N

    # RRC TX filter
    h       = _rrc(BR/2, N*BR, rolloff, h_taps)
    h_delay = (len(h) - 1) // 2
    yup     = lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))[h_delay:]

    # Canal AWGN
    k_bits  = np.log2(M)
    SNR_ch  = 10**(EbNo_db / 10) * k_bits / N
    Pn      = np.var(yup) / SNR_ch
    noise   = np.sqrt(Pn / 2) * (np.random.randn(len(yup)) + 1j * np.random.randn(len(yup)))
    rx      = yup + noise

    # Matched filter + decimación
    ymf     = lfilter(np.conj(h[::-1]), 1, np.concatenate([rx, np.zeros(h_delay)]))[h_delay:]
    rx_down = ymf[::N]

    # Normalización antes del slicer
    norm_factor = N * np.sum(h**2)
    rx_norm     = rx_down / norm_factor

    # Compensación de delay
    delay_est = estimate_delay(ak, rx_norm)
    if delay_est > 0:
        rx_aligned, tx_aligned = rx_norm[delay_est:], ak[:len(rx_norm) - delay_est]
    elif delay_est < 0:
        rx_aligned, tx_aligned = rx_norm[:len(ak) + delay_est], ak[-delay_est:]
    else:
        rx_aligned, tx_aligned = rx_norm, ak[:len(rx_norm)]

    # Slicer + BER (60% final para descartar transitorios)
    ak_hat   = slicer(rx_aligned, M)
    start    = int(0.4 * len(ak_hat))
    n_errors = int(np.sum(ak_hat[start:] != tx_aligned[start:]))
    ser_sim  = n_errors / len(ak_hat[start:])
    ber_sim  = ser_sim / k_bits

    return {
        'ak': ak, 'xup': xup, 'yup': yup,
        'rx': rx, 'ymf': ymf,
        'rx_norm': rx_norm,
        'h': h, 'h_delay': h_delay, 'norm_factor': norm_factor,
        'delay_est': delay_est,
        'ber_sim': ber_sim, 'ser_sim': ser_sim, 'n_errors': n_errors,
    }


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
