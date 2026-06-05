import numpy as np
from scipy.special import erfc


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
