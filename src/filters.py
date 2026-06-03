import numpy as np


def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n


def raised_cosine(fc, fs, rolloff, n_taps, t0=0):
    rolloff = rolloff + 0.0001
    Ts = 1 / fs
    T = 1 / fc

    n_taps = round_odd(n_taps)

    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    h_v = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v) \
          / (1 - (2 * rolloff * tn_v)**2)

    h_v = h_v / np.sum(h_v)
    return h_v


def root_raised_cosine(fc, fs, rolloff, n_taps, t0=0):
    Ts = 1 / fs
    rolloff = rolloff + 0.0001
    T = 1 / fc

    n_taps = round_odd(n_taps)

    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    numerator = (
        np.sin(np.pi * (1 - rolloff) * tn_v)
        + 4 * rolloff * tn_v * np.cos(np.pi * (1 + rolloff) * tn_v)
    )
    denominator = np.pi * tn_v * (1 - (4 * rolloff * tn_v)**2)

    with np.errstate(invalid='ignore', divide='ignore'):
        h_v = numerator / denominator

    center = (n_taps - 1) // 2
    h_v[center] = (1 + rolloff * (4/np.pi - 1))

    h_v = h_v / np.sum(h_v)
    return h_v
