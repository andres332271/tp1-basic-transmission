import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import root_raised_cosine
from tools import qammod, slicer, estimate_delay, ber_mqam

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(msg)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
M        = 16
L        = int(1e4)   # aumentar L para Eb/N0 altos
BR       = 32e9
N        = 2
rolloff  = 0.1
h_taps   = 203
EbNo_db  = 10

T  = 1 / BR
fs = N * BR
Ts = 1 / fs

log("=== Basic TX-RX Parameters ===")
log(f"M        = {M}-QAM")
log(f"L        = {L} symbols")
log(f"BR       = {BR/1e9:.1f} GBd")
log(f"N        = {N}")
log(f"rolloff  = {rolloff}")
log(f"h_taps   = {h_taps}")
log(f"Eb/N0    = {EbNo_db} dB")
log(f"fs       = {fs/1e9:.1f} GHz")
log()

# -------------------------------------------------
# QAM Modulator
# -------------------------------------------------
x_aux = np.random.randint(0, M, L)
ak = qammod(x_aux, M)

# -------------------------------------------------
# Upsampling
# -------------------------------------------------
xup = np.zeros(L * N, dtype=complex)
xup[::N] = ak * N

# -------------------------------------------------
# RRC TX filter
# -------------------------------------------------
h = root_raised_cosine(BR/2, fs, rolloff, h_taps)
h_delay = 0

yup = lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
yup = yup[h_delay:]

# -------------------------------------------------
# Channel AWGN
# -------------------------------------------------
k_bits  = np.log2(M)
EbNo    = 10**(EbNo_db / 10)
SNR_slc = EbNo * k_bits
SNR_ch  = SNR_slc / N

Ps   = np.var(yup)
Pn   = Ps / SNR_ch
noise = np.sqrt(Pn / 2) * (np.random.randn(len(yup)) + 1j * np.random.randn(len(yup)))
rx   = yup + noise

# -------------------------------------------------
# RX: matched filter + decimation
# -------------------------------------------------
h_mf  = np.conj(h[::-1])
ymf   = lfilter(h_mf, 1, np.concatenate([rx, np.zeros(h_delay)]))
ymf   = ymf[h_delay:]

PHASE    = 0
rx_down  = ymf[PHASE::N]

# -------------------------------------------------
# Delay estimation and compensation
# -------------------------------------------------
delay_est = estimate_delay(ak, rx_down)
log(f"Estimated symbol delay = {delay_est}")

if delay_est > 0:
    rx_aligned = rx_down[delay_est:]
    tx_aligned = ak[:len(rx_aligned)]
elif delay_est < 0:
    tx_aligned = ak[-delay_est:]
    rx_aligned = rx_down[:len(tx_aligned)]
else:
    rx_aligned = rx_down
    tx_aligned = ak[:len(rx_aligned)]

# -------------------------------------------------
# Slicer + BER
# -------------------------------------------------
ak_hat = slicer(rx_aligned, M)

use_frac     = 0.6
start        = int((1 - use_frac) * len(ak_hat))
ak_hat_cut   = ak_hat[start:]
tx_cut       = tx_aligned[start:]

n_errors = np.sum(ak_hat_cut != tx_cut)
ser_sim  = n_errors / len(ak_hat_cut)
ber_sim  = ser_sim / np.log2(M)
ber_theo = ber_mqam(EbNo_db, M)

log()
log(f"Theo BER = {ber_theo:.2e}")
log(f"Sim  SER = {ser_sim:.2e}")
log(f"Sim  BER = {ber_sim:.2e}")
log(f"Errors   = {n_errors}")

# -------------------------------------------------
# Constellation plot
# -------------------------------------------------
pts = rx_down[500:-100]
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.real(pts), np.imag(pts), 'o', markersize=2, alpha=0.5)
ax.set_xlabel('In-Phase (I)')
ax.set_ylabel('Quadrature (Q)')
ax.set_title(f'Constelación RX — {M}-QAM, Eb/N0 = {EbNo_db:.0f} dB')
if M == 4:
    LIM = 2
elif M == 16:
    LIM = 5
else:
    LIM = 9
ax.set_xlim([-LIM, LIM])
ax.set_ylim([-LIM, LIM])
ax.set_aspect('equal')
ax.grid(True)
fig.savefig(os.path.join(RESULTS_DIR, 'basic_tx_rx_01_constellation.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------
# Save text output
# -------------------------------------------------
log()
log("=== Output files saved ===")
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.startswith('basic_tx_rx'):
        log(f"  {os.path.join(RESULTS_DIR, fname)}")

with open(os.path.join(RESULTS_DIR, 'basic_tx_rx_output.txt'), 'w') as f_out:
    f_out.write('\n'.join(_log_lines) + '\n')
