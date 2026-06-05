import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from numpy.fft import fft, fftshift, fftfreq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import root_raised_cosine
from tools import qammod, slicer, estimate_delay, ber_mqam, welch_psd, eye_diagram

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(msg)

# -------------------------------------------------
# Parameters (CLI o defaults)
# -------------------------------------------------
parser = argparse.ArgumentParser(description='Simulador TX-RX M-QAM con filtro RRC y canal AWGN')
parser.add_argument('--M',       type=int,   default=16,    help='Orden de modulación (4=QPSK, 16=QAM16)')
parser.add_argument('--L',       type=int,   default=10000, help='Cantidad de símbolos')
parser.add_argument('--N',       type=int,   default=2,     help='Tasa de sobremuestreo')
parser.add_argument('--rolloff', type=float, default=0.1,   help='Rolloff del filtro RRC')
parser.add_argument('--h_taps',  type=int,   default=203,   help='Cantidad de taps del filtro RRC')
parser.add_argument('--EbNo',    type=float, default=10.0,  help='Eb/N0 deseado en dB')
args = parser.parse_args()

M       = args.M
L       = args.L
N       = args.N
rolloff = args.rolloff
h_taps  = args.h_taps
EbNo_db = args.EbNo
BR      = 32e9   # fijo por enunciado

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
h_delay = (len(h) - 1) // 2

log(f"h_delay  = {h_delay} samples")

yup = lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
yup = yup[h_delay:]

# -------------------------------------------------
# Channel AWGN
# -------------------------------------------------
k_bits  = np.log2(M)
EbNo    = 10**(EbNo_db / 10)
SNR_slc = EbNo * k_bits
SNR_ch  = SNR_slc / N

Ps    = np.var(yup)
Pn    = Ps / SNR_ch
noise = np.sqrt(Pn / 2) * (np.random.randn(len(yup)) + 1j * np.random.randn(len(yup)))
rx    = yup + noise

# -------------------------------------------------
# RX: matched filter + decimation
# -------------------------------------------------
h_mf = np.conj(h[::-1])
ymf  = lfilter(h_mf, 1, np.concatenate([rx, np.zeros(h_delay)]))
ymf  = ymf[h_delay:]

PHASE   = 0   # picos en k·N tras compensación de delay en TX y MF
rx_down = ymf[PHASE::N]

# -------------------------------------------------
# Normalización antes del slicer
# La ganancia del cascade RRC→MF en el instante de muestreo es N·Σh²:
# N del upsampling, Σh² de la autocorrelación del filtro en lag cero.
# -------------------------------------------------
norm_factor = N * np.sum(h**2)
rx_norm     = rx_down / norm_factor

log(f"Norm factor = {norm_factor:.4f}  (N·Σh² = {N}·{np.sum(h**2):.4f})")

# -------------------------------------------------
# Delay estimation and compensation
# -------------------------------------------------
delay_est = estimate_delay(ak, rx_norm)
log(f"Estimated symbol delay = {delay_est}")

if delay_est > 0:
    rx_aligned = rx_norm[delay_est:]
    tx_aligned = ak[:len(rx_aligned)]
elif delay_est < 0:
    tx_aligned = ak[-delay_est:]
    rx_aligned = rx_norm[:len(tx_aligned)]
else:
    rx_aligned = rx_norm
    tx_aligned = ak[:len(rx_aligned)]

# -------------------------------------------------
# Slicer + BER
# -------------------------------------------------
ak_hat = slicer(rx_aligned, M)

use_frac   = 0.6
start      = int((1 - use_frac) * len(ak_hat))
n_errors   = np.sum(ak_hat[start:] != tx_aligned[start:])
ser_sim    = n_errors / len(ak_hat[start:])
ber_sim    = ser_sim / np.log2(M)
ber_theo   = ber_mqam(EbNo_db, M)

log()
log(f"Theo BER = {ber_theo:.2e}")
log(f"Sim  SER = {ser_sim:.2e}")
log(f"Sim  BER = {ber_sim:.2e}")
log(f"Errors   = {n_errors}")

# -------------------------------------------------
# PSD helpers (compartidos por plots 02 y 03)
# -------------------------------------------------
NFFT   = 1024 * 8
f_nyq  = BR / 2 / 1e9
f_bw   = BR / 2 * (1 + rolloff) / 1e9

_, Pxx_xup = welch_psd(xup, fs, NFFT // 4)
psd_norm   = np.mean(Pxx_xup)   # referencia 0 dB: espectro plano de la entrada blanca

def bw_lines(ax):
    for sign in (+1, -1):
        ax.axvline(sign * f_nyq, color='steelblue', linestyle='--', linewidth=1.6,
                   label=f'±BR/2 = ±{f_nyq:.0f} GHz' if sign == 1 else None)
        ax.axvline(sign * f_bw,  color='tomato',    linestyle='--', linewidth=1.6,
                   label=f'±BR/2·(1+β) = ±{f_bw:.0f} GHz' if sign == 1 else None)

m_qam      = int(np.sqrt(M))
sym_levels = sorted({2*(k % m_qam) - m_qam + 1 for k in range(M)})

# -------------------------------------------------
# Plot 01: Diagrama de ojo TX
# -------------------------------------------------
eye_diagram(yup[500:], N,
            title=f"Diagrama de ojo TX — RRC (β={rolloff}, {M}-QAM)",
            levels=sym_levels,
            save_path=os.path.join(RESULTS_DIR, 'basic_tx_rx_01_eye_diagram_tx.png'))

# -------------------------------------------------
# Plot 02: PSD TX vs PSD entrada MF
# -------------------------------------------------
f_tx,  Pxx_tx  = welch_psd(yup, fs, NFFT // 4)
f_mfi, Pxx_mfi = welch_psd(rx,  fs, NFFT // 4)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(f_tx/1e9,  10*np.log10(Pxx_tx  / psd_norm), label='PSD TX (salida transmisor)')
ax.plot(f_mfi/1e9, 10*np.log10(Pxx_mfi / psd_norm), label='PSD entrada MF (TX + ruido AWGN)')
bw_lines(ax)
ax.set_xlabel('Frecuencia [GHz]')
ax.set_ylabel('PSD [dB]')
ax.set_title(f'PSD TX vs entrada MF — RRC (β={rolloff}, Eb/N0={EbNo_db:.0f} dB)')
ax.set_xlim(-BR/1e9, BR/1e9)
ax.set_ylim(-100, 5)
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'basic_tx_rx_02_psd_tx_vs_mf_in.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------
# Plot 03: PSD entrada MF vs PSD salida MF
# -------------------------------------------------
f_mfo, Pxx_mfo = welch_psd(ymf, fs, NFFT // 4)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(f_mfi/1e9, 10*np.log10(Pxx_mfi / psd_norm), label='PSD entrada MF')
ax.plot(f_mfo/1e9, 10*np.log10(Pxx_mfo / psd_norm), label='PSD salida MF (antes de decimar)')
bw_lines(ax)
ax.set_xlabel('Frecuencia [GHz]')
ax.set_ylabel('PSD [dB]')
ax.set_title(f'PSD entrada vs salida MF — RRC (β={rolloff}, Eb/N0={EbNo_db:.0f} dB)')
ax.set_xlim(-BR/1e9, BR/1e9)
ax.set_ylim(-100, 5)
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'basic_tx_rx_03_psd_mf_in_vs_mf_out.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------
# Plot 04: Constelación RX a la entrada del slicer
# -------------------------------------------------
pts = rx_norm[500:-100]
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
fig.savefig(os.path.join(RESULTS_DIR, 'basic_tx_rx_04_constellation.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------
# Plot 05: Histogramas I/Q a la entrada del slicer
# -------------------------------------------------
pts_hist = rx_norm[500:-100]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, part, label in zip(axes, [np.real, np.imag], ['In-Phase (I)', 'Quadrature (Q)']):
    ax.hist(part(pts_hist), bins=80, density=True,
            color='steelblue', alpha=0.8, edgecolor='none')
    for lvl in sym_levels:
        ax.axvline(lvl, color='tomato', linestyle='--', linewidth=1.2,
                   label='Nivel de símbolo' if lvl == sym_levels[0] else None)
    ax.set_xlabel(label)
    ax.set_ylabel('Densidad')
    ax.set_title(f'Histograma {label} — {M}-QAM, Eb/N0={EbNo_db:.0f} dB')
    ax.legend(fontsize=8)
    ax.grid(True)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'basic_tx_rx_05_histograms_iq.png'), dpi=150, bbox_inches='tight')
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
