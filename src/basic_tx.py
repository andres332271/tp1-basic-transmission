# -----------------------------------------------------------------------------
#                                   UBA
# Programmer(s): Francisco G. Rainero
# -----------------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift, fftfreq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import raised_cosine, root_raised_cosine

plt.close('all')

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(msg)

# -------------------------------
# Basic TX M-QAM
# -------------------------------
L = 10000          # Simulation Length
BR = 32e9          # Baud Rate
N = 4              # Oversampling rate
rolloff = 0.5      # Pulse shaping rolloff
h_taps = 101       # Pulse shaping taps
M = 16             # Modulation order. QPSK (QAM4): M=4, QAM-16: M=16

fs = N * BR        # Sampling rate
T = 1 / BR         # Symbol period
Ts = 1 / fs        # Sample period

log("=== Basic TX M-QAM Parameters ===")
log(f"L        = {L} symbols")
log(f"BR       = {BR/1e9:.1f} GBd")
log(f"N        = {N} (oversampling rate)")
log(f"rolloff  = {rolloff}")
log(f"h_taps   = {h_taps}")
log(f"M        = {M}-QAM")
log(f"fs       = {fs/1e9:.1f} GHz")
log(f"T        = {T*1e12:.3f} ps")
log(f"Ts       = {Ts*1e12:.3f} ps")
log()

# -------------------------------
# QAM MODULATION (Gray mapping)
# -------------------------------
x_aux = np.random.randint(0, M, L)

def qammod(symbols, M):
    m = int(np.sqrt(M))
    re = 2*(symbols % m) - m + 1
    im = 2*(symbols // m) - m + 1
    return (re + 1j*im)

ak = qammod(x_aux, M)

log(f"Symbols generated: {len(ak)}")
log(f"Symbol power (mean |ak|^2): {np.mean(np.abs(ak)**2):.4f}")
log()

# -------------------------------
# Upsampling
# -------------------------------
xup = np.zeros(L*N, dtype=complex)
xup[::N] = ak * N

# -------------------------------
# Filters
# -------------------------------
h_rc  = raised_cosine(BR/2, fs, rolloff, h_taps)
h_rrc = root_raised_cosine(BR/2, fs, rolloff, h_taps)
h_delay = (len(h_rc) - 1) // 2

log(f"RC  filter: {len(h_rc)} taps, delay {h_delay} samples")
log(f"RRC filter: {len(h_rrc)} taps, delay {h_delay} samples")
log()

def apply_filter(h, xup, h_delay):
    y = signal.lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
    return y[h_delay:]

yup_rc  = apply_filter(h_rc,  xup, h_delay)
yup_rrc = apply_filter(h_rrc, xup, h_delay)

# -------------------------------
# Point 1 & 2: Time domain plots
# s(t) via plot(), symbols via stem()
# -------------------------------
t_symbols = np.arange(30) * T * BR       # normalized: t·BR
t_samples = np.arange(30 * N) * Ts * BR  # normalized: t·BR

def save_time_domain(yup, filter_name, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, part, label in zip(axes,
                                [np.real, np.imag],
                                ['Real', 'Imag']):
        ax.plot(t_samples, part(yup[:30*N]), '-x', label=f"s(t) — {filter_name}")
        ax.stem(t_symbols, part(ak[:30]),
                linefmt='r-', markerfmt='ro', basefmt=" ", label="Símbolos TX")
        ax.set_xlabel("Tiempo normalizado (t · BR)")
        ax.set_title(f"TX {label} — {filter_name} (β={rolloff})")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

# Point 1: RC
save_time_domain(yup_rc,  f"RC",  'basic_tx_01_time_domain_rc.png')
# Point 2: RRC
save_time_domain(yup_rrc, f"RRC", 'basic_tx_02_time_domain_rrc.png')

# -------------------------------
# PSD helpers (normalización a 0 dB en f=0)
# -------------------------------
NFFT = 1024 * 8

def welch_normalized_dB(sig, sample_rate, nperseg):
    f_w, Pxx = signal.welch(sig, sample_rate, nperseg=nperseg, return_onesided=False)
    f_w = fftshift(f_w)
    Pxx = fftshift(Pxx)
    f0_idx = np.argmin(np.abs(f_w))
    Pxx_dB = 10 * np.log10(Pxx / Pxx[f0_idx])
    return f_w, Pxx_dB

def h_squared_dB(h, n_fft, sample_rate):
    H = np.abs(fft(h, n_fft))
    H_sq = H**2
    # DC en índice 0, normalizado por sum(h)=H(0)=1, así que H_sq[0]=1
    H_sq_dB = 10 * np.log10(np.maximum(H_sq / H_sq[0], 1e-10))
    f_h = fftshift(fftfreq(n_fft, 1/sample_rate))
    return f_h, fftshift(H_sq_dB)

# -------------------------------
# Point 3: PSD a la entrada del filtro (xup)
# -------------------------------
f_in, Pxx_in_dB = welch_normalized_dB(xup, fs, nperseg=NFFT//4)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(f_in/1e9, Pxx_in_dB)
ax3.set_xlabel("Frecuencia [GHz]")
ax3.set_ylabel("PSD [dB]")
ax3.set_title("PSD a la entrada del filtro — método de Welch")
ax3.set_xlim(-BR/1e9, BR/1e9)
ax3.set_ylim(-20, 5)
ax3.grid(True)
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, 'basic_tx_03_psd_input.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

# -------------------------------
# Point 4: PSD a la salida del filtro + |H(f)|²  (RC y RRC)
# -------------------------------
for h, yup, fname_suffix, filter_name in [
    (h_rc,  yup_rc,  'rc',  'RC'),
    (h_rrc, yup_rrc, 'rrc', 'RRC'),
]:
    f_out, Pxx_out_dB = welch_normalized_dB(yup, fs, nperseg=NFFT//4)
    f_h,   H_sq_dB   = h_squared_dB(h, NFFT, fs)

    f_nyq = BR / 2 / 1e9
    f_bw  = BR / 2 * (1 + rolloff) / 1e9

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_out/1e9, Pxx_out_dB,     label=f"PSD salida filtro {filter_name}")
    ax.plot(f_h/1e9,   H_sq_dB, '--',  label=f"|H_{filter_name}(f)|²")
    for sign in (+1, -1):
        ax.axvline(sign * f_nyq, color='steelblue', linestyle='--', linewidth=1.6,
                   label=f'±BR/2 = ±{f_nyq:.0f} GHz' if sign == 1 else None)
        ax.axvline(sign * f_bw,  color='tomato',    linestyle='--', linewidth=1.6,
                   label=f'±BR/2·(1+β) = ±{f_bw:.0f} GHz' if sign == 1 else None)
    ax.set_xlabel("Frecuencia [GHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(f"PSD salida vs |H(f)|² — {filter_name} (β={rolloff})")
    ax.set_xlim(-BR/1e9, BR/1e9)
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f'basic_tx_04_psd_output_vs_h2_{fname_suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

# -------------------------------
# Eye Diagram (RC)
# -------------------------------
def eye_diagram(sig, sps, num_symbols=200, save_path=None):
    samples = sps * 2
    truncated = sig[:num_symbols*samples]
    reshaped = truncated.reshape((-1, samples))
    fig, ax = plt.subplots()
    for row in reshaped:
        ax.plot(np.real(row), alpha=0.3, color='steelblue')
    ax.set_title(f"Diagrama de ojo — RC (β={rolloff})")
    ax.set_xlabel("Muestra")
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

eye_diagram(yup_rc[500:5000], N,
            save_path=os.path.join(RESULTS_DIR, 'basic_tx_05_eye_diagram.png'))

# -------------------------------
# Constellation TX
# -------------------------------
fig6, ax6 = plt.subplots(figsize=(6, 6))
ax6.scatter(np.real(ak), np.imag(ak), s=5, alpha=0.5)
ax6.set_title(f"Constelación TX ({M}-QAM)")
ax6.set_xlabel("In-Phase (I)")
ax6.set_ylabel("Quadrature (Q)")
ax6.grid(True)
ax6.axis("equal")
fig6.savefig(os.path.join(RESULTS_DIR, 'basic_tx_06_constellation.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)

# -------------------------------
# Save text output
# -------------------------------
log("=== Output files saved ===")
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.startswith('basic_tx'):
        log(f"  {os.path.join(RESULTS_DIR, fname)}")

with open(os.path.join(RESULTS_DIR, 'basic_tx_output.txt'), 'w') as f_out:
    f_out.write('\n'.join(_log_lines) + '\n')
