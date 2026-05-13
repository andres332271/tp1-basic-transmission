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
from numpy.fft import fft, fftshift

plt.close('all')

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Capture stdout to text file
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
rolloff = 0.1      # Pulse shaping rolloff
h_taps = 101       # Pulse shaping taps
M = 16              # Modulation order. QPSK (QAM4): M=4 y QAM-16: M=16

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
# Raised Cosine Filter
# -------------------------------
def raised_cosine(beta, span, sps):
    t = np.arange(-span//2, span//2 + 1) / sps
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0
        elif beta != 0 and abs(t[i]) == 1/(2*beta):
            h[i] = (np.pi/4)*np.sinc(1/(2*beta))
        else:
            h[i] = (np.sinc(t[i]) *
                    np.cos(np.pi*beta*t[i]) /
                    (1 - (2*beta*t[i])**2))
    return h/np.sum(h)

h = raised_cosine(rolloff, h_taps, N)
h_delay = (h_taps - 1) // 2

log(f"RC filter taps: {len(h)}, delay: {h_delay} samples")
log()

yup = signal.lfilter(h, 1, np.concatenate([xup, np.zeros(h_delay)]))
yup = yup[h_delay+1:]

# -------------------------------
# Time domain plot
# -------------------------------
t_symbols = np.arange(L) * T
t_samples = np.arange(len(yup)) * Ts
LPLOT = 30 * N

fig1, axes = plt.subplots(2, 1, figsize=(6, 6))

axes[0].plot(t_samples[:LPLOT], np.real(yup[:LPLOT]), '-x', label="Filtered signal")
axes[0].stem(t_symbols[:30], np.real(ak[:30]),
             linefmt='r-', markerfmt='ro', basefmt=" ", label="Tx symbols")
axes[0].set_title("TX Real Part")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_samples[:LPLOT], np.imag(yup[:LPLOT]), '-x', label="Filtered signal")
axes[1].stem(t_symbols[:30], np.imag(ak[:30]),
             linefmt='r-', markerfmt='ro', basefmt=" ", label="Tx symbols")
axes[1].set_title("TX Imag Part")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, 'basic_tx_01_time_domain.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)

# -------------------------------
# Filter Frequency Response
# -------------------------------
NFFT = 1024*8
H_abs = np.abs(fft(h, NFFT))
f = np.linspace(-fs/2, fs/2, NFFT)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(f/1e9, fftshift(20*np.log10(H_abs/np.max(H_abs))))
ax2.set_title("Filter Frequency Response (dB)")
ax2.set_xlabel("Frequency [GHz]")
ax2.grid(True)
fig2.savefig(os.path.join(RESULTS_DIR, 'basic_tx_02_filter_freq_db.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(h)
ax3.set_title("Filter Impulse Response")
ax3.set_xlabel("Samples")
ax3.grid(True)
fig3.savefig(os.path.join(RESULTS_DIR, 'basic_tx_03_filter_impulse.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

# -------------------------------
# PSD using Welch
# -------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 5))

f1, Pxx1 = signal.welch(ak, BR, nperseg=NFFT//2, return_onesided=False)
f3, Pxx3 = signal.welch(xup, fs, nperseg=NFFT//4, return_onesided=False)
f2, Pxx2 = signal.welch(yup, fs, nperseg=NFFT//4, return_onesided=False)

ax4.plot(fftshift(f1)/1e9, fftshift(10*np.log10(Pxx1/np.max(Pxx1))), label="PSD symbols")
ax4.plot(fftshift(f2)/1e9, fftshift(10*np.log10(Pxx2/np.max(Pxx2))), label="PSD filtered")
ax4.plot(fftshift(f3)/1e9, fftshift(10*np.log10(Pxx3/np.max(Pxx3))), label="PSD xup")
ax4.legend()
ax4.set_xlabel("Frequency [GHz]")
ax4.set_ylabel("PSD [dB]")
ax4.grid(True)
fig4.savefig(os.path.join(RESULTS_DIR, 'basic_tx_04_psd_welch.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)

# -------------------------------
# Eye Diagram
# -------------------------------
def eye_diagram(sig, sps, num_symbols=200, save_path=None):
    samples = sps * 2
    truncated = sig[:num_symbols*samples]
    reshaped = truncated.reshape((-1, samples))
    fig, ax = plt.subplots()
    for row in reshaped:
        ax.plot(np.real(row), alpha=0.3)
    ax.set_title("Eye Diagram")
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

eye_diagram(yup[500:5000], N,
            save_path=os.path.join(RESULTS_DIR, 'basic_tx_05_eye_diagram.png'))

# -------------------------------
# Constellation Diagram
# -------------------------------
fig6, ax6 = plt.subplots(figsize=(6, 6))
ax6.scatter(np.real(ak), np.imag(ak), s=5, alpha=0.5)
ax6.set_title("TX Constellation (16-QAM)")
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
