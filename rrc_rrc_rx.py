import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

plt.close("all")

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(msg)

# -------------------------------------------------
# Parámetros
# -------------------------------------------------

BR = 32e9
N = 4
rolloff = 0.1
h_taps = 101

fs = N * BR
Ts = 1 / fs

log("=== RRC Filter Parameters ===")
log(f"BR      = {BR/1e9:.1f} GBd")
log(f"N       = {N}")
log(f"rolloff = {rolloff}")
log(f"h_taps  = {h_taps}")
log(f"fs      = {fs/1e9:.1f} GHz")
log(f"Ts      = {Ts*1e12:.3f} ps")
log()

# -------------------------------------------------
# Función round_odd
# -------------------------------------------------

def round_odd(n):
    n = int(np.round(n))
    if n % 2 == 0:
        n += 1
    return n

# -------------------------------------------------
# Raised Cosine
# -------------------------------------------------

def raised_cosine(fc, fs, rolloff, n_taps, t0=0):

    rolloff = rolloff + 0.0001
    Ts = 1 / fs
    T = 1 / fc

    n_taps = round_odd(n_taps)

    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    # np.sinc ya es sinc normalizado
    h_v = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v) \
          / (1 - (2 * rolloff * tn_v)**2)

    h_v = h_v / np.sum(h_v)

    return h_v

# -------------------------------------------------
# Root Raised Cosine = sqrt(raised cosine en freq)
# -------------------------------------------------

def root_raised_cosine(fc, fs, rolloff, n_taps, t0=0):

    Ts = 1 / fs
    rolloff = rolloff + 0.0001
    T = 1 / fc

    # Force to odd
    n_taps = round_odd(n_taps)

    # Time vector
    n = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
    t_v = n * Ts + t0
    tn_v = t_v * 2 / T

    # Filter taps
    numerator = (
        np.sin(np.pi * (1 - rolloff) * tn_v)
        + 4 * rolloff * tn_v * np.cos(np.pi * (1 + rolloff) * tn_v)
    )

    denominator = (
        np.pi * tn_v * (1 - (4 * rolloff * tn_v)**2)
    )

    h_v = numerator / denominator

    # Valor central (evita NaN en tn_v = 0)
    center = (n_taps - 1) // 2
    h_v[center] = (1 + rolloff * (4/np.pi - 1))

    # Normalización
    h_v = h_v / np.sum(h_v)

    return h_v

# -------------------------------------------------
# Generación filtros
# -------------------------------------------------

h_rrc = root_raised_cosine(BR/2, fs, rolloff, h_taps)
h_rrc_rrc = np.convolve(h_rrc, h_rrc)
h_rc = raised_cosine(BR/2, fs, rolloff, len(h_rrc_rrc))

log(f"RRC filter length:     {len(h_rrc)} taps")
log(f"RRC*RRC filter length: {len(h_rrc_rrc)} taps")
log(f"RC filter length:      {len(h_rc)} taps")

# Verify RRC*RRC ≈ RC
max_diff = np.max(np.abs(h_rrc_rrc - h_rc))
log(f"Max difference |RRC*RRC - RC|: {max_diff:.6e}")
log()

# -------------------------------------------------
# FFTs
# -------------------------------------------------

NFFT = 2048
f = np.arange(-NFFT/2, NFFT/2) * fs / NFFT

H_RRC = fftshift(np.abs(fft(h_rrc, NFFT)))
H_RRC_RRC = H_RRC * H_RRC
H_RC = fftshift(np.abs(fft(h_rc, NFFT)))

# -------------------------------------------------
# PLOTS
# -------------------------------------------------

n_rrc_v = np.arange(-(h_taps-1)//2, (h_taps-1)//2 + 1)
n_rc_v = np.arange(-(len(h_rrc_rrc)-1)//2, (len(h_rrc_rrc)-1)//2 + 1)

fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.plot(n_rrc_v, h_rrc, '--k', linewidth=1)
ax1.plot(n_rc_v, h_rrc_rrc, 'r', linewidth=2)
ax1.plot(n_rc_v, h_rc, '--b', linewidth=1.5)
ax1.set_title('hrrc*hrrc = hrc')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend(['hrrc', 'hrrc*hrrc', 'hrc'])
fig1.savefig(os.path.join(RESULTS_DIR, 'rrc_01_impulse_response.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(f/1e9, H_RRC, '--k', linewidth=1)
ax2.plot(f/1e9, H_RRC_RRC, 'r', linewidth=2)
ax2.plot(f/1e9, H_RC, '--b', linewidth=1.5)
ax2.axvline(BR/2/1e9, linestyle='--', color='k')
ax2.set_title('Hrrc.Hrrc = Hrc')
ax2.set_xlabel('Freq [GHz]')
ax2.set_ylabel('Amplitude')
ax2.grid(True)
ax2.legend(['Hrrc', 'Hrrc.Hrrc', 'Hrc'])
fig2.savefig(os.path.join(RESULTS_DIR, 'rrc_02_freq_response.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(5, 5))
ax3.plot(f/1e9, 20*np.log10(H_RRC), '--k', linewidth=1)
ax3.plot(f/1e9, 20*np.log10(H_RRC_RRC), 'r', linewidth=2)
ax3.plot(f/1e9, 20*np.log10(H_RC), '--b', linewidth=1.5)
ax3.axvline(BR/2/1e9, linestyle='--', color='k')
ax3.set_title('Hrrc.Hrrc = Hrc')
ax3.set_xlabel('Freq [GHz]')
ax3.set_ylabel('Amplitude [dB]')
ax3.grid(True)
ax3.legend(['Hrrc', 'Hrrc.Hrrc', 'Hrc'])
fig3.savefig(os.path.join(RESULTS_DIR, 'rrc_03_freq_response_db.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

# -------------------------------------------------
# Save text output
# -------------------------------------------------

log("=== Output files saved ===")
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.startswith('rrc_'):
        log(f"  {os.path.join(RESULTS_DIR, fname)}")

with open(os.path.join(RESULTS_DIR, 'rrc_output.txt'), 'w') as f_out:
    f_out.write('\n'.join(_log_lines) + '\n')
