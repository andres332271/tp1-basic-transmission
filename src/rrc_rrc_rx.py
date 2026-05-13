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
h_taps = 101
ROLLOFF_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]

fs = N * BR
Ts = 1 / fs

log("=== RRC Filter Parameters ===")
log(f"BR       = {BR/1e9:.1f} GBd")
log(f"N        = {N}")
log(f"rolloffs = {ROLLOFF_VALUES}")
log(f"h_taps   = {h_taps}")
log(f"fs       = {fs/1e9:.1f} GHz")
log(f"Ts       = {Ts*1e12:.3f} ps")
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

    h_v = np.sinc(tn_v) * np.cos(np.pi * rolloff * tn_v) \
          / (1 - (2 * rolloff * tn_v)**2)

    h_v = h_v / np.sum(h_v)

    return h_v

# -------------------------------------------------
# Root Raised Cosine
# -------------------------------------------------

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

    denominator = (
        np.pi * tn_v * (1 - (4 * rolloff * tn_v)**2)
    )

    h_v = numerator / denominator

    # Valor central (evita NaN en tn_v = 0)
    center = (n_taps - 1) // 2
    h_v[center] = (1 + rolloff * (4/np.pi - 1))

    h_v = h_v / np.sum(h_v)

    return h_v

# -------------------------------------------------
# Eje temporal normalizado (común a todos los filtros)
# -------------------------------------------------

n_taps = round_odd(h_taps)
n_v = np.arange(-(n_taps - 1)//2, (n_taps - 1)//2 + 1)
t_norm_v = n_v * Ts * BR     # tiempo normalizado a T = 1/BR

NFFT = 2048
f = np.arange(-NFFT/2, NFFT/2) * fs / NFFT

# -------------------------------------------------
# PLOTS
# -------------------------------------------------

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig1, ax1 = plt.subplots(figsize=(7, 4))
fig2, ax2 = plt.subplots(figsize=(7, 4))
fig3, ax3 = plt.subplots(figsize=(7, 4))

for color, beta in zip(colors, ROLLOFF_VALUES):
    h_rrc = root_raised_cosine(BR/2, fs, beta, h_taps)
    H_RRC = fftshift(np.abs(fft(h_rrc, NFFT)))

    label = f'β = {beta}'

    ax1.plot(t_norm_v, h_rrc, color=color, linewidth=1.5, label=label)

    ax2.plot(f/1e9, H_RRC, color=color, linewidth=1.5, label=label)

    H_RRC_dB = 20 * np.log10(np.maximum(H_RRC, 1e-10))
    ax3.plot(f/1e9, H_RRC_dB, color=color, linewidth=1.5, label=label)

    log(f"RRC (β={beta}): {n_taps} taps")

log()

# Figura 1 — respuesta al impulso
ax1.axhline(0, color='k', linewidth=0.5)
ax1.set_title('Respuesta al impulso — Root Raised Cosine')
ax1.set_xlabel('Tiempo normalizado (t · BR)')
ax1.set_ylabel('Amplitud')
ax1.grid(True)
ax1.legend()
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, 'rrc_01_impulse_response.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)

# Figura 2 — respuesta en frecuencia lineal
ax2.axvline(BR/2/1e9, linestyle='--', color='k', linewidth=0.8, label=f'BR/2 = {BR/2/1e9:.0f} GHz')
ax2.set_title('Respuesta en frecuencia — Root Raised Cosine')
ax2.set_xlabel('Frecuencia [GHz]')
ax2.set_ylabel('Amplitud')
ax2.set_xlim(-BR/1e9, BR/1e9)
ax2.grid(True)
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'rrc_02_freq_response.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)

# Figura 3 — respuesta en frecuencia en dB
ax3.axvline(BR/2/1e9, linestyle='--', color='k', linewidth=0.8, label=f'BR/2 = {BR/2/1e9:.0f} GHz')
ax3.set_title('Respuesta en frecuencia [dB] — Root Raised Cosine')
ax3.set_xlabel('Frecuencia [GHz]')
ax3.set_ylabel('Amplitud [dB]')
ax3.set_xlim(-BR/1e9, BR/1e9)
ax3.set_ylim(-80, 5)
ax3.grid(True)
ax3.legend()
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, 'rrc_03_freq_response_db.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

# -------------------------------------------------
# Verificación RRC * RRC = RC
# -------------------------------------------------

# Eje temporal para el filtro convolucionado (2*h_taps - 1 taps)
n_conv = 2 * n_taps - 1
n_conv_v = np.arange(-(n_conv - 1)//2, (n_conv - 1)//2 + 1)
t_norm_conv = n_conv_v * Ts * BR

fig4, ax4 = plt.subplots(figsize=(7, 4))
fig5, ax5 = plt.subplots(figsize=(7, 4))

log("=== Verificación RRC*RRC = RC ===")

for color, beta in zip(colors, ROLLOFF_VALUES):
    h_rrc = root_raised_cosine(BR/2, fs, beta, h_taps)
    h_rrc_rrc = np.convolve(h_rrc, h_rrc)           # 201 taps
    h_rc = raised_cosine(BR/2, fs, beta, n_conv)     # 201 taps

    # --- Dominio temporal ---
    diff_time = h_rrc_rrc - h_rc
    max_diff = np.max(np.abs(diff_time))
    log(f"β={beta:4.2f}:  max|RRC*RRC - RC| = {max_diff:.4e}")

    ax4.plot(t_norm_conv, diff_time, color=color, linewidth=1.5, label=f'β = {beta}')

    # --- Dominio frecuencial ---
    H_RRC = fftshift(np.abs(fft(h_rrc, NFFT)))
    H_RC_f = fftshift(np.abs(fft(h_rc, NFFT)))
    freq_error_dB = 20 * np.log10(np.maximum(np.abs(H_RRC**2 - H_RC_f), 1e-10))

    ax5.plot(f/1e9, freq_error_dB, color=color, linewidth=1.5, label=f'β = {beta}')

log()

# Figura 4 — error en tiempo
ax4.axhline(0, color='k', linewidth=0.5)
ax4.set_title('Error temporal: RRC*RRC − RC')
ax4.set_xlabel('Tiempo normalizado (t · BR)')
ax4.set_ylabel('Error de amplitud')
ax4.grid(True)
ax4.legend()
fig4.tight_layout()
fig4.savefig(os.path.join(RESULTS_DIR, 'rrc_04_verification_time.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)

# Figura 5 — error en frecuencia
ax5.axvline(BR/2/1e9, linestyle='--', color='k', linewidth=0.8, label=f'BR/2 = {BR/2/1e9:.0f} GHz')
ax5.set_title('Error frecuencial: |H_RRC² − H_RC| [dB]')
ax5.set_xlabel('Frecuencia [GHz]')
ax5.set_ylabel('Error [dB]')
ax5.set_xlim(-BR/1e9, BR/1e9)
ax5.grid(True)
ax5.legend()
fig5.tight_layout()
fig5.savefig(os.path.join(RESULTS_DIR, 'rrc_05_verification_freq.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)

# -------------------------------------------------
# Save text output
# -------------------------------------------------

log("=== Output files saved ===")
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.startswith('rrc_'):
        log(f"  {os.path.join(RESULTS_DIR, fname)}")

with open(os.path.join(RESULTS_DIR, 'rrc_output.txt'), 'w') as f_out:
    f_out.write('\n'.join(_log_lines) + '\n')
