import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import ber_mqam, simulate_txrx

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# Parámetros de simulación para el barrido
# -------------------------------------------------
N       = 2
rolloff = 0.1
h_taps  = 203
L_sweep = 200_000   # suficiente para ~20+ errores en BER~1e-4

# Puntos de Eb/N0 elegidos para cubrir BER ∈ [~1e-5, ~5e-2] por modulación
SWEEP = {
    4:  [0, 2, 4, 6, 8, 10],     # QPSK   — BER aprox [7e-2 → 4e-6]
    16: [6, 8, 10, 12, 14, 16],   # 16-QAM — BER aprox [5e-2 → 2e-6]
}

colors = {4: '#1f77b4', 16: '#ff7f0e'}
labels = {4: 'QPSK', 16: '16-QAM'}

# -------------------------------------------------
# Barrido de simulación
# -------------------------------------------------
sim_results = {}
for M_val, ebno_points in SWEEP.items():
    print(f"\n[{labels[M_val]}] Barrido Eb/N0: {ebno_points} dB  (L={L_sweep:,})")
    pts = []
    for ebn0 in ebno_points:
        res = simulate_txrx(M=M_val, L=L_sweep, N=N,
                            rolloff=rolloff, h_taps=h_taps, EbNo_db=ebn0)
        ber = res['ber_sim']
        print(f"  Eb/N0 = {ebn0:5.1f} dB  →  BER = {ber:.2e}  ({res['n_errors']} errores)")
        if ber > 0:
            pts.append((ebn0, ber))
    sim_results[M_val] = pts

# -------------------------------------------------
# Plot: curvas teóricas + puntos simulados
# -------------------------------------------------
EbN0_th = np.arange(0, 21, 0.25)

fig, ax = plt.subplots(figsize=(9, 6))

for M_val in (4, 16):
    c     = colors[M_val]
    label = labels[M_val]

    # Curva teórica
    ax.semilogy(EbN0_th, ber_mqam(EbN0_th, M_val),
                color=c, linestyle='--', linewidth=1.8, label=f'{label} (teórico)')

    # Puntos simulados
    if sim_results[M_val]:
        xs, ys = zip(*sim_results[M_val])
        ax.semilogy(xs, ys, 'o', color=c, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=f'{label} (simulado)')

ax.set_xlim(0, 20)
ax.set_ylim(1e-6, 5e-1)
ax.set_xlabel('Eb/N0 [dB]')
ax.set_ylabel('BER')
ax.set_title('BER vs Eb/N0 — Curvas teóricas y simulación')
ax.legend()
ax.grid(True, which='both')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'theo_curves_01_ber.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nFigura guardada: {os.path.join(RESULTS_DIR, 'theo_curves_01_ber.png')}")
