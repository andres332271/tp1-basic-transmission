import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import ber_mqam

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# Curvas teóricas de BER
# -------------------------------------------------
EbN0_dB = np.arange(0, 21, 0.5)

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(EbN0_dB, ber_mqam(EbN0_dB, 4),  '--', label='QPSK (teórico)')
ax.semilogy(EbN0_dB, ber_mqam(EbN0_dB, 16), '--', label='16-QAM (teórico)')

ax.set_ylim([1e-6, 5e-2])
ax.set_xlabel('Eb/N0 [dB]')
ax.set_ylabel('BER')
ax.set_title('BER vs Eb/N0 — Curvas teóricas')
ax.legend()
ax.grid(True, which='both')

fig.savefig(os.path.join(RESULTS_DIR, 'theo_curves_01_ber.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
