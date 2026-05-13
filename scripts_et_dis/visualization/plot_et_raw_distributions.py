#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# =========================
# Configuración manual
# =========================
RAW_DIR = "../et_distributions"   # cambia aquí si quieres: et_distributions_exchange, et_distributions_inelastic, etc.
N_PLOT = 9
SEED = 42
SAVE_NAME = "qct_raw_et_distributions.png"

RAW_INDEX = os.path.join(RAW_DIR, "et_distributions_index.csv")
RAW_DIST_DIR = os.path.join(RAW_DIR, "distributions")

OUTDIR = os.path.join(RAW_DIR, "plots")
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# Helpers
# =========================
def load_distribution(filepath):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]


# =========================
# Cargar índice
# =========================
raw_df = pd.read_csv(RAW_INDEX)

if len(raw_df) == 0:
    raise RuntimeError("The index file is empty.")

rng = np.random.default_rng(SEED)
n_plot = min(N_PLOT, len(raw_df))

selected_rows = raw_df.sample(n=n_plot, random_state=SEED).reset_index(drop=True)


# =========================
# Figura
# =========================
ncols = 3
nrows = int(np.ceil(n_plot / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
axes = np.atleast_1d(axes).ravel()

for ax, (_, row) in zip(axes, selected_rows.iterrows()):

    dist_file = os.path.join(RAW_DIST_DIR, row["dist_file"])

    if not os.path.exists(dist_file):
        ax.set_title("Missing file")
        ax.axis("off")
        continue

    E_raw, P_raw = load_distribution(dist_file)

    ax.plot(E_raw, P_raw, label="QCT raw", linewidth=2)

    process = row["process"] if "process" in row.index else "unknown"

    if "n_selected" in row.index:
        counts = f"selected={int(row['n_selected'])}/{int(row['n_total'])}"
    elif "n_reactive" in row.index:
        counts = f"reactive={int(row['n_reactive'])}/{int(row['n_total'])}"
    else:
        counts = ""

    title = (
        f"id={int(row['sample_id'])} | {process}\n"
        f"E={row['E_in']} v={row['v_in']} j={row['j_in']} | {counts}"
    )

    ax.set_title(title)
    ax.set_xlabel("E_out (eV)")
    ax.set_ylabel("P(E_out)")
    ax.grid(True)
    ax.legend()

# Apagar ejes vacíos
for i in range(n_plot, len(axes)):
    axes[i].axis("off")

plt.tight_layout()

outfile = os.path.join(OUTDIR, SAVE_NAME)
plt.savefig(outfile, dpi=300)
print(f"Saved figure to: {outfile}")

plt.show()
