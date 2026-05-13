#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Compare QCT raw and QCT smooth ET distributions"
)

parser.add_argument("--n", type=int, default=9, help="Number of samples to plot")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--save-name",
    default="compare_qct_raw_vs_smooth.png",
    help="Output figure name"
)

args = parser.parse_args()


# =========================
# Paths
# =========================
RAW_DIR = "../et_distributions_exchange"
SMOOTH_DIR = "../et_distributions_exchange_smoothed"

RAW_INDEX = os.path.join(RAW_DIR, "et_distributions_index.csv")
RAW_DIST_DIR = os.path.join(RAW_DIR, "distributions")

SMOOTH_INDEX = os.path.join(SMOOTH_DIR, "et_distributions_smoothed_index.csv")
SMOOTH_DIST_DIR = os.path.join(SMOOTH_DIR, "distributions")

OUTDIR = os.path.join(SMOOTH_DIR, "comparison_plots")
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# Helpers
# =========================
def load_distribution(filepath):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]


def get_row_by_sample_id(df, sid):
    sub = df[df["sample_id"] == sid]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


# =========================
# Load metadata
# =========================
print("RAW_INDEX:", RAW_INDEX)
print("SMOOTH_INDEX:", SMOOTH_INDEX)

if not os.path.exists(RAW_INDEX):
    raise FileNotFoundError(f"Raw index not found: {RAW_INDEX}")

if not os.path.exists(SMOOTH_INDEX):
    raise FileNotFoundError(f"Smooth index not found: {SMOOTH_INDEX}")

raw_df = pd.read_csv(RAW_INDEX)
smooth_df = pd.read_csv(SMOOTH_INDEX)


# =========================
# Common sample IDs
# =========================
common_ids = sorted(set(raw_df["sample_id"]) & set(smooth_df["sample_id"]))

if len(common_ids) == 0:
    raise RuntimeError("No common sample_id found between raw and smooth distributions.")

rng = np.random.default_rng(args.seed)
n_plot = min(args.n, len(common_ids))
selected_ids = rng.choice(common_ids, size=n_plot, replace=False)


# =========================
# Figure
# =========================
ncols = 3
nrows = int(np.ceil(n_plot / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
axes = np.atleast_1d(axes).ravel()

for ax, sid in zip(axes, selected_ids):
    sid = int(sid)

    row_raw = get_row_by_sample_id(raw_df, sid)
    row_smooth = get_row_by_sample_id(smooth_df, sid)

    if row_raw is None or row_smooth is None:
        ax.axis("off")
        continue

    raw_file = os.path.join(RAW_DIST_DIR, row_raw["dist_file"])

    smooth_name = (
        row_smooth["smoothed_file"]
        if "smoothed_file" in row_smooth.index
        else row_smooth["dist_file"]
    )

    smooth_file = os.path.join(SMOOTH_DIST_DIR, smooth_name)

    if not os.path.exists(raw_file):
        ax.set_title(f"Missing raw file id={sid}")
        ax.axis("off")
        continue

    if not os.path.exists(smooth_file):
        ax.set_title(f"Missing smooth file id={sid}")
        ax.axis("off")
        continue

    E_raw, P_raw = load_distribution(raw_file)
    E_smooth, P_smooth = load_distribution(smooth_file)

    ax.plot(E_raw, P_raw, label="QCT raw", linewidth=2)
    ax.plot(E_smooth, P_smooth, label="QCT smooth", linewidth=2, linestyle="--")

    process = row_raw["process"] if "process" in row_raw.index else "unknown"

    if "n_selected" in row_raw.index:
        counts = f"selected={int(row_raw['n_selected'])}/{int(row_raw['n_total'])}"
    elif "n_reactive" in row_raw.index:
        counts = f"reactive={int(row_raw['n_reactive'])}/{int(row_raw['n_total'])}"
    else:
        counts = ""

    title = (
        f"id={sid} | {process}\n"
        f"E={row_raw['E_in']} v={row_raw['v_in']} j={row_raw['j_in']} | {counts}"
    )

    ax.set_title(title)
    ax.set_xlabel("E_out (eV)")
    ax.set_ylabel("P(E_out)")
    ax.grid(True)
    ax.legend()


# Hide empty axes
for i in range(len(selected_ids), len(axes)):
    axes[i].axis("off")


plt.tight_layout()

outfile = os.path.join(OUTDIR, args.save_name)
plt.savefig(outfile, dpi=300)

print(f"Saved figure to: {outfile}")

plt.show()
