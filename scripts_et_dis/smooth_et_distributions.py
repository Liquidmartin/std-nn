#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


# =========================
# Leer archivo de configuración
# =========================
def read_conditions(filename="conditions_smooth.txt"):
    params = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line == "" or line.startswith("#"):
                continue

            key, value = line.split()
            params[key] = value

    params["sigma_ev"] = float(params["sigma_ev"])

    return params


# =========================
# Gaussian smoothing
# =========================
def gaussian_kernel_1d(sigma_bins, radius=None):
    if sigma_bins <= 0:
        return np.array([1.0])

    if radius is None:
        radius = int(np.ceil(4.0 * sigma_bins))

    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_probability(P, sigma_bins):
    kernel = gaussian_kernel_1d(sigma_bins)
    P_smooth = np.convolve(P, kernel, mode="same")

    P_smooth = np.clip(P_smooth, 0.0, None)

    s = P_smooth.sum()
    if s > 0:
        P_smooth /= s

    return P_smooth


# =========================
# Main
# =========================
def main():
    params = read_conditions()

    INPUT_DIR = params["input_dir"]
    OUTPUT_DIR = params["output_dir"]
    SIGMA_EV = params["sigma_ev"]

    INPUT_INDEX = os.path.join(INPUT_DIR, "et_distributions_index.csv")
    INPUT_DIST_DIR = os.path.join(INPUT_DIR, "distributions")

    OUTPUT_DIST_DIR = os.path.join(OUTPUT_DIR, "distributions")
    OUTPUT_INDEX = os.path.join(OUTPUT_DIR, "et_distributions_smoothed_index.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIST_DIR, exist_ok=True)

    index_df = pd.read_csv(INPUT_INDEX)

    new_rows = []

    for _, row in index_df.iterrows():
        infile = os.path.join(INPUT_DIST_DIR, row["dist_file"])
        outfile = os.path.join(OUTPUT_DIST_DIR, row["dist_file"])

        if not os.path.exists(infile):
            print(f"Missing file: {infile}")
            continue

        data = np.loadtxt(infile)
        E = data[:, 0]
        P = data[:, 1]

        if len(E) < 2:
            continue

        dE = E[1] - E[0]
        sigma_bins = SIGMA_EV / dE

        P_smooth = smooth_probability(P, sigma_bins)

        np.savetxt(
            outfile,
            np.column_stack([E, P_smooth]),
            header="E_out  P_smooth",
            fmt="%.6f %.12e"
        )

        new_row = row.to_dict()
        new_row["smoothed_file"] = row["dist_file"]
        new_row["sigma_ev"] = SIGMA_EV
        new_row["sum_P_smooth"] = float(P_smooth.sum())
        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(OUTPUT_INDEX, index=False)

    print("DONE.")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Sigma:      {SIGMA_EV} eV")
    print(f"Saved smoothed distributions in: {OUTPUT_DIST_DIR}")
    print(f"Saved index: {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
