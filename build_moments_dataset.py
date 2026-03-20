#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


BASE_DIR = "et_distributions"
INDEX_FILE = os.path.join(BASE_DIR, "et_distributions_index.csv")
DIST_DIR = os.path.join(BASE_DIR, "distributions")
OUTPUT_FILE = os.path.join(BASE_DIR, "et_moments_dataset.csv")


def load_distribution(filepath):
    data = np.loadtxt(filepath)
    E = data[:, 0]
    P = data[:, 1]
    return E, P


def compute_moments(E, P):
    s = np.sum(P)
    if s <= 0:
        return np.nan, np.nan, np.nan

    # por seguridad, renormalizar
    P = P / s

    mean_E = np.sum(E * P)
    var_E = np.sum(P * (E - mean_E) ** 2)
    std_E = np.sqrt(var_E)

    return mean_E, var_E, std_E


def main():
    index_df = pd.read_csv(INDEX_FILE)

    rows = []

    for _, row in index_df.iterrows():
        dist_path = os.path.join(DIST_DIR, row["dist_file"])

        if not os.path.exists(dist_path):
            continue

        E, P = load_distribution(dist_path)
        mean_E, var_E, std_E = compute_moments(E, P)

        rows.append({
            "sample_id": int(row["sample_id"]),
            "E_in": float(row["E_in"]),
            "v_in": int(row["v_in"]),
            "j_in": int(row["j_in"]),
            "n_chunks": int(row["n_chunks"]),
            "n_total": int(row["n_total"]),
            "n_reactive": int(row["n_reactive"]),
            "reactive_probability": float(row["reactive_probability"]),
            "mean_Eout": mean_E,
            "var_Eout": var_E,
            "std_Eout": std_E,
            "dist_file": row["dist_file"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Number of samples: {len(out_df)}")


if __name__ == "__main__":
    main()
