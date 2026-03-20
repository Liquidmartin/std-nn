#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict


# -----------------------------
# Leer archivo de configuración
# -----------------------------
def read_conditions(filename="conditions.txt"):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            key, value = line.split()
            params[key] = value

    # convertir tipos
    params["emin"] = float(params["emin"])
    params["emax"] = float(params["emax"])
    params["bin_width"] = float(params["bin_width"])

    return params


# -----------------------------
# Parsear nombre de carpeta
# -----------------------------
def parse_folder_name(name):
    pattern = r"v(\d+)-j(\d+)-E([0-9.]+)-(\d+)"
    match = re.match(pattern, name)
    if match:
        v = int(match.group(1))
        j = int(match.group(2))
        E = float(match.group(3))
        return v, j, E
    return None


# -----------------------------
# Leer archivo out
# -----------------------------
def read_out_file(filepath):
    try:
        df = pd.read_csv(filepath, delim_whitespace=True)
        return df
    except Exception:
        return None


# -----------------------------
# MAIN
# -----------------------------
def main():

    params = read_conditions()

    ROOT = params["root"]
    OUTDIR = params["outdir"]
    EMIN = params["emin"]
    EMAX = params["emax"]
    BIN_WIDTH = params["bin_width"]

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "distributions"), exist_ok=True)

    # Grid energético
    E_bins = np.arange(EMIN, EMAX + BIN_WIDTH, BIN_WIDTH)
    E_centers = 0.5 * (E_bins[:-1] + E_bins[1:])

    # Guardar grid
    np.savetxt(os.path.join(OUTDIR, "et_grid.dat"), E_centers)

    # Agrupar carpetas por condición inicial
    groups = defaultdict(list)

    for folder in os.listdir(ROOT):
        parsed = parse_folder_name(folder)
        if parsed:
            groups[parsed].append(os.path.join(ROOT, folder))

    index_rows = []
    sample_id = 0

    # -----------------------------
    # Loop por condición inicial
    # -----------------------------
    for (v, j, E), folders in sorted(groups.items()):

        sample_id += 1

        E_out_all = []
        n_total = 0
        n_reactive = 0

        for folder in folders:
            out_file = os.path.join(folder, "out")

            if not os.path.exists(out_file):
                continue

            df = read_out_file(out_file)
            if df is None:
                continue

            n_total += len(df)

            # Filtrar reactivas
            mask = df["Product"].isin(["O2+O3O1", "O3+O1O2"])
            reactive_df = df[mask]

            n_reactive += len(reactive_df)

            if len(reactive_df) > 0:
                E_out_all.extend(reactive_df["ProdEtra"].values)

        E_out_all = np.array(E_out_all)

        # Histograma
        if len(E_out_all) > 0:
            hist, _ = np.histogram(E_out_all, bins=E_bins)
            P = hist / np.sum(hist)
        else:
            P = np.zeros(len(E_centers))

        # Guardar distribución
        dist_filename = f"ETdist_id{sample_id:05d}.dat"
        dist_path = os.path.join(OUTDIR, "distributions", dist_filename)

        np.savetxt(
            dist_path,
            np.column_stack([E_centers, P]),
            header="E_out  P",
            fmt="%.6f %.12e"
        )

        # Guardar índice
        row = {
            "sample_id": sample_id,
            "E_in": E,
            "v_in": v,
            "j_in": j,
            "n_chunks": len(folders),
            "n_total": n_total,
            "n_reactive": n_reactive,
            "reactive_probability": n_reactive / n_total if n_total > 0 else 0,
            "dist_file": dist_filename,
            "sum_P": np.sum(P),
            "min_Eout": np.min(E_out_all) if len(E_out_all) > 0 else np.nan,
            "max_Eout": np.max(E_out_all) if len(E_out_all) > 0 else np.nan,
        }

        index_rows.append(row)

        print(f"[{sample_id}] v={v} j={j} E={E} -> reactive={n_reactive}")

    # Guardar índice
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(OUTDIR, "et_distributions_index.csv"), index=False)

    print("\nDONE.")


if __name__ == "__main__":
    main()
