#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict


# ==========================================================
# Leer archivo de configuración
# ==========================================================
def read_conditions(filename="conditions.txt"):
    params = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line == "" or line.startswith("#"):
                continue

            key, value = line.split()
            params[key] = value

    params["emin"] = float(params["emin"])
    params["emax"] = float(params["emax"])
    params["bin_width"] = float(params["bin_width"])

    # proceso por defecto
    if "process" not in params:
        params["process"] = "exchange"

    return params


# ==========================================================
# Parsear nombre de carpeta
# Ejemplo: v0-j71-E6.0-1
# ==========================================================
def parse_folder_name(name):
    pattern = r"^v(\d+)-j(\d+)-E([0-9.]+)-(\d+)$"
    match = re.match(pattern, name)

    if match:
        v = int(match.group(1))
        j = int(match.group(2))
        E = float(match.group(3))
        chunk = int(match.group(4))
        return v, j, E, chunk

    return None


# ==========================================================
# Leer archivo out
# ==========================================================
def read_out_file(filepath):
    try:
        df = pd.read_csv(filepath, delim_whitespace=True)
        return df
    except Exception as exc:
        print(f"Could not read {filepath}: {exc}")
        return None


# ==========================================================
# Máscaras de procesos
# ==========================================================
def get_process_mask(df, process):
    """
    Processes:
      exchange:
          Product == O2+O3O1 or O3+O1O2

      inelastic:
          Product == O1+O2O3
          and rounded(v_f) != v_i OR rounded(j_f) != j_i

      elastic:
          Product == O1+O2O3
          and rounded(v_f) == v_i AND rounded(j_f) == j_i

      exchange_inelastic:
          exchange OR inelastic

      all_bound:
          exchange OR inelastic OR elastic
          i.e. Product in [O1+O2O3, O2+O3O1, O3+O1O2]
    """

    exchange_products = ["O2+O3O1", "O3+O1O2"]
    nonreactive_product = "O1+O2O3"

    exchange_mask = df["Product"].isin(exchange_products)

    same_product_mask = df["Product"] == nonreactive_product

    # Discretizar SOLO para clasificar elastic/inelastic
    vf_round = np.rint(df["v_f"].astype(float)).astype(int)
    jf_round = np.rint(df["j_f"].astype(float)).astype(int)

    vi = df["v_i"].astype(int)
    ji = df["j_i"].astype(int)

    changed_state = (vf_round != vi) | (jf_round != ji)
    same_state = (vf_round == vi) & (jf_round == ji)

    inelastic_mask = same_product_mask & changed_state
    elastic_mask = same_product_mask & same_state

    if process == "exchange":
        return exchange_mask

    elif process == "inelastic":
        return inelastic_mask

    elif process == "elastic":
        return elastic_mask

    elif process in ["exchange_inelastic", "exchange+inelastic"]:
        return exchange_mask | inelastic_mask

    elif process == "all_bound":
        return exchange_mask | inelastic_mask | elastic_mask

    else:
        raise ValueError(
            f"Unknown process '{process}'. Use one of: "
            "exchange, inelastic, elastic, exchange_inelastic, all_bound"
        )


# ==========================================================
# MAIN
# ==========================================================
def main():

    params = read_conditions()

    ROOT = params["root"]
    OUTDIR = params["outdir"]
    EMIN = params["emin"]
    EMAX = params["emax"]
    BIN_WIDTH = params["bin_width"]
    PROCESS = params["process"]

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "distributions"), exist_ok=True)

    print("====================================")
    print("Building ET distributions")
    print("Root:     ", ROOT)
    print("Outdir:   ", OUTDIR)
    print("Process:  ", PROCESS)
    print("E range:  ", EMIN, EMAX)
    print("Bin width:", BIN_WIDTH)
    print("====================================")

    # Grid energético
    E_bins = np.arange(EMIN, EMAX + BIN_WIDTH, BIN_WIDTH)
    E_centers = 0.5 * (E_bins[:-1] + E_bins[1:])

    np.savetxt(os.path.join(OUTDIR, "et_grid.dat"), E_centers)

    # Agrupar carpetas por condición inicial
    groups = defaultdict(list)

    for folder in os.listdir(ROOT):
        full_path = os.path.join(ROOT, folder)

        if not os.path.isdir(full_path):
            continue

        parsed = parse_folder_name(folder)

        if parsed:
            v, j, E, chunk = parsed
            groups[(v, j, E)].append(full_path)

    index_rows = []
    sample_id = 0

    # Loop por condición inicial
    for (v, j, E), folders in sorted(groups.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):

        sample_id += 1

        E_out_all = []
        n_total = 0
        n_selected = 0

        for folder in sorted(folders):
            out_file = os.path.join(folder, "out")

            if not os.path.exists(out_file):
                continue

            df = read_out_file(out_file)

            if df is None:
                continue

            required_cols = ["Product", "v_i", "j_i", "v_f", "j_f", "ProdEtra"]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                print(f"Skipping {out_file}: missing columns {missing}")
                continue

            n_total += len(df)

            mask = get_process_mask(df, PROCESS)
            selected_df = df[mask]

            n_selected += len(selected_df)

            if len(selected_df) > 0:
                Eout = selected_df["ProdEtra"].astype(float).values
                Eout = Eout[np.isfinite(Eout)]
                Eout = Eout[Eout >= 0.0]
                E_out_all.extend(Eout.tolist())

        E_out_all = np.array(E_out_all, dtype=float)

        # Histograma
        if len(E_out_all) > 0:
            hist, _ = np.histogram(E_out_all, bins=E_bins)
            P = hist.astype(float)

            if P.sum() > 0:
                P /= P.sum()
        else:
            P = np.zeros(len(E_centers), dtype=float)

        # Guardar distribución
        dist_filename = f"ETdist_id{sample_id:05d}.dat"
        dist_path = os.path.join(OUTDIR, "distributions", dist_filename)

        np.savetxt(
            dist_path,
            np.column_stack([E_centers, P]),
            header="E_out  P",
            fmt="%.6f %.12e"
        )

        row = {
            "sample_id": sample_id,
            "E_in": E,
            "v_in": v,
            "j_in": j,
            "process": PROCESS,
            "n_chunks": len(folders),
            "n_total": n_total,
            "n_selected": n_selected,
            "selected_probability": n_selected / n_total if n_total > 0 else 0.0,
            "dist_file": dist_filename,
            "sum_P": float(np.sum(P)),
            "min_Eout": float(np.min(E_out_all)) if len(E_out_all) > 0 else np.nan,
            "max_Eout": float(np.max(E_out_all)) if len(E_out_all) > 0 else np.nan,
        }

        # Mantener compatibilidad si el proceso es exchange
        if PROCESS == "exchange":
            row["n_reactive"] = n_selected
            row["reactive_probability"] = row["selected_probability"]

        index_rows.append(row)

        print(
            f"[{sample_id}] v={v} j={j} E={E} "
            f"-> selected={n_selected}/{n_total}"
        )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(OUTDIR, "et_distributions_index.csv"), index=False)

    print("\nDONE.")
    print(f"Index saved in: {os.path.join(OUTDIR, 'et_distributions_index.csv')}")
    print(f"Distributions saved in: {os.path.join(OUTDIR, 'distributions')}")


if __name__ == "__main__":
    main()
