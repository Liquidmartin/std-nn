#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

BASE_DIR = "et_distributions"
INPUT_FILE = os.path.join(BASE_DIR, "et_moments_dataset.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "et_moments_dataset_filtered.csv")

MIN_REACTIVE = 20


def main():

    df = pd.read_csv(INPUT_FILE)

    print("Total samples:", len(df))

    # Filtrar
    df_filtered = df[df["n_reactive"] >= MIN_REACTIVE].copy()

    print("Filtered samples:", len(df_filtered))

    # porcentaje que queda
    perc = 100 * len(df_filtered) / len(df)
    print(f"Kept: {perc:.2f}%")

    # Guardar
    df_filtered.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)

    # Estadísticas útiles
    print("\nStats after filtering:")
    print(df_filtered["n_reactive"].describe())


if __name__ == "__main__":
    main()
