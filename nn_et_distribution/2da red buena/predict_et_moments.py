#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import torch
import torch.nn as nn


MODEL_DIR = "nn_et_moments"
MODEL_FILE = f"{MODEL_DIR}/best_model.pt"
SCALER_FILE = f"{MODEL_DIR}/scalers.json"


class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    with open(SCALER_FILE, "r") as f:
        scalers = json.load(f)

    model = MLP(in_dim=3, out_dim=2)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    X_mean = np.array(scalers["X_mean"], dtype=float)
    X_std = np.array(scalers["X_std"], dtype=float)
    Y_mean = np.array(scalers["Y_mean"], dtype=float)
    Y_std = np.array(scalers["Y_std"], dtype=float)

    return model, X_mean, X_std, Y_mean, Y_std


def predict(E_in, v_in, j_in):
    model, X_mean, X_std, Y_mean, Y_std = load_model()

    x = np.array([[E_in, v_in, j_in]], dtype=float)
    x_scaled = (x - X_mean) / X_std

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(x_tensor).numpy()

    y = y_scaled * Y_std + Y_mean

    mean_Eout = float(y[0, 0])
    std_Eout = float(max(0.0, y[0, 1]))  # asegurar no negatividad

    return mean_Eout, std_Eout


if __name__ == "__main__":
    # ejemplo
    E_in = 6.0
    v_in = 0
    j_in = 71

    mean_Eout, std_Eout = predict(E_in, v_in, j_in)

    print(f"Input: E_in={E_in}, v_in={v_in}, j_in={j_in}")
    print(f"Predicted mean_Eout = {mean_Eout:.6f} eV")
    print(f"Predicted std_Eout  = {std_Eout:.6f} eV")
