#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
BASE_DIR = "et_distributions"
INPUT_FILE = os.path.join(BASE_DIR, "et_moments_dataset_filtered.csv")
OUTDIR = "nn_et_moments"

os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
BATCH_SIZE = 64
EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Reproducibility
# =========================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================
# Dataset
# =========================
class MomentsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================
# Model
# =========================
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


# =========================
# Helpers
# =========================
def standardize(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0

    train_s = (train - mean) / std
    val_s = (val - mean) / std
    test_s = (test - mean) / std
    return train_s, val_s, test_s, mean, std


def split_indices(n, test_frac=0.15, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)

    r2 = []
    for k in range(y_true.shape[1]):
        ss_res = np.sum((y_true[:, k] - y_pred[:, k]) ** 2)
        ss_tot = np.sum((y_true[:, k] - np.mean(y_true[:, k])) ** 2)
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    r2 = np.array(r2)

    return {
        "rmse_mean_Eout": float(rmse[0]),
        "rmse_std_Eout": float(rmse[1]),
        "r2_mean_Eout": float(r2[0]),
        "r2_std_Eout": float(r2[1]),
    }


def evaluate_model(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)
            loss = criterion(out, yb)

            losses.append(loss.item())
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return np.mean(losses), preds, trues


# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_FILE)

# Keep only necessary columns
feature_cols = ["E_in", "v_in", "j_in"]
target_cols = ["mean_Eout", "std_Eout"]

df = df.dropna(subset=feature_cols + target_cols).copy()

X = df[feature_cols].values
Y = df[target_cols].values

# Optional: remove non-physical std
mask = Y[:, 1] >= 0.0
df = df.loc[mask].reset_index(drop=True)
X = X[mask]
Y = Y[mask]

print("Samples used:", len(df))

# =========================
# Split
# =========================
train_idx, val_idx, test_idx = split_indices(
    len(df), test_frac=TEST_FRAC, val_frac=VAL_FRAC, seed=RANDOM_SEED
)

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

# Standardize using train only
X_train_s, X_val_s, X_test_s, X_mean, X_std = standardize(X_train, X_val, X_test)
Y_train_s, Y_val_s, Y_test_s, Y_mean, Y_std = standardize(Y_train, Y_val, Y_test)

# Datasets / loaders
train_ds = MomentsDataset(X_train_s, Y_train_s)
val_ds = MomentsDataset(X_val_s, Y_val_s)
test_ds = MomentsDataset(X_test_s, Y_test_s)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Model / optimizer
# =========================
model = MLP(in_dim=3, out_dim=2).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =========================
# Training
# =========================
best_val_loss = np.inf
best_epoch = -1
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
}

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)
    val_loss, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTDIR, "best_model.pt"))
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")

# =========================
# Load best and test
# =========================
model.load_state_dict(torch.load(os.path.join(OUTDIR, "best_model.pt"), map_location=DEVICE))

_, pred_test_s, true_test_s = evaluate_model(model, test_loader, criterion, DEVICE)

# Unscale predictions
pred_test = pred_test_s * Y_std + Y_mean
true_test = true_test_s * Y_std + Y_mean

# Enforce non-negative std prediction
pred_test[:, 1] = np.clip(pred_test[:, 1], 0.0, None)

metrics = compute_metrics(true_test, pred_test)

print("\nTest metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

# =========================
# Save outputs
# =========================
pd.DataFrame({
    "train_loss": history["train_loss"],
    "val_loss": history["val_loss"],
}).to_csv(os.path.join(OUTDIR, "loss_history.csv"), index=False)

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save scaler info
scalers = {
    "X_mean": X_mean.tolist(),
    "X_std": X_std.tolist(),
    "Y_mean": Y_mean.tolist(),
    "Y_std": Y_std.tolist(),
    "feature_cols": feature_cols,
    "target_cols": target_cols,
}
with open(os.path.join(OUTDIR, "scalers.json"), "w") as f:
    json.dump(scalers, f, indent=2)

# Save test predictions
test_df = pd.DataFrame({
    "E_in": X_test[:, 0],
    "v_in": X_test[:, 1],
    "j_in": X_test[:, 2],
    "true_mean_Eout": true_test[:, 0],
    "pred_mean_Eout": pred_test[:, 0],
    "true_std_Eout": true_test[:, 1],
    "pred_std_Eout": pred_test[:, 1],
})
test_df.to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)

# =========================
# Plots
# =========================
plt.figure(figsize=(7, 5))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training history")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "loss_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(true_test[:, 0], pred_test[:, 0], alpha=0.6)
mn = min(true_test[:, 0].min(), pred_test[:, 0].min())
mx = max(true_test[:, 0].max(), pred_test[:, 0].max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True mean_Eout")
plt.ylabel("Predicted mean_Eout")
plt.title("mean_Eout: true vs predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "parity_mean_Eout.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(true_test[:, 1], pred_test[:, 1], alpha=0.6)
mn = min(true_test[:, 1].min(), pred_test[:, 1].min())
mx = max(true_test[:, 1].max(), pred_test[:, 1].max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True std_Eout")
plt.ylabel("Predicted std_Eout")
plt.title("std_Eout: true vs predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "parity_std_Eout.png"), dpi=200)
plt.close()

print(f"\nAll outputs saved in: {OUTDIR}")
