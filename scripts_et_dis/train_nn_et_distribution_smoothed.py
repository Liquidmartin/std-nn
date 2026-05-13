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
BASE_DIR = "et_distributions_exchange_smoothed"
INDEX_FILE = os.path.join(BASE_DIR, "et_distributions_smoothed_index.csv")
DIST_DIR = os.path.join(BASE_DIR, "distributions")

OUTDIR = "nn_et_distribution_exchange_smoothed"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 42
MIN_SELECTED = 50

TEST_FRAC = 0.15
VAL_FRAC = 0.15
BATCH_SIZE = 64
EPOCHS = 800
LR = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 80
MSE_WEIGHT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================
# Dataset
# =========================
class DistributionDataset(Dataset):
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
class DistMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        logits = self.feature_net(x)
        log_probs = torch.log_softmax(logits, dim=1)
        return log_probs


# =========================
# Helpers
# =========================
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


def standardize(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0

    train_s = (train - mean) / std
    val_s = (val - mean) / std
    test_s = (test - mean) / std

    return train_s, val_s, test_s, mean, std


def load_distribution(filepath):
    data = np.loadtxt(filepath)
    E = data[:, 0]
    P = data[:, 1]
    return E, P


def compute_loss(out_log, yb, kl_loss, mse_loss, mse_weight):
    out_prob = torch.exp(out_log)
    loss = kl_loss(out_log, yb) + mse_weight * mse_loss(out_prob, yb)
    return loss


def evaluate_model(model, loader, kl_loss, mse_loss, mse_weight, device):
    model.eval()
    losses = []
    preds_log = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out_log = model(xb)
            loss = compute_loss(out_log, yb, kl_loss, mse_loss, mse_weight)

            losses.append(loss.item())
            preds_log.append(out_log.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds_log = np.vstack(preds_log)
    trues = np.vstack(trues)

    return np.mean(losses), preds_log, trues


def distribution_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    l1_per_sample = np.sum(np.abs(y_true - y_pred), axis=1)
    mean_l1 = np.mean(l1_per_sample)

    return {
        "mse_all_bins": float(mse),
        "mae_all_bins": float(mae),
        "mean_L1_per_sample": float(mean_l1),
    }


def distribution_moments(E, P):
    mu = np.sum(E * P)
    var = np.sum((E - mu) ** 2 * P)
    sigma = np.sqrt(var)
    return mu, sigma


def rmsd(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


# =========================
# Load metadata
# =========================
print("Using input folder:", BASE_DIR)
print("Index file:", INDEX_FILE)
print("Output folder:", OUTDIR)

df = pd.read_csv(INDEX_FILE)

print("Total samples before filtering:", len(df))

# =========================
# Filtro de calidad
# =========================
if "n_selected" in df.columns:
    df = df[df["n_selected"] >= MIN_SELECTED].copy()
    print(f"Filtering with n_selected >= {MIN_SELECTED}")

elif "n_reactive" in df.columns:
    df = df[df["n_reactive"] >= MIN_SELECTED].copy()
    print(f"Filtering with n_reactive >= {MIN_SELECTED}")

else:
    print("WARNING: No n_selected or n_reactive column found. No filtering applied.")

df = df.reset_index(drop=True)

print("Samples after filtering:", len(df))

feature_cols = ["E_in", "v_in", "j_in"]

X_list = []
Y_list = []
sample_ids = []
dist_files = []

E_grid_ref = None

for _, row in df.iterrows():

    if "smoothed_file" in row.index:
        dist_name = row["smoothed_file"]
    else:
        dist_name = row["dist_file"]

    dist_path = os.path.join(DIST_DIR, dist_name)

    if not os.path.exists(dist_path):
        continue

    E, P = load_distribution(dist_path)

    if E_grid_ref is None:
        E_grid_ref = E.copy()
    else:
        if len(E) != len(E_grid_ref) or not np.allclose(E, E_grid_ref):
            continue

    Psum = np.sum(P)

    if Psum <= 0:
        continue

    P = P / Psum

    X_list.append([row["E_in"], row["v_in"], row["j_in"]])
    Y_list.append(P)
    sample_ids.append(int(row["sample_id"]))
    dist_files.append(dist_name)

X = np.array(X_list, dtype=float)
Y = np.array(Y_list, dtype=float)
sample_ids = np.array(sample_ids)

print("Samples used:", len(X))
print("Number of bins:", Y.shape[1])

if len(X) == 0:
    raise RuntimeError("No samples available for training.")


# =========================
# Split
# =========================
train_idx, val_idx, test_idx = split_indices(
    len(X),
    test_frac=TEST_FRAC,
    val_frac=VAL_FRAC,
    seed=RANDOM_SEED
)

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

ids_test = sample_ids[test_idx]
X_test_original = X[test_idx]

X_train_s, X_val_s, X_test_s, X_mean, X_std = standardize(
    X_train,
    X_val,
    X_test
)

train_ds = DistributionDataset(X_train_s, Y_train)
val_ds = DistributionDataset(X_val_s, Y_val)
test_ds = DistributionDataset(X_test_s, Y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# Model
# =========================
model = DistMLP(in_dim=3, out_dim=Y.shape[1]).to(DEVICE)

kl_loss = nn.KLDivLoss(reduction="batchmean")
mse_loss = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# =========================
# Training
# =========================
best_val_loss = np.inf
best_epoch = -1
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": []
}

for epoch in range(1, EPOCHS + 1):

    model.train()
    batch_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        out_log = model(xb)
        loss = compute_loss(out_log, yb, kl_loss, mse_loss, MSE_WEIGHT)

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    val_loss, _, _ = evaluate_model(
        model,
        val_loader,
        kl_loss,
        mse_loss,
        MSE_WEIGHT,
        DEVICE
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(OUTDIR, "best_model.pt")
        )

    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:4d} | "
            f"train_loss={train_loss:.6e} | "
            f"val_loss={val_loss:.6e}"
        )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6e}")


# =========================
# Test
# =========================
model.load_state_dict(
    torch.load(
        os.path.join(OUTDIR, "best_model.pt"),
        map_location=DEVICE
    )
)

test_loss, pred_log_test, true_test = evaluate_model(
    model,
    test_loader,
    kl_loss,
    mse_loss,
    MSE_WEIGHT,
    DEVICE
)

pred_test = np.exp(pred_log_test)
pred_test = pred_test / pred_test.sum(axis=1, keepdims=True)

metrics = distribution_metrics(true_test, pred_test)
metrics["test_loss"] = float(test_loss)
metrics["mse_weight"] = float(MSE_WEIGHT)

print("\nTest metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.6e}")


# =========================
# Save outputs
# =========================
pd.DataFrame(history).to_csv(
    os.path.join(OUTDIR, "loss_history.csv"),
    index=False
)

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUTDIR, "scalers.json"), "w") as f:
    json.dump(
        {
            "X_mean": X_mean.tolist(),
            "X_std": X_std.tolist(),
            "feature_cols": feature_cols,
        },
        f,
        indent=2
    )

np.savetxt(
    os.path.join(OUTDIR, "energy_grid.dat"),
    E_grid_ref
)

np.save(
    os.path.join(OUTDIR, "true_test.npy"),
    true_test
)

np.save(
    os.path.join(OUTDIR, "pred_test.npy"),
    pred_test
)

np.save(
    os.path.join(OUTDIR, "test_sample_ids.npy"),
    ids_test
)


# =========================
# Plot training curve
# =========================
plt.figure(figsize=(7, 5))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("KL + MSE loss")
plt.title("Training history - ET smooth exchange")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "loss_curve.png"), dpi=200)
plt.close()


# =========================
# Plot test examples
# =========================
n_examples = min(9, len(pred_test))
example_idx = np.linspace(0, len(pred_test) - 1, n_examples, dtype=int)

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.ravel()

for ax, idx in zip(axes, example_idx):
    ax.plot(E_grid_ref, true_test[idx], label="true")
    ax.plot(E_grid_ref, pred_test[idx], label="pred")

    E_in = X_test_original[idx, 0]
    v_in = X_test_original[idx, 1]
    j_in = X_test_original[idx, 2]

    ax.set_title(
        f"id={ids_test[idx]} | "
        f"E={E_in:.2f} eV, v={int(v_in)}, j={int(j_in)}"
    )

    ax.set_xlabel("E_out (eV)")
    ax.set_ylabel("P")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "test_examples.png"), dpi=200)
plt.close()


# =========================
# Parity plots
# =========================
mu_true = []
mu_pred = []
sigma_true = []
sigma_pred = []

for i in range(len(pred_test)):

    mu_t, sig_t = distribution_moments(E_grid_ref, true_test[i])
    mu_p, sig_p = distribution_moments(E_grid_ref, pred_test[i])

    mu_true.append(mu_t)
    mu_pred.append(mu_p)
    sigma_true.append(sig_t)
    sigma_pred.append(sig_p)

mu_true = np.array(mu_true)
mu_pred = np.array(mu_pred)
sigma_true = np.array(sigma_true)
sigma_pred = np.array(sigma_pred)

rmsd_mu = rmsd(mu_true, mu_pred)
r2_mu = r2_score(mu_true, mu_pred)

rmsd_sigma = rmsd(sigma_true, sigma_pred)
r2_sigma = r2_score(sigma_true, sigma_pred)

metrics["rmsd_mean_Eout"] = float(rmsd_mu)
metrics["r2_mean_Eout"] = float(r2_mu)
metrics["rmsd_std_Eout"] = float(rmsd_sigma)
metrics["r2_std_Eout"] = float(r2_sigma)

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


# Mean parity
plt.figure(figsize=(6, 6))
plt.scatter(mu_true, mu_pred, alpha=0.7)

minv = min(mu_true.min(), mu_pred.min())
maxv = max(mu_true.max(), mu_pred.max())

plt.plot([minv, maxv], [minv, maxv])
plt.xlabel("True mean E_out")
plt.ylabel("Predicted mean E_out")
plt.title("Parity plot - Mean Energy")
plt.text(
    0.05,
    0.95,
    f"RMSD = {rmsd_mu:.4f} eV\n$R^2$ = {r2_mu:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "parity_mean.png"), dpi=200)
plt.close()


# Std parity
plt.figure(figsize=(6, 6))
plt.scatter(sigma_true, sigma_pred, alpha=0.7)

minv = min(sigma_true.min(), sigma_pred.min())
maxv = max(sigma_true.max(), sigma_pred.max())

plt.plot([minv, maxv], [minv, maxv])
plt.xlabel("True std E_out")
plt.ylabel("Predicted std E_out")
plt.title("Parity plot - Std Energy")
plt.text(
    0.05,
    0.95,
    f"RMSD = {rmsd_sigma:.4f} eV\n$R^2$ = {r2_sigma:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "parity_std.png"), dpi=200)
plt.close()

print("Saved parity plots.")
print(f"RMSD mean E_out: {rmsd_mu:.6f}, R2 mean E_out: {r2_mu:.6f}")
print(f"RMSD std E_out:  {rmsd_sigma:.6f}, R2 std E_out:  {r2_sigma:.6f}")
print(f"\nAll outputs saved in: {OUTDIR}")
