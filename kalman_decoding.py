"""
Kalman Filter neural decoding pipeline on the S1 (somatosensory cortex) dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.metrics import get_R2, get_rho

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = "Claude_DropBox_Data/example_data_s1.pickle"

with open(DATA_PATH, "rb") as f:
    neural_data, vels_binned = pickle.load(f, encoding="latin1")

print(f"Neural data shape : {neural_data.shape}  (time bins x neurons)")
print(f"Velocity shape    : {vels_binned.shape}  (time bins x [x_vel, y_vel])")

# ── 2. Build output matrix: position + velocity + acceleration ─────────────────
BIN_SIZE = 0.05  # seconds per bin

pos_binned = np.zeros(vels_binned.shape)
for i in range(pos_binned.shape[0] - 1):
    pos_binned[i + 1, 0] = pos_binned[i, 0] + vels_binned[i, 0] * BIN_SIZE
    pos_binned[i + 1, 1] = pos_binned[i, 1] + vels_binned[i, 1] * BIN_SIZE

temp = np.diff(vels_binned, axis=0)
acc_binned = np.concatenate((temp, temp[-1:, :]), axis=0)

y_kf = np.concatenate((pos_binned, vels_binned, acc_binned), axis=1)
X_kf = neural_data  # Kalman filter uses raw firing rates, no history window needed

# ── 3. Lag alignment ──────────────────────────────────────────────────────────
LAG = 0  # bins; negative = spikes lead output, positive = spikes trail
num_examples = X_kf.shape[0]

if LAG < 0:
    y_kf = y_kf[-LAG:, :]
    X_kf = X_kf[:num_examples + LAG, :]
elif LAG > 0:
    y_kf = y_kf[:num_examples - LAG, :]
    X_kf = X_kf[LAG:num_examples, :]

# ── 4. Train / validation / test split (time-ordered) ────────────────────────
TRAIN_RANGE  = [0.00, 0.70]
TEST_RANGE   = [0.70, 0.85]
VALID_RANGE  = [0.85, 1.00]

n = X_kf.shape[0]

def make_idx(r, n):
    return np.arange(int(round(r[0] * n)) + 1, int(round(r[1] * n)) - 1)

train_idx = make_idx(TRAIN_RANGE, n)
test_idx  = make_idx(TEST_RANGE,  n)
valid_idx = make_idx(VALID_RANGE, n)

X_train, y_train = X_kf[train_idx], y_kf[train_idx]
X_test,  y_test  = X_kf[test_idx],  y_kf[test_idx]
X_valid, y_valid = X_kf[valid_idx], y_kf[valid_idx]

print(f"\nSplit sizes — train: {len(train_idx)}, test: {len(test_idx)}, valid: {len(valid_idx)}")

# ── 5. Normalise (z-score inputs, zero-centre outputs) ───────────────────────
X_mean = np.nanmean(X_train, axis=0)
X_std  = np.nanstd(X_train,  axis=0)
X_std[X_std == 0] = 1  # avoid divide-by-zero for silent neurons

X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std

y_mean  = np.mean(y_train, axis=0)
y_train = y_train - y_mean
y_test  = y_test  - y_mean
y_valid = y_valid - y_mean

# ── 6. Fit Kalman Filter and predict ─────────────────────────────────────────
model_kf = KalmanFilterDecoder(C=1)
model_kf.fit(X_train, y_train)

y_valid_pred = model_kf.predict(X_valid, y_valid)
y_test_pred  = model_kf.predict(X_test,  y_test)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
# Columns 2 & 3 are x- and y-velocity
R2_valid  = get_R2(y_valid, y_valid_pred)
rho_valid = get_rho(y_valid, y_valid_pred)

R2_test   = get_R2(y_test,  y_test_pred)
rho_test  = get_rho(y_test,  y_test_pred)

print("\n── Validation set ──")
print(f"  R²  (x-vel, y-vel): {R2_valid[2]:.4f}, {R2_valid[3]:.4f}")
print(f"  ρ²  (x-vel, y-vel): {rho_valid[2]**2:.4f}, {rho_valid[3]**2:.4f}")

print("\n── Test set ──")
print(f"  R²  (x-vel, y-vel): {R2_test[2]:.4f}, {R2_test[3]:.4f}")
print(f"  ρ²  (x-vel, y-vel): {rho_test[2]**2:.4f}, {rho_test[3]**2:.4f}")

# ── 8. Plot ───────────────────────────────────────────────────────────────────
WINDOW = slice(0, 1000)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

for ax, col, label in zip(axes, [2, 3], ["X velocity", "Y velocity"]):
    true = y_valid[WINDOW, col] + y_mean[col]
    pred = y_valid_pred[WINDOW, col] + y_mean[col]
    ax.plot(true, color="steelblue",  label="True",      linewidth=1)
    ax.plot(pred, color="tomato",     label="Predicted", linewidth=1, alpha=0.8)
    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=8)
    r2 = get_R2(y_valid[:, [col]].reshape(-1, 1),
                y_valid_pred[:, [col]].reshape(-1, 1))[0]
    ax.set_title(f"{label}  —  R² = {r2:.3f}")

axes[-1].set_xlabel("Time bin")
fig.suptitle("Kalman Filter Decoding  |  S1 dataset", fontsize=13)
fig.tight_layout()

OUT = "kalman_decoding_results.png"
fig.savefig(OUT, dpi=150)
print(f"\nPlot saved to {OUT}")
