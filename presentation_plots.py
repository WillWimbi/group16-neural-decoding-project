"""
Generates the three presentation figure types from Kalman filter predictions:
  1. Qualitative prediction check (full validation set, two outputs)
  2. Prediction check with red/green quality regions
  3. Error histograms with green/red background bands
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.metrics import get_R2, get_rho

# ── Pipeline (mirrors kalman_decoding.py) ─────────────────────────────────────
DATA_PATH = "Claude_DropBox_Data/example_data_s1.pickle"
BIN_SIZE  = 0.05
LAG       = 0

with open(DATA_PATH, "rb") as f:
    neural_data, vels_binned = pickle.load(f, encoding="latin1")

pos_binned = np.zeros(vels_binned.shape)
for i in range(pos_binned.shape[0] - 1):
    pos_binned[i + 1, 0] = pos_binned[i, 0] + vels_binned[i, 0] * BIN_SIZE
    pos_binned[i + 1, 1] = pos_binned[i, 1] + vels_binned[i, 1] * BIN_SIZE

temp       = np.diff(vels_binned, axis=0)
acc_binned = np.concatenate((temp, temp[-1:, :]), axis=0)

y_kf = np.concatenate((pos_binned, vels_binned, acc_binned), axis=1)
X_kf = neural_data
n    = X_kf.shape[0]

def make_idx(r, n):
    return np.arange(int(round(r[0] * n)) + 1, int(round(r[1] * n)) - 1)

train_idx = make_idx([0.00, 0.70], n)
test_idx  = make_idx([0.70, 0.85], n)
valid_idx = make_idx([0.85, 1.00], n)

X_train, y_train = X_kf[train_idx], y_kf[train_idx]
X_test,  y_test  = X_kf[test_idx],  y_kf[test_idx]
X_valid, y_valid = X_kf[valid_idx], y_kf[valid_idx]

X_mean = np.nanmean(X_train, axis=0)
X_std  = np.nanstd(X_train,  axis=0)
X_std[X_std == 0] = 1

X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std

y_mean  = np.mean(y_train, axis=0)
y_train = y_train - y_mean
y_test  = y_test  - y_mean
y_valid = y_valid - y_mean

model_kf = KalmanFilterDecoder(C=1)
model_kf.fit(X_train, y_train)
y_valid_pred = model_kf.predict(X_valid, y_valid)

print("Pipeline complete. Generating figures...")

# ── Figure 1: Qualitative prediction check (slide 5) ─────────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig1.suptitle("Qualitative Prediction Check", fontsize=14)

for ax, col in zip(axes, [2, 3]):
    true = y_valid[:, col] + y_mean[col]
    pred = y_valid_pred[:, col] + y_mean[col]
    t    = np.arange(len(true))

    ax.plot(t, true, color="#5B9BD5", linestyle="--", linewidth=0.8,
            label=f"True Output (idx {col})")
    ax.plot(t, pred, color="#ED7D31", linestyle="-",  linewidth=0.8,
            label=f"Predicted Output (idx {col})")
    ax.set_ylabel("Value")
    ax.set_title(f"Kalman Filter Prediction (Output Index {col})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(np.percentile(true, 0.5) - 2, np.percentile(true, 99.5) + 2)

axes[-1].set_xlabel("Time bins")
fig1.tight_layout()
fig1.savefig("fig1_qualitative_check.png", dpi=150)
print("Saved fig1_qualitative_check.png")

# ── Figure 2: Prediction check with red/green regions (slide 6) ──────────────
WINDOW = 400  # bins per quality-assessment window
col    = 2    # show x-velocity (output index 2)

true = y_valid[:, col] + y_mean[col]
pred = y_valid_pred[:, col] + y_mean[col]
t    = np.arange(len(true))

# Compute windowed MAE and threshold
n_windows  = len(true) // WINDOW
window_mae = np.array([
    np.mean(np.abs(pred[i*WINDOW:(i+1)*WINDOW] - true[i*WINDOW:(i+1)*WINDOW]))
    for i in range(n_windows)
])
threshold = np.mean(window_mae)

fig2, ax = plt.subplots(figsize=(14, 4))

# Draw coloured background regions
for i, mae in enumerate(window_mae):
    colour = "#90EE90" if mae <= threshold else "#FFB6B6"
    ax.axvspan(i * WINDOW, (i + 1) * WINDOW, color=colour, alpha=0.5, linewidth=0)

ax.plot(t, true, color="#5B9BD5", linestyle="--", linewidth=0.8,
        label=f"True Output (idx {col})")
ax.plot(t, pred, color="#ED7D31", linestyle="-",  linewidth=0.8,
        label=f"Predicted Output (idx {col})")

ax.set_xlabel("Time bins")
ax.set_ylabel("Value")
ax.set_title(f"Kalman Filter Prediction (Output Index {col})")
ax.set_xlim(0, n_windows * WINDOW)

green_patch = mpatches.Patch(color="#90EE90", alpha=0.7, label="Good prediction (MAE ≤ mean)")
red_patch   = mpatches.Patch(color="#FFB6B6", alpha=0.7, label="Poor prediction (MAE > mean)")
ax.legend(handles=[ax.lines[0], ax.lines[1], green_patch, red_patch],
          fontsize=8, loc="upper right")

fig2.tight_layout()
fig2.savefig("fig2_region_check.png", dpi=150)
print("Saved fig2_region_check.png")

# ── Figure 3: Error histograms (slide 7) ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
fig3.suptitle("Prediction Error Histograms — Kalman Filter", fontsize=13)

for ax, col, label in zip(axes, [2, 3], ["x-velocity (output 2)", "y-velocity (output 3)"]):
    errors = y_valid_pred[:, col] - y_valid[:, col]

    err_mean = np.mean(errors)
    err_std  = np.std(errors)

    lo, hi = err_mean - err_std, err_mean + err_std
    x_min  = np.floor(np.percentile(errors, 0.5))
    x_max  = np.ceil(np.percentile(errors, 99.5))

    # Background bands
    ax.axvspan(x_min, lo,  color="#FFB6B6", alpha=0.6, zorder=0)
    ax.axvspan(lo,    hi,  color="#90EE90", alpha=0.6, zorder=0)
    ax.axvspan(hi,    x_max, color="#FFB6B6", alpha=0.6, zorder=0)

    ax.hist(errors, bins=40, color="#2E7D32", edgecolor="white",
            linewidth=0.4, zorder=2)

    ax.axvline(err_mean, color="black", linestyle="--", linewidth=1, zorder=3)
    ax.set_xlabel(f"Prediction Error ({label})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Prediction Error Histogram ({label})")
    ax.set_xlim(x_min, x_max)

    green_patch = mpatches.Patch(color="#90EE90", alpha=0.7,
                                  label=f"±1 std  [{lo:.1f}, {hi:.1f}]")
    red_patch   = mpatches.Patch(color="#FFB6B6", alpha=0.7, label="Tails")
    ax.legend(handles=[green_patch, red_patch], fontsize=8)

fig3.tight_layout()
fig3.savefig("fig3_error_histograms.png", dpi=150)
print("Saved fig3_error_histograms.png")

print("\nAll figures saved.")
