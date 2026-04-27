"""
Autoencoder sweep: trains autoencoders at varying latent dimensions,
feeds compressed neural data into the Kalman filter, then generates
the three remaining presentation figures (slides 11-13).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import warnings
import os
warnings.filterwarnings("ignore")

OUTPUT_DIR = "autoencoder_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow import keras

from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.metrics import get_R2, get_rho

tf.random.set_seed(42)
np.random.seed(42)

# ── 1. Load & preprocess data (same pipeline as before) ───────────────────────
DATA_PATH = "Claude_DropBox_Data/example_data_s1.pickle"
BIN_SIZE  = 0.05

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
N_NEURONS = X_kf.shape[1]  # 52

def make_idx(r, n):
    return np.arange(int(round(r[0] * n)) + 1, int(round(r[1] * n)) - 1)

train_idx = make_idx([0.00, 0.70], n)
valid_idx = make_idx([0.85, 1.00], n)

X_train, y_train = X_kf[train_idx], y_kf[train_idx]
X_valid, y_valid = X_kf[valid_idx], y_kf[valid_idx]

# Z-score inputs, zero-centre outputs (train stats only)
X_mean = np.nanmean(X_train, axis=0)
X_std  = np.nanstd(X_train,  axis=0)
X_std[X_std == 0] = 1

X_train_z = (X_train - X_mean) / X_std
X_valid_z = (X_valid - X_mean) / X_std

y_mean    = np.mean(y_train, axis=0)
y_train_c = y_train - y_mean
y_valid_c = y_valid - y_mean

print(f"Data loaded: {N_NEURONS} neurons, {len(train_idx)} train bins, {len(valid_idx)} valid bins")

# ── 2. Autoencoder builder ────────────────────────────────────────────────────
def build_autoencoder(n_neurons, latent_dim):
    inputs  = keras.Input(shape=(n_neurons,))
    encoded = keras.layers.Dense(latent_dim, activation="relu")(inputs)
    decoded = keras.layers.Dense(n_neurons,  activation="linear")(encoded)
    autoencoder = keras.Model(inputs, decoded)
    encoder     = keras.Model(inputs, encoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder

# ── 3. Sweep latent dimensions ────────────────────────────────────────────────
LATENT_DIMS = [4, 8, 13, 26, 32, N_NEURONS]  # N_NEURONS = uncompressed baseline
results = []  # list of dicts

for ldim in LATENT_DIMS:
    print(f"\nLatent dim: {ldim}", end="  ", flush=True)

    tag = f"latent{ldim:03d}"
    out = os.path.join(OUTPUT_DIR, tag)
    os.makedirs(out, exist_ok=True)

    if ldim == N_NEURONS:
        # Uncompressed baseline — skip autoencoder, perfect reconstruction
        X_tr = X_train_z
        X_va = X_valid_z
        recon_r2 = 1.0
        # Save the z-scored data as the "encoded" output for consistency
        np.save(f"{out}/X_train_encoded.npy", X_tr)
        np.save(f"{out}/X_valid_encoded.npy", X_va)
        np.save(f"{out}/X_valid_reconstructed.npy", X_va)  # reconstruction = original
        print("(uncompressed baseline)")
    else:
        ae, enc = build_autoencoder(N_NEURONS, ldim)
        ae.fit(X_train_z, X_train_z,
               epochs=80, batch_size=256,
               validation_split=0.1,
               verbose=0)

        # Reconstruction quality: how well does X̂ = decode(encode(X)) match X?
        X_reconstructed = ae.predict(X_valid_z, verbose=0)
        recon_r2 = float(np.mean(get_R2(X_valid_z, X_reconstructed)))

        X_tr = enc.predict(X_train_z, verbose=0)
        X_va = enc.predict(X_valid_z, verbose=0)

        # Save encoded representations and reconstruction
        np.save(f"{out}/X_train_encoded.npy", X_tr)
        np.save(f"{out}/X_valid_encoded.npy",  X_va)
        np.save(f"{out}/X_valid_reconstructed.npy", X_reconstructed)

        # Save model weights
        ae.save(f"{out}/autoencoder.keras")
        enc.save(f"{out}/encoder.keras")

        # Re-normalise latent representations using train stats
        lt_mean = np.mean(X_tr, axis=0)
        lt_std  = np.std(X_tr,  axis=0)
        lt_std[lt_std == 0] = 1
        X_tr = (X_tr - lt_mean) / lt_std
        X_va = (X_va - lt_mean) / lt_std
        print(f"autoencoder trained, encoded shape: {X_tr.shape[1]}  recon R²={recon_r2:.3f}")

    model_kf = KalmanFilterDecoder(C=1)
    model_kf.fit(X_tr, y_train_c)
    y_pred = model_kf.predict(X_va, y_valid_c)

    errors = y_pred - y_valid_c          # shape (T, 6)
    R2     = get_R2(y_valid_c, y_pred)   # shape (6,)
    rho    = get_rho(y_valid_c, y_pred)  # shape (6,)

    # Velocity columns: indices 2 (x) and 3 (y)
    results.append({
        "latent_dim":       ldim,
        "compression":      N_NEURONS / ldim,
        "error_mean_xv":    np.mean(errors[:, 2]),
        "error_mean_yv":    np.mean(errors[:, 3]),
        "error_std_xv":     np.std(errors[:, 2]),
        "error_std_yv":     np.std(errors[:, 3]),
        "r2_xv":            R2[2],
        "r2_yv":            R2[3],
        "rho2_xv":          rho[2] ** 2,
        "rho2_yv":          rho[3] ** 2,
        "recon_r2":         recon_r2,
        "y_pred":           y_pred,   # saved for qualitative plots
    })
    print(f"  R² x-vel={R2[2]:.3f}  y-vel={R2[3]:.3f}")

print("\nSweep complete. Saving metrics summary...")

# Save all metrics to a CSV
import csv
csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
fieldnames = ["latent_dim", "compression", "error_mean_xv", "error_mean_yv",
              "error_std_xv", "error_std_yv", "r2_xv", "r2_yv",
              "rho2_xv", "rho2_yv", "recon_r2"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in fieldnames})
print(f"Saved {csv_path}")

print("Generating figures...")

# ── Helper arrays ─────────────────────────────────────────────────────────────
ldims       = np.array([r["latent_dim"]    for r in results])
compressions= np.array([r["compression"]   for r in results])
em_xv = np.array([r["error_mean_xv"] for r in results])
em_yv = np.array([r["error_mean_yv"] for r in results])
es_xv = np.array([r["error_std_xv"]  for r in results])
es_yv = np.array([r["error_std_yv"]  for r in results])
r2_xv = np.array([r["r2_xv"]         for r in results])
r2_yv = np.array([r["r2_yv"]         for r in results])
rh_xv    = np.array([r["rho2_xv"]  for r in results])
rh_yv    = np.array([r["rho2_yv"]  for r in results])
recon_r2 = np.array([r["recon_r2"] for r in results])

ldim_labels = [str(d) for d in ldims]
BASE = N_NEURONS  # uncompressed reference

# ── Figure 4: Results grid (slide 11 + reconstruction R²) ────────────────────
from matplotlib.gridspec import GridSpec

fig4 = plt.figure(figsize=(13, 13))
fig4.suptitle("Results: Kalman Decoding Across Latent Dimensions", fontsize=13)
gs = GridSpec(3, 2, figure=fig4, hspace=0.45, wspace=0.3)

ax_em  = fig4.add_subplot(gs[0, 0])
ax_es  = fig4.add_subplot(gs[0, 1])
ax_r2  = fig4.add_subplot(gs[1, 0])
ax_rh  = fig4.add_subplot(gs[1, 1])
ax_rec = fig4.add_subplot(gs[2, :])   # spans full bottom row

base_idx = list(ldims).index(BASE)

# Four velocity-decoding panels
for (title, xv, yv), ax in zip(
    [("error_mean", em_xv, em_yv),
     ("error_std",  es_xv, es_yv),
     ("r2",         r2_xv, r2_yv),
     ("rho2",       rh_xv, rh_yv)],
    [ax_em, ax_es, ax_r2, ax_rh]
):
    ax.plot(ldim_labels, xv, "o-", color="#4472C4", label="x-velocity")
    ax.plot(ldim_labels, yv, "s-", color="#ED7D31", label="y-velocity")
    ax.axvline(x=base_idx, color="salmon", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.plot(base_idx, xv[base_idx], "o",
            markersize=14, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    ax.plot(base_idx, yv[base_idx], "o",
            markersize=14, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    ax.set_title(title)
    ax.set_xlabel("latent_dim")
    ax.set_ylabel(title)
    ax.set_xticks(range(len(ldim_labels)))
    ax.set_xticklabels(ldim_labels)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Reconstruction R² panel (single line — averaged across neurons)
ax_rec.plot(ldim_labels, recon_r2, "D-", color="#5B9359", linewidth=1.5,
            markersize=7, label="mean R² across neurons")
ax_rec.axvline(x=base_idx, color="salmon", linestyle="--", linewidth=1.2, alpha=0.8)
ax_rec.plot(base_idx, recon_r2[base_idx], "o",
            markersize=14, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
ax_rec.set_title("Autoencoder Reconstruction R²  (X̂ vs original X, averaged over neurons)")
ax_rec.set_xlabel("latent_dim")
ax_rec.set_ylabel("reconstruction R²")
ax_rec.set_xticks(range(len(ldim_labels)))
ax_rec.set_xticklabels(ldim_labels)
ax_rec.set_ylim(-0.05, 1.05)
ax_rec.axhline(y=1.0, color="grey", linestyle=":", linewidth=1)
ax_rec.legend(fontsize=8)
ax_rec.grid(True, alpha=0.3)

fig4.savefig("fig4_results_grid.png", dpi=150, bbox_inches="tight")
print("Saved fig4_results_grid.png")

# ── Figure 5: Compression factor vs latent_dim (slide 12) ────────────────────
fig5, ax = plt.subplots(figsize=(7, 5))
ax.scatter(ldims, compressions, color="#003399", s=120, zorder=3)
for x, y in zip(ldims, compressions):
    ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=9)
ax.set_xlabel("latent_dim")
ax.set_ylabel("compression factor")
ax.set_title("Compression Factor vs Latent Dim")
ax.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig("fig5_compression.png", dpi=150)
print("Saved fig5_compression.png")

# ── Figure 6: Side-by-side comparison worst vs best (slide 13) ───────────────
WINDOW = 400
col    = 3  # y-velocity

def make_region_plot(ax, true_raw, pred_raw, title):
    t = np.arange(len(true_raw))
    n_windows = len(true_raw) // WINDOW
    window_mae = np.array([
        np.mean(np.abs(pred_raw[i*WINDOW:(i+1)*WINDOW] - true_raw[i*WINDOW:(i+1)*WINDOW]))
        for i in range(n_windows)
    ])
    threshold = np.mean(window_mae)
    for i, mae in enumerate(window_mae):
        colour = "#90EE90" if mae <= threshold else "#FFB6B6"
        ax.axvspan(i * WINDOW, (i + 1) * WINDOW, color=colour, alpha=0.5, linewidth=0)
    ax.plot(t, true_raw, color="#5B9BD5", linestyle="--", linewidth=0.7,
            label=f"True Output (idx {col})")
    ax.plot(t, pred_raw, color="#ED7D31", linestyle="-",  linewidth=0.7,
            label=f"Predicted Output (idx {col})")
    ax.set_title(title)
    ax.set_xlim(0, n_windows * WINDOW)
    ax.set_ylabel("Value")

worst_pred = results[0]["y_pred"]   # smallest latent_dim
best_pred  = results[-1]["y_pred"]  # uncompressed

true_v = y_valid_c[:, col] + y_mean[col]
worst_v = worst_pred[:, col] + y_mean[col]
best_v  = best_pred[:,  col] + y_mean[col]

fig6, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig6.suptitle(f"Side-by-Side Comparison ({LATENT_DIMS[0]}n vs {LATENT_DIMS[-1]}n)", fontsize=13)

make_region_plot(ax_top, true_v, worst_v,
                 f"Kalman Filter Prediction — {LATENT_DIMS[0]}d (most compressed)")
make_region_plot(ax_bot, true_v, best_v,
                 f"Kalman Filter Prediction — {LATENT_DIMS[-1]}d (uncompressed)")

ax_bot.set_xlabel("Time bins")
for ax in (ax_top, ax_bot):
    ax.legend(fontsize=8, loc="upper right")

fig6.tight_layout()
fig6.savefig("fig6_sidebyside.png", dpi=150)
print("Saved fig6_sidebyside.png")

print("\nAll figures saved.")
