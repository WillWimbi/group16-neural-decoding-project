"""
Autoencoder latent-dimension sweep + Kalman filter decoding on the
S1 dataset (Glaser et al. 2017).

Standalone: the Kalman filter and metrics are implemented in this file
so the script does not depend on the Neural_Decoding package.

Inputs
------
example_data_s1.pickle: a list [neural_data, vels] where
    neural_data : (T, 52) float, spike counts per 50 ms bin
    vels        : (T, 2)  float, hand velocity (vx, vy)
Set DATA_PATH below to point to it.

Outputs (in this folder)
------------------------
fig1_qualitative_check.png
fig2_region_check.png
fig3_error_histograms.png
fig4_results_grid.png
fig5_compression.png
fig6_sidebyside.png
results.json
"""
import os, json, pickle, time
import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Path to example_data_s1.pickle --------------------------------------
# Set the S1_DATA environment variable, or replace the empty string below
# with the full path on your machine.
DATA_PATH = os.environ.get("S1_DATA", "")
if not DATA_PATH:
    raise SystemExit(
        "Set S1_DATA (env var) to the full path of example_data_s1.pickle, "
        "or hard-code DATA_PATH at the top of this file."
    )
# --------------------------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = 0.05
LATENT_DIMS = [4, 8, 13, 26, 32, 52]
KF_C = 1.0


# ===== Kalman filter (Wu et al. 2003 form, with a noise-scaling C) ========
class KalmanFilter:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X_train, y_train):
        X = np.matrix(y_train.T)
        Z = np.matrix(X_train.T)
        nt = X.shape[1]
        X2, X1 = X[:, 1:], X[:, : nt - 1]
        A = X2 * X1.T * inv(X1 * X1.T)
        W = (X2 - A * X1) * (X2 - A * X1).T / (nt - 1) / self.C
        H = Z * X.T * inv(X * X.T)
        Q = ((Z - H * X) * (Z - H * X).T) / nt
        self.params = (A, W, H, Q)

    def predict(self, X_test, y_test):
        A, W, H, Q = self.params
        X = np.matrix(y_test.T)
        Z = np.matrix(X_test.T)
        n_state = X.shape[0]
        states = np.empty(X.shape)
        P = np.matrix(np.zeros((n_state, n_state)))
        state = X[:, 0]
        states[:, 0] = np.squeeze(state)
        I = np.matrix(np.eye(n_state))
        for t in range(X.shape[1] - 1):
            P_m = A * P * A.T + W
            state_m = A * state
            K = P_m * H.T * inv(H * P_m * H.T + Q)
            P = (I - K * H) * P_m
            state = state_m + K * (Z[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(state)
        return states.T


def r2_per_output(y_true, y_pred):
    out = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        ym = y_true[:, i].mean()
        out[i] = 1 - np.sum((y_pred[:, i] - y_true[:, i]) ** 2) / np.sum(
            (y_true[:, i] - ym) ** 2
        )
    return out


def rho_per_output(y_true, y_pred):
    out = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        out[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    return out


# ===== Data loading and split =============================================
with open(DATA_PATH, "rb") as f:
    spikes, vels = pickle.load(f, encoding="latin1")
print(f"data: spikes {spikes.shape}, vels {vels.shape}")

# Build kinematic state [pos_x, pos_y, vx, vy, ax, ay]. The KF benefits from
# the full state even though we only score on velocity.
pos = np.zeros_like(vels)
for i in range(pos.shape[0] - 1):
    pos[i + 1] = pos[i] + vels[i] * DT
acc = np.diff(vels, axis=0)
acc = np.concatenate([acc, acc[-1:]], axis=0)
y_full = np.concatenate([pos, vels, acc], axis=1)
X_full = spikes.copy()

# 70/15/15 chronological split with a 1-bin buffer between contiguous sets.
n = X_full.shape[0]


def slice_set(lo, hi):
    return np.arange(int(round(lo * n)) + 1, int(round(hi * n)) - 1)


tr, te, va = slice_set(0.0, 0.70), slice_set(0.70, 0.85), slice_set(0.85, 1.00)
print(f"split: train {len(tr)}  test {len(te)}  valid {len(va)}")

x_mu = X_full[tr].mean(axis=0)
x_sd = X_full[tr].std(axis=0) + 1e-12
X_z = (X_full - x_mu) / x_sd


# ===== Autoencoder ========================================================
class AE(nn.Module):
    def __init__(self, n_in, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, n_in))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_and_encode(X_train_z, X_all_z, latent, epochs=200, lr=1e-3, bs=256):
    model = AE(X_train_z.shape[1], latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X_train_z, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        idx = torch.randperm(Xt.shape[0], device=DEVICE)
        for s in range(0, Xt.shape[0], bs):
            b = Xt[idx[s : s + bs]]
            recon, _ = model(b)
            loss = ((recon - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        recon_all, z_all = model(torch.tensor(X_all_z, dtype=torch.float32, device=DEVICE))
    Rnp = recon_all.cpu().numpy()
    ss_res = ((X_all_z - Rnp) ** 2).sum(axis=0)
    ss_tot = ((X_all_z - X_all_z.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return z_all.cpu().numpy(), float((1 - ss_res / ss_tot).mean())


def kf_on_latents(Z, y):
    Ztr, Zva = Z[tr], Z[va]
    ytr, yva = y[tr], y[va]
    z_mu = Ztr.mean(axis=0)
    z_sd = Ztr.std(axis=0) + 1e-12
    Ztr = (Ztr - z_mu) / z_sd
    Zva = (Zva - z_mu) / z_sd
    y_mu = ytr.mean(axis=0)
    ytr = ytr - y_mu
    yva = yva - y_mu
    m = KalmanFilter(C=KF_C)
    m.fit(Ztr, ytr)
    yp = m.predict(Zva, yva)
    return dict(yva=yva, yp=yp, R2=r2_per_output(yva, yp), rho=rho_per_output(yva, yp))


# ===== Sweep ==============================================================
results = {}
for ld in LATENT_DIMS:
    t0 = time.time()
    if ld == 52:
        Z = X_z.copy()
        recon = 1.0
    else:
        Z, recon = train_and_encode(X_z[tr], X_z, ld)
    out = kf_on_latents(Z, y_full)
    out["recon_R2"] = recon
    out["seconds"] = time.time() - t0
    results[ld] = out
    print(
        f"latent_dim={ld:>3}  R2_vx={out['R2'][2]:+.3f}  R2_vy={out['R2'][3]:+.3f}  "
        f"reconR2={recon:.3f}  ({out['seconds']:.1f}s)"
    )


# ===== Plotting ===========================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def shade_chunks(ax, err, chunk=800):
    m = err.mean()
    for s in range(0, len(err), chunk):
        e = min(s + chunk, len(err))
        ax.axvspan(s, e, color="lightgreen" if err[s:e].mean() < m else "lightcoral", alpha=0.4)


def fig1():
    o = results[52]
    t = np.arange(len(o["yva"]))
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for ax, idx in zip(axes, [2, 3]):
        ax.plot(t, o["yva"][:, idx], "C0", lw=0.6, label=f"True Output (idx {idx})")
        ax.plot(t, o["yp"][:, idx], "C1", lw=0.6, alpha=0.85, label=f"Predicted Output (idx {idx})")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"Kalman Filter Prediction (Output Index {idx})")
    axes[1].set_xlabel("Time bins")
    fig.suptitle("Qualitative Prediction Check")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig1_qualitative_check.png"), dpi=110)
    plt.close()


def fig2():
    o = results[52]
    t = np.arange(len(o["yva"]))
    err = np.abs(o["yva"][:, 2] - o["yp"][:, 2])
    fig, ax = plt.subplots(figsize=(13, 3.5))
    shade_chunks(ax, err)
    ax.plot(t, o["yva"][:, 2], "C0", lw=0.6)
    ax.plot(t, o["yp"][:, 2], "C1", lw=0.6, alpha=0.85)
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Value")
    ax.set_title("Kalman Filter Prediction (Output Index 2)")
    handles = [
        plt.Line2D([], [], color="C0", label="True Output (idx 2)"),
        plt.Line2D([], [], color="C1", label="Predicted Output (idx 2)"),
        Patch(facecolor="lightgreen", alpha=0.4, label="Good prediction (MAE < mean)"),
        Patch(facecolor="lightcoral", alpha=0.4, label="Poor prediction (MAE > mean)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig2_region_check.png"), dpi=110)
    plt.close()


def fig3():
    o = results[52]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, idx, name in zip(
        axes, [2, 3], ["x-velocity (output 2)", "y-velocity (output 3)"]
    ):
        err = o["yva"][:, idx] - o["yp"][:, idx]
        m, sd = err.mean(), err.std()
        ax.hist(err, bins=30, color="seagreen", edgecolor="white")
        lo, hi = ax.get_xlim()
        ax.axvspan(lo, m - sd, color="lightcoral", alpha=0.3, label="Tails")
        ax.axvspan(m + sd, hi, color="lightcoral", alpha=0.3)
        ax.axvspan(m - sd, m + sd, color="lightgreen", alpha=0.4, label=f"±1 std  [{m-sd:.1f}, {m+sd:.1f}]")
        ax.axvline(m, color="black", linestyle="--")
        ax.set_title(f"Prediction Error Histogram ({name})")
        ax.set_xlabel(f"Prediction Error ({name})")
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle("Prediction Error Histograms — Kalman Filter", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig3_error_histograms.png"), dpi=110)
    plt.close()


def fig4():
    lds = LATENT_DIMS
    em_x = [(results[ld]["yva"][:, 2] - results[ld]["yp"][:, 2]).mean() for ld in lds]
    em_y = [(results[ld]["yva"][:, 3] - results[ld]["yp"][:, 3]).mean() for ld in lds]
    es_x = [(results[ld]["yva"][:, 2] - results[ld]["yp"][:, 2]).std() for ld in lds]
    es_y = [(results[ld]["yva"][:, 3] - results[ld]["yp"][:, 3]).std() for ld in lds]
    r2x = [results[ld]["R2"][2] for ld in lds]
    r2y = [results[ld]["R2"][3] for ld in lds]
    rh2x = [results[ld]["rho"][2] ** 2 for ld in lds]
    rh2y = [results[ld]["rho"][3] ** 2 for ld in lds]
    rec = [results[ld]["recon_R2"] for ld in lds]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes[2, 1].axis("off")
    panels = [
        (axes[0, 0], "error_mean", em_x, em_y),
        (axes[0, 1], "error_std", es_x, es_y),
        (axes[1, 0], "r2", r2x, r2y),
        (axes[1, 1], "rho2", rh2x, rh2y),
    ]
    for ax, t, vx, vy in panels:
        ax.plot(lds, vx, "o-", color="C0", label="x-velocity")
        ax.plot(lds, vy, "o-", color="C1", label="y-velocity")
        ax.scatter([52, 52], [vx[-1], vy[-1]], s=200, edgecolor="red", facecolor="none", lw=2)
        ax.set_title(t)
        ax.set_xlabel("latent_dim")
        ax.set_ylabel(t)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(lds)
    ax = axes[2, 0]
    ax.plot(lds, rec, "o-", color="seagreen", label="mean R² across neurons")
    ax.scatter([52], [rec[-1]], s=200, edgecolor="red", facecolor="none", lw=2)
    ax.set_title("Autoencoder Reconstruction R² (X vs original X, averaged over neurons)")
    ax.set_xlabel("latent_dim")
    ax.set_ylabel("reconstruction R²")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(lds)
    fig.suptitle("Results: Kalman Decoding Across Latent Dimensions")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig4_results_grid.png"), dpi=110)
    plt.close()


def fig5():
    fig, ax = plt.subplots(figsize=(8, 6))
    for ld in LATENT_DIMS:
        cf = 52 / ld
        ax.scatter(ld, cf, s=120, color="navy")
        ax.annotate(f"{cf:.1f}x", (ld, cf), xytext=(8, 3), textcoords="offset points")
    ax.set_xlabel("latent_dim")
    ax.set_ylabel("compression factor")
    ax.set_title("Compression Factor vs Latent Dim")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig5_compression.png"), dpi=110)
    plt.close()


def fig6():
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for ax, ld, label in zip(axes, [4, 52], ["4d (most compressed)", "52d (uncompressed)"]):
        o = results[ld]
        t = np.arange(len(o["yva"]))
        err = np.abs(o["yva"][:, 3] - o["yp"][:, 3])
        shade_chunks(ax, err)
        ax.plot(t, o["yva"][:, 3], "C0", lw=0.6, label="True Output (idx 3)")
        ax.plot(t, o["yp"][:, 3], "C1", lw=0.6, alpha=0.85, label="Predicted Output (idx 3)")
        ax.set_title(f"Kalman Filter Prediction — {label}")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel("Time bins")
    fig.suptitle("Side-by-Side Comparison (4n vs 52n)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig6_sidebyside.png"), dpi=110)
    plt.close()


for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
    fn()
    print(f"wrote {fn.__name__}")


summary = {
    str(ld): dict(
        R2_vx=float(results[ld]["R2"][2]),
        R2_vy=float(results[ld]["R2"][3]),
        rho2_vx=float(results[ld]["rho"][2] ** 2),
        rho2_vy=float(results[ld]["rho"][3] ** 2),
        recon_R2=float(results[ld]["recon_R2"]),
        seconds=float(results[ld]["seconds"]),
    )
    for ld in LATENT_DIMS
}
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("done")
