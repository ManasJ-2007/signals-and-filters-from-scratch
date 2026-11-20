# EE-229 Assignment 1, Question 3
# Reconstructing x(t) from its samples with a finite sum of sincs

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def x_of_t(t):
    return (0.6 * np.cos(2 * np.pi * 120 * t)
            + 0.3 * np.cos(2 * np.pi * 300 * t + np.pi / 4)
            + 0.1 * np.cos(2 * np.pi * 600 * t - np.pi / 3))


def my_sinc(u):
    # sinc(u) = sin(pi u)/(pi u), and 1 at u = 0
    out = np.ones(len(u))
    nz = u != 0
    out[nz] = np.sin(np.pi * u[nz]) / (np.pi * u[nz])
    return out


def sinc_reconstruct(t, fs, L):
    xhat = np.zeros(len(t))
    for i in range(len(t)):
        q = int(np.floor(fs * t[i] + 0.5))       # nearest sample index
        k = np.arange(q - L, q + L + 1)          # 2L+1 sample indices
        samples = x_of_t(k / fs)                 # taken from the formula, even for t < 0 or t > 0.1 s
        xhat[i] = np.sum(samples * my_sinc(fs * t[i] - k))
    return xhat


t = np.arange(1001) / 40000      # 0 to 0.025 s
x_true = x_of_t(t)


def make_plots(fs, L_list, name):
    results = []
    recon = []
    for L in L_list:
        xh = sinc_reconstruct(t, fs, L)
        rmse = np.sqrt(np.sum((xh - x_true) ** 2) / 1001)
        # separate the error at sample instants from the error between them
        at_samples = np.abs((t * fs) - np.round(t * fs)) < 1e-9
        err_at = np.max(np.abs(xh[at_samples] - x_true[at_samples]))
        err_between = np.max(np.abs(xh[~at_samples] - x_true[~at_samples]))
        results.append((L, 2 * L + 1, rmse, err_at, err_between))
        recon.append(xh)

    n_plots = len(L_list)
    for view in ["full", "zoom"]:
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.4 * n_plots + 0.4),
                                 sharex=True, sharey=True, squeeze=False)
        for j in range(n_plots):
            ax = axes[j, 0]
            ax.plot(t, x_true, "k-", linewidth=1, label="x(t) (known)")
            ax.plot(t, recon[j], "k--", linewidth=1, label=f"reconstruction, L = {L_list[j]}")
            ks = np.arange(0, int(0.025 * fs) + 1)
            ax.plot(ks / fs, x_of_t(ks / fs), "ko", markersize=3, markerfacecolor="none",
                    label=f"samples (fs = {fs} Hz)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, color="k", linestyle=":", linewidth=0.4)
            ax.legend(loc="upper right", fontsize=7)
            if view == "full":
                ax.set_xlim(0, 0.025)
            else:
                ax.set_xlim(0.010, 0.012)
        axes[-1, 0].set_xlabel("Time (s)")
        title_view = "0 to 0.025 s" if view == "full" else "close view 0.010 to 0.012 s"
        fig.suptitle(f"Q3: sinc reconstruction, fs = {fs} Hz, {title_view}")
        fig.tight_layout()
        fig.savefig(f"figures/q3_{name}_{view}.png", dpi=150)
        plt.close(fig)

    print(f"\nfs = {fs} Hz")
    print("  L   terms      RMSE      max err at samples   max err between samples")
    for r in results:
        print(f"{r[0]:4d}  {r[1]:5d}   {r[2]:.6f}   {r[3]:.2e}             {r[4]:.6f}")


fs = 4000
print("Q3(a): fs =", fs, "Hz, 2B = 1200 Hz, so fs > 2B")
make_plots(4000, [2, 8, 32, 128], "fs4000")

print("\nQ3(c): fs = 1000 Hz < 2B = 1200 Hz")
make_plots(1000, [128], "fs1000")

# what the fs = 1000 Hz samples actually describe: 600 Hz is replaced by 400 Hz
x_alias = (0.6 * np.cos(2 * np.pi * 120 * t)
           + 0.3 * np.cos(2 * np.pi * 300 * t + np.pi / 4)
           + 0.1 * np.cos(2 * np.pi * 400 * t + np.pi / 3))
xh = sinc_reconstruct(t, 1000, 128)
print("RMSE between reconstruction and the aliased signal (400 Hz version):",
      np.sqrt(np.sum((xh - x_alias) ** 2) / 1001))
