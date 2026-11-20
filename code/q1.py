# EE-229 Assignment 1, Question 1
# Bandlimited signal, sampling and DFT
# numpy is used only for arrays and elementary functions, matplotlib for plots

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def x_of_t(t):
    # the given signal
    return (0.6 * np.cos(2 * np.pi * 120 * t)
            + 0.3 * np.cos(2 * np.pi * 300 * t + np.pi / 4)
            + 0.1 * np.cos(2 * np.pi * 600 * t - np.pi / 3))


def my_dft(x):
    # X[k] = sum over n of x[n] (cos(2 pi k n / N) - j sin(2 pi k n / N))
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        theta = 2 * np.pi * k * n / N
        X[k] = np.sum(x * np.cos(theta)) - 1j * np.sum(x * np.sin(theta))
    return X


def dft_freqs(N, fs):
    # frequency of each bin, negative frequencies for k >= N/2
    f = np.zeros(N)
    for k in range(N):
        if k < N / 2:
            f[k] = k * fs / N
        else:
            f[k] = (k - N) * fs / N
    return f


# ---------------- part (a) ----------------
freq = [-600, -300, -120, 120, 300, 600]
mag = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]
phase = [np.pi / 3, -np.pi / 4, 0, 0, np.pi / 4, -np.pi / 3]
phase_txt = ["+pi/3", "-pi/4", "0", "0", "+pi/4", "-pi/3"]

print("Q1(a) frequency components")
print("f (Hz)   magnitude   phase (rad)")
for i in range(6):
    print(f"{freq[i]:6d}   {mag[i]:9.2f}   {phase[i]:8.4f}  ({phase_txt[i]})")
print("Smallest bandlimit B = 600 Hz")

plt.figure(figsize=(8, 4))
plt.stem(freq, mag, linefmt="k-", markerfmt="ko", basefmt="k-")
for i in range(6):
    plt.text(freq[i] + 15, mag[i] + 0.01, "phase " + phase_txt[i], fontsize=9)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Q1(a): two-sided magnitude line spectrum of x(t)")
plt.xlim(-750, 750)
plt.ylim(0, 0.36)
plt.grid(True, color="k", linestyle=":", linewidth=0.4)
plt.tight_layout()
plt.savefig("figures/q1a_line_spectrum.png", dpi=150)
plt.close()


# ---------------- parts (b) and (c) ----------------
def sample_and_analyse(fs, N, part):
    T = 1 / fs
    n = np.arange(N)
    xn = x_of_t(n * T)

    # fine reference grid, 0 to 0.025 s
    t_fine = np.arange(1001) / 40000
    x_fine = x_of_t(t_fine)

    # samples that fall inside 0 <= t <= 0.025 s
    show = n * T <= 0.025

    plt.figure(figsize=(8, 4))
    plt.plot(t_fine, x_fine, "k-", linewidth=1, label="x(t) on 40 kHz grid")
    plt.plot(n[show] * T, xn[show], "ko", markersize=4, markerfacecolor="none",
             label=f"samples x[n], fs = {fs} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Q1({part}): x(t) and its samples at fs = {fs} Hz")
    plt.xlim(0, 0.025)
    plt.grid(True, color="k", linestyle=":", linewidth=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"figures/q1{part}_samples_fs{fs}.png", dpi=150)
    plt.close()

    # DFT
    X = my_dft(xn)
    f = dft_freqs(N, fs)
    order = np.argsort(f)          # put the frequencies in increasing order
    f_sorted = f[order]
    M_sorted = np.abs(X[order]) / N

    plt.figure(figsize=(8, 4))
    plt.stem(f_sorted, M_sorted, linefmt="k-", markerfmt="k.", basefmt="k-")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X[k]| / N")
    plt.title(f"Q1({part}): DFT magnitude, fs = {fs} Hz, N = {N}")
    plt.xlim(-fs / 2, fs / 2)
    plt.ylim(0, 0.35)
    plt.grid(True, color="k", linestyle=":", linewidth=0.4)
    plt.tight_layout()
    plt.savefig(f"figures/q1{part}_dft_fs{fs}.png", dpi=150)
    plt.close()

    # six largest bins
    big = np.argsort(M_sorted)[::-1][:6]
    big = big[np.argsort(f_sorted[big])]
    print(f"\nQ1({part}) fs = {fs} Hz, N = {N}, delta f = {fs / N} Hz")
    print("peak f (Hz)   |X[k]|/N    phase of X[k] (rad)")
    for i in big:
        k = order[i]
        ph = np.arctan2(X[k].imag, X[k].real)
        print(f"{f_sorted[i]:10.1f}   {M_sorted[i]:8.4f}    {ph:8.4f}")
    others = np.delete(M_sorted, big)
    print("largest of the remaining bins:", others.max())


sample_and_analyse(4000, 400, "b")
sample_and_analyse(1000, 100, "c")

# aliasing check for part (c)
fs = 1000
for f0 in [120, 300, 600]:
    l = int(np.floor((f0 + fs / 2) / fs))   # integer that brings f0 into [-fs/2, fs/2)
    print(f"f0 = {f0} Hz -> l = {l}, f_alias = {f0 - l * fs} Hz")
