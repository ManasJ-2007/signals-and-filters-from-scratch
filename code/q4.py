# EE-229 Assignment 1, Question 4
# Phase of a pure delay and a first-order all-pass filter, and their effect on speech
#
# Input: speech_input.wav in the q4_audio folder (my speech, mono, 16 kHz,
# exactly 5 s, same recording as Q2). The output audio is also saved in q4_audio.

import os
import wave
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
fs = 16000

# folder with the Q4 audio files (next to the code folder)
audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q4_audio")

D = 4        # Filter 1 delay in samples
a = 0.7      # Filter 2 all-pass coefficient


def read_wav(name):
    w = wave.open(name, "rb")
    n = w.getnframes()
    data = np.frombuffer(w.readframes(n), dtype=np.int16)
    w.close()
    return data / 32768.0


def write_wav(name, x):
    x = np.clip(x, -1, 32767 / 32768)
    w = wave.open(name, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(fs)
    w.writeframes((x * 32768).astype(np.int16).tobytes())
    w.close()


def my_unwrap(p):
    # add or subtract 2 pi so that neighbouring values differ by at most pi
    out = np.zeros(len(p))
    out[0] = p[0]
    for i in range(1, len(p)):
        d = p[i] - p[i - 1]
        while d > np.pi:
            d = d - 2 * np.pi
        while d < -np.pi:
            d = d + 2 * np.pi
        out[i] = out[i - 1] + d
    return out


# ---------------- part (b): frequency responses ----------------
r = np.arange(513)
w = np.pi * r / 512

# Filter 1: H1 = e^{-j w D} = cos(wD) - j sin(wD)
C1 = np.cos(w * D)
S1 = -np.sin(w * D)

# Filter 2: H2 = (a + e^{-jw}) / (1 + a e^{-jw})
num = (a + np.cos(w)) - 1j * np.sin(w)
den = (1 + a * np.cos(w)) - 1j * a * np.sin(w)
H2 = num / den
C2 = H2.real
S2 = H2.imag

mag1 = np.sqrt(C1 ** 2 + S1 ** 2)
mag2 = np.sqrt(C2 ** 2 + S2 ** 2)
th1 = np.arctan2(S1, C1)
th2 = np.arctan2(S2, C2)
ph1 = my_unwrap(th1)
ph2 = my_unwrap(th2)

print("max | |H1| - 1 | =", np.max(np.abs(mag1 - 1)))
print("max | |H2| - 1 | =", np.max(np.abs(mag2 - 1)))
print("unwrapped phase at w = pi: filter 1 = %.4f rad, filter 2 = %.4f rad" % (ph1[-1], ph2[-1]))

# group delay between neighbouring grid points
tau1 = -(ph1[1:] - ph1[:-1]) / (w[1:] - w[:-1])
tau2 = -(ph2[1:] - ph2[:-1]) / (w[1:] - w[:-1])
w_mid = (w[1:] + w[:-1]) / 2
print("filter 1 delay: min %.4f, max %.4f samples" % (tau1.min(), tau1.max()))
print("filter 2 delay: first %.4f, last %.4f samples" % (tau2[0], tau2[-1]))
print("formula (1-a^2)/(1+2a cos w + a^2) at w=0 and w=pi: %.4f, %.4f"
      % ((1 - a * a) / (1 + a) ** 2, (1 - a * a) / (1 - a) ** 2))
for f_check in [100, 500, 1000, 2000, 4000, 6000, 8000]:
    i = np.argmin(np.abs(w_mid - 2 * np.pi * f_check / fs))
    print("  around %5d Hz: tau2 = %.3f samples = %.3f ms" % (f_check, tau2[i], 1000 * tau2[i] / fs))

f_axis = w * fs / (2 * np.pi)
f_mid = w_mid * fs / (2 * np.pi)

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(w, mag1, "k-", linewidth=1)
ax[0].set_title(f"Filter 1 (pure delay, D = {D}): |H1|")
ax[1].plot(w, mag2, "k-", linewidth=1)
ax[1].set_title(f"Filter 2 (all-pass, a = {a}): |H2|")
for x in ax:
    x.set_ylim(0, 1.5)
    x.set_ylabel("Magnitude")
    x.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[1].set_xlabel("w (rad/sample)   [f = w fs / 2pi, fs = 16000 Hz]")
ax[1].set_xlim(0, np.pi)
fig.tight_layout()
fig.savefig("figures/q4b_magnitude.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
ax[0].plot(w, th1, "k:", linewidth=1, label="atan2 (wrapped)")
ax[0].plot(w, ph1, "k-", linewidth=1, label="unwrapped")
ax[0].set_title(f"Filter 1 phase (D = {D})")
ax[1].plot(w, th2, "k:", linewidth=1, label="atan2 (wrapped)")
ax[1].plot(w, ph2, "k-", linewidth=1, label="unwrapped")
ax[1].set_title(f"Filter 2 phase (a = {a})")
for x in ax:
    x.set_ylabel("Phase (rad)")
    x.grid(True, color="k", linestyle=":", linewidth=0.4)
    x.legend(loc="lower left", fontsize=8)
ax[1].set_xlabel("w (rad/sample)   [f = w fs / 2pi, fs = 16000 Hz]")
ax[1].set_xlim(0, np.pi)
fig.tight_layout()
fig.savefig("figures/q4b_phase.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(w_mid, tau1, "k-", linewidth=1, label=f"Filter 1, pure delay D = {D}")
ax.plot(w_mid, tau2, "k--", linewidth=1, label=f"Filter 2, all-pass a = {a}")
ax.set_xlabel("w (rad/sample)   [f = w fs / 2pi, fs = 16000 Hz]")
ax.set_ylabel("Group delay (samples)")
ax.set_title("Q4(b): group delay estimates (1 sample = 62.5 us)")
ax.set_xlim(0, np.pi)
ax.set_ylim(0, 6)
ax.grid(True, color="k", linestyle=":", linewidth=0.4)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("figures/q4b_group_delay.png", dpi=150)
plt.close(fig)

# ---------------- part (c): apply the filters ----------------
s = read_wav(os.path.join(audio_dir, "speech_input.wav"))
N = len(s)
P = N + 1000
x = np.zeros(P)
x[:N] = s                     # 1000 zeros appended

# Filter 1: u1[n] = s[n - D]
u1 = np.zeros(P)
for n in range(D, P):
    u1[n] = x[n - D]

# Filter 2: u2[n] = a s[n] + s[n-1] - a u2[n-1], with s[-1] = 0, u2[-1] = 0
u2 = np.zeros(P)
s_prev = 0.0
u_prev = 0.0
for n in range(P):
    u2[n] = a * x[n] + s_prev - a * u_prev
    s_prev = x[n]
    u_prev = u2[n]

print("\nenergy: input %.4f, filter 1 %.4f, filter 2 %.4f"
      % (np.sum(x ** 2), np.sum(u1 ** 2), np.sum(u2 ** 2)))
print("tail of filter 2 output |u2| at last sample:", abs(u2[-1]))

t = np.arange(P) / fs
fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, sharey=True)
ax[0].plot(t, x, "k-", linewidth=0.5)
ax[0].set_title("input s[n] (with 1000 zeros appended)")
ax[1].plot(t, u1, "k-", linewidth=0.5)
ax[1].set_title(f"Filter 1 output u1[n], D = {D}")
ax[2].plot(t, u2, "k-", linewidth=0.5)
ax[2].set_title(f"Filter 2 output u2[n], a = {a}")
for z in ax:
    z.set_ylabel("Amplitude")
    z.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[2].set_xlabel("Time (s)")
ax[2].set_xlim(0, 5)
fig.suptitle("Q4(c): complete signals, fs = 16000 Hz")
fig.tight_layout()
fig.savefig("figures/q4c_full.png", dpi=150)
plt.close(fig)

t1, t2 = 1.00, 1.05
n1, n2 = int(t1 * fs), int(t2 * fs)
idx = np.arange(n1, n2)
fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, sharey=True)
ax[0].plot(t[idx], x[idx], "k-", linewidth=0.8)
ax[0].set_title("input s[n]")
ax[1].plot(t[idx], u1[idx], "k-", linewidth=0.8)
ax[1].set_title(f"Filter 1 output u1[n] (not shifted)")
ax[2].plot(t[idx], u2[idx], "k-", linewidth=0.8)
ax[2].set_title(f"Filter 2 output u2[n] (not shifted)")
for z in ax:
    z.set_ylabel("Amplitude")
    z.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[2].set_xlabel("Time (s)")
ax[2].set_xlim(t1, t2)
fig.suptitle("Q4(c): 1.00 to 1.05 s")
fig.tight_layout()
fig.savefig("figures/q4c_zoom.png", dpi=150)
plt.close(fig)

# remove the delay of filter 1
err1 = np.max(np.abs(u1[idx + D] - x[idx]))
print("\nmax |u1[n+D] - s[n]| over 1.00-1.05 s:", err1)

# try shifts 0..20 for filter 2
print("shift  relative error of u2[n+shift] vs s[n] over 1.00-1.05 s")
rel = []
for sh in range(21):
    e = np.sqrt(np.sum((u2[idx + sh] - x[idx]) ** 2) / np.sum(x[idx] ** 2))
    rel.append(e)
    print("%3d    %.3f" % (sh, e))
best = int(np.argmin(rel))
print("best shift:", best, "samples, relative error", round(rel[best], 3))

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
ax[0].plot(t[idx], x[idx], "k-", linewidth=1.2, label="s[n]")
ax[0].plot(t[idx], u1[idx + D], "k--", linewidth=0.8, label=f"u1[n+{D}]")
ax[0].set_title("Filter 1 after removing the delay")
ax[1].plot(t[idx], x[idx], "k-", linewidth=1.2, label="s[n]")
ax[1].plot(t[idx], u2[idx + best], "k--", linewidth=0.8, label=f"u2[n+{best}] (best shift)")
ax[1].set_title("Filter 2 with the best shift from 0 to 20")
for z in ax:
    z.set_ylabel("Amplitude")
    z.grid(True, color="k", linestyle=":", linewidth=0.4)
    z.legend(loc="upper right", fontsize=8)
ax[1].set_xlabel("Time (s)")
ax[1].set_xlim(t1, t2)
fig.suptitle("Q4(c): shifted outputs compared with the input, 1.00 to 1.05 s")
fig.tight_layout()
fig.savefig("figures/q4c_shifted.png", dpi=150)
plt.close(fig)

plt.figure(figsize=(8, 3.5))
plt.plot(np.arange(21), rel, "ko-", markersize=4)
plt.xlabel("Shift (samples)")
plt.ylabel("Relative RMS difference")
plt.title("Q4(c): Filter 2, u2[n+shift] compared with s[n], 1.00 to 1.05 s")
plt.xlim(0, 20)
plt.grid(True, color="k", linestyle=":", linewidth=0.4)
plt.tight_layout()
plt.savefig("figures/q4c_shift_error.png", dpi=150)
plt.close()


# magnitude spectra of the complete zero-padded records
def dft_some_bins(q, k_list):
    Pq = len(q)
    nn = np.arange(Pq)
    Q = np.zeros(len(k_list), dtype=complex)
    for i in range(len(k_list)):
        theta = 2 * np.pi * k_list[i] * nn / Pq
        Q[i] = np.sum(q * np.cos(theta)) - 1j * np.sum(q * np.sin(theta))
    return Q


k_list = np.arange(0, 40501, 100)
fk = k_list * fs / P
X = np.abs(dft_some_bins(x, k_list)) / P
U1 = np.abs(dft_some_bins(u1, k_list)) / P
U2 = np.abs(dft_some_bins(u2, k_list)) / P
print("\nmax spectrum difference: filter 1 %.2e, filter 2 %.2e (largest |X| = %.2e)"
      % (np.max(np.abs(U1 - X)), np.max(np.abs(U2 - X)), np.max(X)))

plt.figure(figsize=(8, 4))
plt.semilogy(fk, X, "k-", linewidth=2.0, label="input s[n]")
plt.semilogy(fk, U1, "k--", linewidth=1.0, label=f"Filter 1 output (D = {D})")
plt.semilogy(fk, U2, "k:", linewidth=1.2, label=f"Filter 2 output (a = {a})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|Q[k]| / P (log scale)")
plt.title("Q4(c): magnitude spectra, P = 81000, every 100th bin")
plt.xlim(0, 8000)
plt.grid(True, color="k", linestyle=":", linewidth=0.4)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/q4c_spectra.png", dpi=150)
plt.close()

# audio, same scale factor for all three
peak = max(np.max(np.abs(x)), np.max(np.abs(u1)), np.max(np.abs(u2)))
scale = 1.0
if peak > 0.99:
    scale = 0.99 / peak
print("\nplayback scale factor =", scale, " peaks:", np.max(np.abs(x)), np.max(np.abs(u2)))
write_wav(os.path.join(audio_dir, "q4_input.wav"), scale * x)
write_wav(os.path.join(audio_dir, "q4_delay_output.wav"), scale * u1)
write_wav(os.path.join(audio_dir, "q4_allpass_output.wav"), scale * u2)
