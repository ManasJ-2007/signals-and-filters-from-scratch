# EE-229 Assignment 1, Question 2
# Fan-noise reduction with an FIR filter
#
# Input files, in the q2_audio folder (already converted to mono, 16 kHz,
# 16-bit, exactly 5 s):
#   speech_input.wav  - my own speech
#   fan_input.wav     - fan recording
# The conversion and the 5 s cut were done before this script with ffmpeg
# (see the report). The output audio is also saved in q2_audio.

import os
import wave
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
fs = 16000

# folder with the Q2 audio files (next to the code folder)
audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q2_audio")


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


def frame_dft(q):
    # Q[k] = sum_{r=0}^{511} q[r] e^{-j 2 pi k r / 512}, k = 0..256
    R = len(q)
    r = np.arange(R)
    Q = np.zeros(R // 2 + 1, dtype=complex)
    for k in range(R // 2 + 1):
        theta = 2 * np.pi * k * r / R
        Q[k] = np.sum(q * np.cos(theta)) - 1j * np.sum(q * np.sin(theta))
    return Q


def freq_response(h, f):
    # H(f) = sum_m h[m] e^{-j 2 pi f m / fs}
    m = np.arange(len(h))
    H = np.zeros(len(f), dtype=complex)
    for i in range(len(f)):
        theta = 2 * np.pi * f[i] * m / fs
        H[i] = np.sum(h * np.cos(theta)) - 1j * np.sum(h * np.sin(theta))
    return H


def my_conv(h, y):
    # z[n] = sum_m h[m] y[n-m], n = 0 .. N+M-2, with y = 0 outside the record.
    # Same sum, but done one m at a time: add h[m] times y shifted by m.
    M = len(h)
    N = len(y)
    z = np.zeros(N + M - 1)
    for m in range(M):
        z[m:m + N] = z[m:m + N] + h[m] * y
    return z


s = read_wav(os.path.join(audio_dir, "speech_input.wav"))
v = read_wav(os.path.join(audio_dir, "fan_input.wav"))
N = len(s)
n = np.arange(N)
t = n / fs
print("N =", N, len(v))

# ---------------- part (a) ----------------
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(t, s, "k-", linewidth=0.5)
ax[0].set_title("Q2(a): speech s[n], fs = 16000 Hz")
ax[0].set_ylabel("Amplitude")
ax[1].plot(t, v, "k-", linewidth=0.5)
ax[1].set_title("Q2(a): fan v[n], fs = 16000 Hz")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("Time (s)")
ax[1].set_xlim(0, 5)
for a in ax:
    a.grid(True, color="k", linestyle=":", linewidth=0.4)
fig.tight_layout()
fig.savefig("figures/q2a_signals.png", dpi=150)
plt.close(fig)

speech_start = 3.70     # s, inside a loud vowel
fan_start = 1.00        # s
qs = s[int(speech_start * fs): int(speech_start * fs) + 512]
qv = v[int(fan_start * fs): int(fan_start * fs) + 512]
Qs = np.abs(frame_dft(qs)) / 512
Qv = np.abs(frame_dft(qv)) / 512
fk = np.arange(257) * fs / 512

fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
ax[0].semilogy(fk, Qs, "k-", linewidth=0.8)
ax[0].set_title(f"Q2(a): speech frame, starts at {speech_start:.2f} s (512 samples)")
ax[1].semilogy(fk, Qv, "k-", linewidth=0.8)
ax[1].set_title(f"Q2(a): fan frame, starts at {fan_start:.2f} s (512 samples)")
for a in ax:
    a.set_ylabel("|Q[k]|/512 (log scale)")
    a.set_ylim(1e-6, 1e-1)
    a.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim(0, 8000)
fig.tight_layout()
fig.savefig("figures/q2a_frame_spectra.png", dpi=150)
plt.close(fig)

print("\nQ2(a) strongest bins")
print("speech frame:", [(fk[i], round(Qs[i], 5)) for i in np.argsort(Qs)[::-1][:6]])
print("fan frame:   ", [(fk[i], round(Qv[i], 5)) for i in np.argsort(Qv)[::-1][:4]])

# the 512 frame only has 31.25 Hz spacing, so look at the fan peak more finely
# using the first 1 s of the fan record and a 1 Hz step
seg = v[:16000]
ff = np.arange(200, 301, 1.0)
scan = np.abs(freq_response(seg, ff)) / 16000      # same sum as H(f) with h = seg
f_fan = ff[np.argmax(scan)]
print("fan peak from 1 Hz scan:", f_fan, "Hz")

# share of energy near the fan peak (from the 512 frames, k*31.25 Hz)
near = (fk >= 200) & (fk <= 300)
print("fan frame energy in 200-300 Hz: %.1f %%" % (100 * np.sum(Qv[near] ** 2) / np.sum(Qv ** 2)))
print("speech frame energy in 200-300 Hz: %.1f %%" % (100 * np.sum(Qs[near] ** 2) / np.sum(Qs ** 2)))

# ---------------- part (b) ----------------
Es = np.sum(s ** 2)
Ev = np.sum(v ** 2)
alpha = np.sqrt(Es / (10 ** (5 / 10) * Ev))
noise = alpha * v
y = s + noise
snr_in = 10 * np.log10(np.sum(s ** 2) / np.sum(noise ** 2))
print("\nQ2(b) Es = %.4f, Ev = %.4f, alpha = %.4f, SNR_in = %.4f dB" % (Es, Ev, alpha, snr_in))

t1, t2 = 1.00, 1.10
w = (t >= t1) & (t < t2)
fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, sharey=True)
ax[0].plot(t[w], s[w], "k-", linewidth=0.8)
ax[0].set_title("clean speech s[n]")
ax[1].plot(t[w], noise[w], "k-", linewidth=0.8)
ax[1].set_title("scaled fan noise alpha v[n]")
ax[2].plot(t[w], y[w], "k-", linewidth=0.8)
ax[2].set_title("noisy speech y[n] = s[n] + alpha v[n]")
for a in ax:
    a.set_ylabel("Amplitude")
    a.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[2].set_xlabel("Time (s)")
ax[2].set_xlim(t1, t2)
fig.suptitle("Q2(b): 1.00 to 1.10 s, fs = 16000 Hz")
fig.tight_layout()
fig.savefig("figures/q2b_signals.png", dpi=150)
plt.close(fig)

# ---------------- part (c) ----------------
def highpass(M):
    # centre sample minus the local average of M samples
    D = (M - 1) // 2
    h = -np.ones(M) / M
    h[D] = h[D] + 1
    return h


def bandstop(M, f0):
    # centre sample minus a "local average" weighted by a cosine at f0.
    # The cosine-weighted average keeps only frequencies near f0,
    # so subtracting it removes a narrow band around f0.
    D = (M - 1) // 2
    m = np.arange(M)
    h = -(2 / M) * np.cos(2 * np.pi * f0 * (m - D) / fs)
    h[D] = h[D] + 1
    return h


f0 = f_fan
choices = [("highpass, M = 21", highpass(21)),
           ("highpass, M = 41", highpass(41)),
           ("band-stop at %.0f Hz, M = 401" % f0, bandstop(401, f0)),
           ("band-stop at %.0f Hz, M = 801" % f0, bandstop(801, f0))]

fgrid = np.arange(0, 8001, 20)
tries_out = []
fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
ax = ax.flatten()
print("\nQ2(c) filter tries")
print("filter                         |H| at 240   |H| at 500   |H| at 1000   delta SNR (dB)")
for i, (name, h) in enumerate(choices):
    M = len(h)
    D = (M - 1) // 2
    # check symmetry h[m] = h[M-1-m]
    assert np.max(np.abs(h - h[::-1])) < 1e-12
    H = np.abs(freq_response(h, fgrid))
    ax[i].plot(fgrid, H, "k-", linewidth=0.8)
    ax[i].set_title(name)
    ax[i].set_xlim(0, 8000)
    ax[i].set_ylim(0, 1.4)
    ax[i].grid(True, color="k", linestyle=":", linewidth=0.4)
    ax[i].set_xlabel("Frequency (Hz)")
    ax[i].set_ylabel("|H(f)|")

    # quick SNR check for every try (same formulas as part d)
    z = my_conv(h, y)
    s_hat = z[D:D + N]
    I = np.arange(D, N - D)
    si = 10 * np.log10(np.sum(s[I] ** 2) / np.sum(noise[I] ** 2))
    so = 10 * np.log10(np.sum(s[I] ** 2) / np.sum((s[I] - s_hat[I]) ** 2))
    g = np.abs(freq_response(h, np.array([240.0, 500.0, 1000.0])))
    print(f"{name:30s} {g[0]:8.3f}   {g[1]:8.3f}   {g[2]:8.3f}     {so - si:+.2f}")
    tries_out.append(s_hat)
fig.suptitle("Q2(c): |H(f)| of the filters tried, fs = 16000 Hz, 20 Hz grid")
fig.tight_layout()
fig.savefig("figures/q2c_filters.png", dpi=150)
plt.close(fig)

# close view of the final filter near the fan frequency
h = bandstop(801, f0)
M = len(h)
D = (M - 1) // 2
fz = np.arange(0, 1001, 2.0)
Hz = np.abs(freq_response(h, fz))
plt.figure(figsize=(8, 3.5))
plt.plot(fz, Hz, "k-", linewidth=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H(f)|")
plt.title("Q2(c): final filter, close view 0 to 1000 Hz (2 Hz grid)")
plt.xlim(0, 1000)
plt.ylim(0, 1.4)
plt.grid(True, color="k", linestyle=":", linewidth=0.4)
plt.tight_layout()
plt.savefig("figures/q2c_final_zoom.png", dpi=150)
plt.close()

print("\nfinal filter: M =", M, "D =", D, "f0 =", f0)
print("h[D] =", h[D], " h[0] = h[M-1] =", h[0], h[M - 1])
print("first 5 coefficients:", np.round(h[:5], 6))
print("sum of coefficients (gain at 0 Hz):", np.sum(h))
Hf = np.abs(freq_response(h, np.array([0.0, 100, 200, 220, 230, 240, 250, 260, 280, 300, 500, 1000, 4000])))
print("largest gain anywhere (2 Hz grid, 0-1000 Hz): %.3f" % np.max(Hz))
print("gain at 0,100,200,220,230,240,250,260,280,300,500,1000,4000 Hz:", np.round(Hf, 3))

# ---------------- part (d) ----------------
z = my_conv(h, y)                 # length N + M - 1
s_hat = z[D:D + N]
I = np.arange(D, N - D)
snr_in_I = 10 * np.log10(np.sum(s[I] ** 2) / np.sum(noise[I] ** 2))
snr_out = 10 * np.log10(np.sum(s[I] ** 2) / np.sum((s[I] - s_hat[I]) ** 2))
print("\nQ2(d) SNR_in,I = %.2f dB, SNR_out = %.2f dB, delta SNR = %+.2f dB"
      % (snr_in_I, snr_out, snr_out - snr_in_I))

# split the output error into what the filter did to speech and what is left of the fan
zs = my_conv(h, s)[D:D + N]
zv = my_conv(h, noise)[D:D + N]
print("speech change only: %.2f dB,  remaining fan only: %.2f dB"
      % (10 * np.log10(np.sum(s[I] ** 2) / np.sum((s[I] - zs[I]) ** 2)),
         10 * np.log10(np.sum(s[I] ** 2) / np.sum(zv[I] ** 2))))
print("fan energy removed: %.1f dB" % (10 * np.log10(np.sum(noise[I] ** 2) / np.sum(zv[I] ** 2))))

r = y - s_hat
fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, sharey=True)
ax[0].plot(t[w], s[w], "k-", linewidth=0.8)
ax[0].set_title("clean speech s[n]")
ax[1].plot(t[w], y[w], "k-", linewidth=0.8)
ax[1].set_title("noisy speech y[n]")
ax[2].plot(t[w], s_hat[w], "k-", linewidth=0.8)
ax[2].set_title("filtered output s_hat[n] (delay removed)")
for a in ax:
    a.set_ylabel("Amplitude")
    a.grid(True, color="k", linestyle=":", linewidth=0.4)
ax[2].set_xlabel("Time (s)")
ax[2].set_xlim(t1, t2)
fig.suptitle("Q2(d): 1.00 to 1.10 s, fs = 16000 Hz")
fig.tight_layout()
fig.savefig("figures/q2d_compare.png", dpi=150)
plt.close(fig)

plt.figure(figsize=(8, 3.5))
plt.plot(t[w], noise[w], "k-", linewidth=1.2, label="known noise alpha v[n]")
plt.plot(t[w], r[w], "k--", linewidth=0.8, label="removed part r[n] = y[n] - s_hat[n]")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Q2(d): removed component and known noise, 1.00 to 1.10 s")
plt.xlim(t1, t2)
plt.grid(True, color="k", linestyle=":", linewidth=0.4)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/q2d_removed.png", dpi=150)
plt.close()

print("energy of r minus alpha v, relative to alpha v: %.1f dB"
      % (10 * np.log10(np.sum((r[I] - noise[I]) ** 2) / np.sum(noise[I] ** 2))))

# ---------------- audio files ----------------
# SNRs are already computed above. One common scale factor for playback.
peak = max(np.max(np.abs(s)), np.max(np.abs(noise)), np.max(np.abs(y)), np.max(np.abs(s_hat)))
scale = 1.0
if peak > 0.99:
    scale = 0.99 / peak
print("\nplayback scale factor =", scale)
write_wav(os.path.join(audio_dir, "q2_clean_speech.wav"), scale * s)
write_wav(os.path.join(audio_dir, "q2_scaled_fan.wav"), scale * noise)
write_wav(os.path.join(audio_dir, "q2_noisy_speech.wav"), scale * y)
write_wav(os.path.join(audio_dir, "q2_filtered_speech.wav"), scale * s_hat)
# outputs of the other tries, only for listening
write_wav(os.path.join(audio_dir, "q2_try1_highpass21.wav"), scale * tries_out[0])
write_wav(os.path.join(audio_dir, "q2_try2_highpass41.wav"), scale * tries_out[1])
write_wav(os.path.join(audio_dir, "q2_try3_bandstop401.wav"), scale * tries_out[2])
