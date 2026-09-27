"""image_reject_rx.py

Receive-side **image-reject downconverters**, the dual of the phasing SSB
exciter:

* **Hartley** — quadrature mixer + the all-pass 90-degree network of this repo.
  A real RF input at f_LO +- f_IF is mixed by cos/sin(w_LO); the wanted (upper)
  and image (lower) both land at audio f_IF but with opposite Q sign.  Passing
  I, Q through the two all-pass legs (90 deg apart) and adding cancels the image
  and keeps the wanted:

      out_wanted[n] = A0{I}[n] + A1{Q}[n]      (A0, A1 = the all-pass legs)

  The image rejection equals the **all-pass IRR** — exactly the phasing figure.

* **Weaver** ("third method") — two quadrature mixing stages and low-pass
  filters, *no* broadband 90-degree network.  The second quadrature mix moves
  wanted and image to different frequencies (w_LO2 -+ f_IF); a final low-pass
  keeps the wanted and drops the image.  Rejection is set by the LPF/quadrature
  balance, not an all-pass, but it has a *secondary image* and degrades for
  tones near the band edge (where wanted and image sit close to the cut-off).

Run:  python image_reject_rx.py
"""

import numpy as np
from scipy.signal import lfilter, firwin

import digital_phshift as dp


def _lpf(x, fc, fs, ntaps=257):
    h = firwin(ntaps, fc / (fs / 2))
    return lfilter(h, 1.0, x)


def _allpass_leg(x, c_col):
    y = np.asarray(x, dtype=float)
    for c in np.atleast_1d(c_col):
        y = lfilter([-c, 1.0], [1.0, -c], y)
    return y


# ---------------------------------------------------------------------------
#  Hartley image-reject downconverter (uses the all-pass 90-degree network)
# ---------------------------------------------------------------------------
def hartley_rx(x, f_lo, fs, c, fcut, s=+1):
    n = np.arange(x.size)
    w = 2 * np.pi * f_lo / fs
    I = _lpf(x * np.cos(w * n), fcut, fs)
    Q = _lpf(x * np.sin(w * n), fcut, fs)
    return _allpass_leg(I, c[:, 0]) + s * _allpass_leg(Q, c[:, 1])


# ---------------------------------------------------------------------------
#  Weaver ("third method") image-reject downconverter (no broadband 90 deg)
# ---------------------------------------------------------------------------
def weaver_rx(x, f_lo, fs, W, s=+1):
    n = np.arange(x.size)
    w1 = 2 * np.pi * f_lo / fs
    w2 = 2 * np.pi * W / fs                       # 2nd LO at the band edge
    I1 = _lpf(x * np.cos(w1 * n), W, fs)          # LPF1 limits to [0, W]
    Q1 = _lpf(x * np.sin(w1 * n), W, fs)
    mix2 = I1 * np.cos(w2 * n) + s * Q1 * np.sin(w2 * n)
    return _lpf(mix2, W, fs)                       # LPF2 drops the image (> W)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _tone_mag(sig, bin_k, Nfft, skip):
    X = np.fft.rfft(sig[skip:skip + Nfft])
    return np.abs(X[bin_k])


def main():
    fs = 48000.0
    f_lo = 10000.0
    fa, fb = 300.0, 3400.0
    fcut = 3800.0
    N = 4

    d = dp.design(90, N, fa, fb, fs, pnorm=40, method="digital", verbose=False)
    c = d["c"]

    # orient Hartley sign so the UPPER RF (f_lo+f) is the wanted one
    Nfft = 1 << 15
    df = fs / Nfft
    skip = 4096
    ntot = skip + Nfft
    n = np.arange(ntot)

    def rf_tone(freq):
        k = int(round(freq / df))
        return np.cos(2 * np.pi * (k * df) * n / fs)

    f_probe = 1500.0
    up = hartley_rx(rf_tone(f_lo + f_probe), f_lo, fs, c, fcut, +1)
    lo = hartley_rx(rf_tone(f_lo - f_probe), f_lo, fs, c, fcut, +1)
    kp = int(round(f_probe / df))
    s = +1 if _tone_mag(up, kp, Nfft, skip) >= _tone_mag(lo, kp, Nfft, skip) else -1

    # orient Weaver sign so the wanted (upper RF) lands in-band at W - f
    kpw = int(round((fb - f_probe) / df))
    wp = _tone_mag(weaver_rx(rf_tone(f_lo + f_probe), f_lo, fs, fb, +1), kpw, Nfft, skip)
    wm = _tone_mag(weaver_rx(rf_tone(f_lo + f_probe), f_lo, fs, fb, -1), kpw, Nfft, skip)
    sw = +1 if wp >= wm else -1

    # --- rejection vs IF: wanted-only vs image-only ------------------------
    ifreqs = np.arange(400.0, 3401.0, 150.0)
    rej_hartley, rej_weaver = [], []
    for f in ifreqs:
        k = int(round(f / df))
        # Hartley
        w_up = _tone_mag(hartley_rx(rf_tone(f_lo + f), f_lo, fs, c, fcut, s), k, Nfft, skip)
        w_lo = _tone_mag(hartley_rx(rf_tone(f_lo - f), f_lo, fs, c, fcut, s), k, Nfft, skip)
        rej_hartley.append(20 * np.log10(w_up / max(w_lo, 1e-15)))
        # Weaver: wanted lands at W-f, image would land at W+f (removed)
        kw = int(round((fb - f) / df))
        ww_up = _tone_mag(weaver_rx(rf_tone(f_lo + f), f_lo, fs, fb, sw), kw, Nfft, skip)
        ww_lo = _tone_mag(weaver_rx(rf_tone(f_lo - f), f_lo, fs, fb, sw), kw, Nfft, skip)
        rej_weaver.append(20 * np.log10(ww_up / max(ww_lo, 1e-15)))
    rej_hartley = np.array(rej_hartley)
    rej_weaver = np.array(rej_weaver)

    # theoretical all-pass IRR
    ph = dp.phfunc(d["params"], ifreqs, fs)
    irr_theo = -20 * np.log10(np.abs(np.tan(np.radians(ph[1] - ph[0] - 90) / 2)))

    print("Image-reject downconverter (fs=%g, f_LO=%g, IF band %g-%g Hz, Hartley N=%d)\n"
          % (fs, f_lo, fa, fb, N))
    print("  Hartley worst-case image rejection = %6.1f dB  (all-pass IRR %.1f dB)"
          % (rej_hartley.min(), irr_theo.min()))
    print("  Weaver  worst-case image rejection = %6.1f dB  (LPF/quadrature limited)"
          % rej_weaver.min())

    # --- illustrative spectra: interleaved wanted / image combs -----------
    f_want = np.arange(500.0, 3300.0, 400.0)     # wanted RF at f_lo + f_want
    f_img = np.arange(700.0, 3300.0, 400.0)      # image  RF at f_lo - f_img
    xin = np.zeros(ntot)
    for f in f_want:
        xin += rf_tone(f_lo + f)
    for f in f_img:
        xin += rf_tone(f_lo - f)

    outH = hartley_rx(xin, f_lo, fs, c, fcut, s)
    outW = weaver_rx(xin, f_lo, fs, fb, sw)

    def spec(sig, real_in=False):
        X = np.fft.rfft(sig[skip:skip + Nfft] * np.hanning(Nfft))
        m = 20 * np.log10(np.maximum(np.abs(X), 1e-9)); return m - m.max()
    fax = np.fft.rfftfreq(Nfft, 1 / fs)

    _plots(fs, f_lo, fa, fb, fax, spec(xin), spec(outH), spec(outW),
           ifreqs, rej_hartley, rej_weaver, irr_theo, N, f_want, f_img)


def _plots(fs, f_lo, fa, fb, fax, sin_db, hart_db, weav_db,
           ifreqs, rejH, rejW, irr_theo, N, f_want, f_img):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) RF input around the LO
    ax[0, 0].plot(fax / 1000, sin_db, color="C0", lw=0.8)
    ax[0, 0].axvspan((f_lo + fa) / 1000, (f_lo + fb) / 1000, color="C2", alpha=0.12,
                     label="wanted band (f_LO+IF)")
    ax[0, 0].axvspan((f_lo - fb) / 1000, (f_lo - fa) / 1000, color="C3", alpha=0.12,
                     label="image band (f_LO−IF)")
    ax[0, 0].axvline(f_lo / 1000, color="0.5", ls="--", lw=0.8, label="f_LO")
    ax[0, 0].set_xlim(5, 15); ax[0, 0].set_ylim(-80, 5)
    ax[0, 0].set_title("RF input: wanted + image around f_LO")
    ax[0, 0].set_xlabel("Frequency [kHz]"); ax[0, 0].set_ylabel("Level [dB]")
    ax[0, 0].grid(True, alpha=0.4); ax[0, 0].legend(fontsize=8)

    # (b) Hartley audio output: wanted comb strong, image comb suppressed
    ax[0, 1].plot(fax, hart_db, color="C0", lw=0.8)
    for f in f_want:
        ax[0, 1].axvline(f, color="C2", lw=0.6, alpha=0.5)
    for f in f_img:
        ax[0, 1].axvline(f, color="C3", lw=0.6, alpha=0.5, ls=":")
    ax[0, 1].set_xlim(0, 3800); ax[0, 1].set_ylim(-110, 5)
    ax[0, 1].set_title("Hartley audio out — wanted (green) kept, image (red) rejected")
    ax[0, 1].set_xlabel("Audio frequency [Hz]"); ax[0, 1].set_ylabel("Level [dB]")
    ax[0, 1].grid(True, alpha=0.4)

    # (c) rejection vs IF
    ax[1, 0].plot(ifreqs, irr_theo, color="C0", lw=1.0, label="all-pass IRR (theory)")
    ax[1, 0].plot(ifreqs, rejH, "o", color="C0", ms=4, label="Hartley (measured)")
    ax[1, 0].plot(ifreqs, rejW, "s-", color="C4", ms=4, label="Weaver (measured)")
    ax[1, 0].set_ylim(0, max(140, np.nanmax(rejW) + 10, np.nanmax(irr_theo) + 5))
    ax[1, 0].set_title("Image rejection vs IF frequency")
    ax[1, 0].set_xlabel("IF (audio) frequency [Hz]"); ax[1, 0].set_ylabel("Rejection [dB]")
    ax[1, 0].grid(True, alpha=0.4); ax[1, 0].legend(fontsize=8)

    # (d) Weaver audio output (frequency-flipped baseband)
    ax[1, 1].plot(fax, weav_db, color="C4", lw=0.8)
    ax[1, 1].set_xlim(0, 3800); ax[1, 1].set_ylim(-110, 5)
    ax[1, 1].set_title("Weaver audio out — wanted kept, image dropped by LPF")
    ax[1, 1].set_xlabel("Audio frequency [Hz]"); ax[1, 1].set_ylabel("Level [dB]")
    ax[1, 1].grid(True, alpha=0.4)

    fig.suptitle("Image-reject downconverters — Hartley (all-pass 90°) vs Weaver "
                 "(N=%d/leg)" % N, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "image_reject_rx.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
