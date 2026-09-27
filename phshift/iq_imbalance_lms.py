"""iq_imbalance_lms.py

I/Q branch **imbalance model** and **blind adaptive (widely-linear LMS)
correction** -- the real-world limit on image rejection, and how to claw it back.

Why it matters
--------------
The all-pass 90-degree network (and the polyphase / Hartley / Weaver chains built
from it) can reach 80-110 dB of image rejection on paper.  In hardware the floor
is set instead by **gain/phase mismatch between the I and Q branches** (mixers,
ADCs, filters).  With amplitude ratio g and phase error phi the observed complex
baseband is a mix of the wanted analytic signal and its conjugate:

    r[n] = mu * s[n] + nu * s*[n],
    mu = (1 + g e^{+j phi}) / 2,   nu = (1 - g e^{-j phi}) / 2
    IRR = |mu / nu|^2      (e.g. 1 dB + 5 deg  ->  ~26 dB, no matter the filter)

Blind correction
----------------
A wanted signal is *proper* (E[s^2] = 0); imbalance makes E[r^2] != 0.  The
widely-linear corrector

    y[n] = r[n] + w * conj(r[n])

is adapted to drive the pseudo-power E[y^2] -> 0 with the complex LMS update

    w <- w - mu_lms * y[n]^2

which converges to w = -E[r^2] / (2 E[|r|^2]) and cancels the conjugate (image)
term -- restoring the rejection to the noise/length floor.  This is the standard
blind I/Q compensator (Valkama/Anttila); it complements pilot/Havens-style
balancing and needs no training signal.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Signal, imbalance, metric
# ---------------------------------------------------------------------------
def analytic_wanted(N, fa, fb, fs, seed=0):
    """Proper single-sided (analytic) wanted signal: random energy only in the
    positive band [fa, fb]."""
    rng = np.random.default_rng(seed)
    X = np.zeros(N, dtype=complex)
    f = np.fft.fftfreq(N, 1 / fs)
    band = (f >= fa) & (f <= fb)                    # positive freqs only
    X[band] = (rng.standard_normal(band.sum())
               + 1j * rng.standard_normal(band.sum()))
    s = np.fft.ifft(X)
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


def apply_imbalance(s, gain_db, phase_deg):
    """Frequency-flat I/Q imbalance: amplitude ratio g and phase error on Q."""
    g = 10 ** (gain_db / 20)
    phi = np.radians(phase_deg)
    I, Q = s.real, s.imag
    Qp = g * (Q * np.cos(phi) + I * np.sin(phi))
    return I + 1j * Qp


def irr_theory_db(gain_db, phase_deg):
    g = 10 ** (gain_db / 20)
    phi = np.radians(phase_deg)
    mu = (1 + g * np.exp(1j * phi)) / 2
    nu = (1 - g * np.exp(-1j * phi)) / 2
    return 20 * np.log10(np.abs(mu) / np.abs(nu))


def irr_measure_db(z, fa, fb, fs):
    """Image rejection = power in +band vs the mirror -band."""
    Z = np.fft.fft(z)
    f = np.fft.fftfreq(z.size, 1 / fs)
    pos = np.sum(np.abs(Z[(f >= fa) & (f <= fb)]) ** 2)
    neg = np.sum(np.abs(Z[(f <= -fa) & (f >= -fb)]) ** 2)
    return 10 * np.log10(pos / max(neg, 1e-30))


# ---------------------------------------------------------------------------
#  Blind widely-linear LMS corrector
# ---------------------------------------------------------------------------
def blind_lms(r, mu=None, w_hist=False):
    """y[n] = r[n] + w*conj(r[n]);  w <- w - mu*y[n]^2  (drives E[y^2] -> 0)."""
    if mu is None:
        mu = 0.001 / np.mean(np.abs(r) ** 2)        # power-normalised step
    y = np.empty_like(r)
    w = 0j
    hist = np.empty(r.size, dtype=complex) if w_hist else None
    for n in range(r.size):
        yn = r[n] + w * np.conj(r[n])
        y[n] = yn
        w = w - mu * yn * yn
        if w_hist:
            hist[n] = w
    return (y, w, hist) if w_hist else (y, w)


def w_optimal(r):
    return -np.mean(r * r) / (2 * np.mean(np.abs(r) ** 2))


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
def main():
    fs = 48000.0
    fa, fb = 2000.0, 12000.0
    N = 1 << 17
    gain_db, phase_deg = 1.0, 5.0                    # branch imbalance

    s = analytic_wanted(N, fa, fb, fs)
    r = apply_imbalance(s, gain_db, phase_deg)
    y, w, hist = blind_lms(r, w_hist=True)

    irr_before = irr_measure_db(r, fa, fb, fs)
    irr_after = irr_measure_db(y[N // 2:], fa, fb, fs)   # after convergence
    print("I/Q imbalance %.1f dB gain, %.1f deg phase\n" % (gain_db, phase_deg))
    print("  theoretical IRR floor : %6.1f dB" % irr_theory_db(gain_db, phase_deg))
    print("  measured before LMS   : %6.1f dB" % irr_before)
    print("  measured after  LMS   : %6.1f dB" % irr_after)
    print("  w (estimated / optimal): %s / %s"
          % (np.round(w, 4), np.round(w_optimal(r), 4)))

    # learning curve: IRR of y over sliding blocks
    L = 4096
    starts = np.arange(0, N - L, L)
    lc = np.array([irr_measure_db(y[a:a + L], fa, fb, fs) for a in starts])
    lc_t = starts / fs * 1000

    # IRR vs phase imbalance: before (theory) and after LMS
    phases = np.arange(1.0, 11.0, 1.0)
    before_p, after_p = [], []
    for ph in phases:
        rp = apply_imbalance(s, gain_db, ph)
        yp, _ = blind_lms(rp)
        before_p.append(irr_theory_db(gain_db, ph))
        after_p.append(irr_measure_db(yp[N // 2:], fa, fb, fs))

    _plots(fs, fa, fb, r, y, hist, w, r, lc_t, lc, irr_before, irr_after,
           phases, np.array(before_p), np.array(after_p), gain_db, phase_deg)


def _plots(fs, fa, fb, r, y, hist, w, r_full, lc_t, lc, irr_before, irr_after,
           phases, before_p, after_p, gain_db, phase_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) two-sided spectra: imbalanced vs corrected
    def spec(z):
        Z = np.fft.fftshift(np.fft.fft(z * np.hanning(z.size)))
        m = 20 * np.log10(np.maximum(np.abs(Z), 1e-9)); return m - m.max()
    f = np.fft.fftshift(np.fft.fftfreq(r.size, 1 / fs)) / 1000
    ax[0, 0].plot(f, spec(r), color="C3", lw=0.7, label="imbalanced r")
    ax[0, 0].plot(f, spec(y), color="C0", lw=0.7, label="corrected y")
    ax[0, 0].axvspan(fa / 1000, fb / 1000, color="C2", alpha=0.10, label="wanted (+f)")
    ax[0, 0].axvspan(-fb / 1000, -fa / 1000, color="C3", alpha=0.10, label="image (−f)")
    ax[0, 0].set_xlim(-fs / 2000, fs / 2000); ax[0, 0].set_ylim(-100, 5)
    ax[0, 0].set_title("Spectrum — LMS removes the conjugate image")
    ax[0, 0].set_xlabel("Frequency [kHz]"); ax[0, 0].set_ylabel("Level [dB]")
    ax[0, 0].grid(True, alpha=0.4); ax[0, 0].legend(fontsize=8, loc="lower center", ncol=2)

    # (b) learning curve
    ax[0, 1].plot(lc_t, lc, color="C0")
    ax[0, 1].axhline(irr_before, color="C3", ls="--", lw=1, label="before (%.0f dB)" % irr_before)
    ax[0, 1].set_title("LMS learning curve — image rejection vs time")
    ax[0, 1].set_xlabel("Time [ms]"); ax[0, 1].set_ylabel("IRR [dB]")
    ax[0, 1].grid(True, alpha=0.4); ax[0, 1].legend(fontsize=8, loc="lower right")

    # (c) IRR vs phase imbalance: before (theory) vs after LMS
    ax[1, 0].plot(phases, before_p, "s--", color="C3", label="before (imbalance floor)")
    ax[1, 0].plot(phases, after_p, "o-", color="C0", label="after blind LMS")
    ax[1, 0].set_title("Image rejection vs phase imbalance (gain err %.1f dB)" % gain_db)
    ax[1, 0].set_xlabel("Phase imbalance [deg]"); ax[1, 0].set_ylabel("IRR [dB]")
    ax[1, 0].grid(True, alpha=0.4); ax[1, 0].legend(fontsize=8)

    # (d) weight convergence
    n_ms = np.arange(hist.size) / fs * 1000
    wopt = w_optimal(r_full)
    ax[1, 1].plot(n_ms, hist.real, color="C0", lw=0.8, label="Re(w)")
    ax[1, 1].plot(n_ms, hist.imag, color="C1", lw=0.8, label="Im(w)")
    ax[1, 1].axhline(wopt.real, color="C0", ls=":", lw=1)
    ax[1, 1].axhline(wopt.imag, color="C1", ls=":", lw=1)
    ax[1, 1].set_xlim(0, min(200, n_ms[-1]))
    ax[1, 1].set_title("Corrector weight w converges (dotted = optimal)")
    ax[1, 1].set_xlabel("Time [ms]"); ax[1, 1].set_ylabel("w")
    ax[1, 1].grid(True, alpha=0.4); ax[1, 1].legend(fontsize=8)

    fig.suptitle("I/Q branch imbalance + blind widely-linear LMS correction  "
                 "(%.1f dB / %.1f° → restored)" % (gain_db, phase_deg), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "iq_imbalance_lms.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
