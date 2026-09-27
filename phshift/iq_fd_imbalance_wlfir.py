"""iq_fd_imbalance_wlfir.py

**Frequency-dependent** I/Q imbalance and its **widely-linear FIR (WL-FIR) LMS**
correction -- the realistic case where the I and Q analog branches have
*different frequency responses* (mismatched filters, a small delay skew), so the
phase/gain error varies across the band and the image rejection is sloped.

A single complex weight (see `iq_imbalance_lms.py`) can only set one gain/phase,
so it flattens the *average* imbalance but leaves a residual slope.  Modelling the
correction as an FIR on the conjugate branch,

    y[n] = r[n-D] - sum_k conj(w_k) * conj(r[n-k])      (K-tap, D = centre delay)

lets the corrector match a frequency-dependent  -nu(f)/mu(f)  and restore the
rejection across the whole band.  The taps adapt by NLMS (an adaptive canceller
using conj(r) as the reference); because  w * conj(r)  lands almost entirely in
the image (-f) band, it removes the image without disturbing the wanted (+f).
"""

import numpy as np
from scipy.signal import welch

import iq_imbalance_lms as iq          # analytic_wanted, blind_lms (1-tap), metrics


# ---------------------------------------------------------------------------
#  Frequency-dependent imbalance (branch delay skew + gain/phase)
# ---------------------------------------------------------------------------
def _frac_delay(x, d):
    """Fractional-sample delay of a real signal via FFT phase."""
    N = x.size
    f = np.fft.fftfreq(N)
    return np.real(np.fft.ifft(np.fft.fft(x) * np.exp(-1j * 2 * np.pi * f * d)))


def apply_fd_imbalance(s, gain_db, phase_deg, delay_samp):
    """I/Q imbalance with a Q-branch delay skew -> phase error grows with freq."""
    g = 10 ** (gain_db / 20)
    phi = np.radians(phase_deg)
    I, Q = s.real, s.imag
    Qd = _frac_delay(Q, delay_samp)
    Qp = g * (Qd * np.cos(phi) + I * np.sin(phi))
    return I + 1j * Qp


# ---------------------------------------------------------------------------
#  Widely-linear FIR LMS corrector
# ---------------------------------------------------------------------------
def wl_fir_lms(r, K=9, mu=0.25, w_hist=False):
    """Blind circularity-restoring widely-linear FIR corrector.

        y[n] = r[n] + sum_{k=0}^{K-1} w_k * conj(r[n-k])
        w_k <- w_k - mu' * y[n] * y[n-k]            (nulls pseudo-autocorr of y)

    The correction term  w * conj(r)  lands in the image (-f) band; the update
    drives the output pseudo-autocorrelation E[y[n] y[n-k]] -> 0.  That quantity
    is zero for a *proper* signal at every lag, so a well-corrected (proper)
    wanted signal is a fixed point and is left untouched whatever its colour;
    only the improper (image) part is removed.  NLMS step.
    """
    N = r.size
    w = np.zeros(K, dtype=complex)
    y = np.array(r, dtype=complex)
    rc = np.conj(r)
    hist = [] if w_hist else None
    for n in range(K, N):
        u = rc[n:n - K:-1]                       # conj(r[n]),...,conj(r[n-K+1])
        yn = r[n] + np.dot(w, u)                 # w^T u  (no conjugation on w)
        y[n] = yn
        yv = y[n:n - K:-1]                       # y[n], y[n-1], ..., y[n-K+1]
        w = w - mu * yn * yv / (np.vdot(yv, yv).real + 1e-9)
        if w_hist and (n % 64 == 0):
            hist.append((n, w.copy()))
    return (y, w, hist) if w_hist else (y, w)


# ---------------------------------------------------------------------------
#  Frequency-resolved image rejection
# ---------------------------------------------------------------------------
def irr_spectrum(z, fs, fa, fb, nperseg=2048):
    f, P = welch(z, fs=fs, nperseg=nperseg, return_onesided=False, detrend=False)
    f = np.fft.fftshift(f); P = np.fft.fftshift(P)
    pos = (f >= fa) & (f <= fb)
    fp = f[pos]
    Pp = P[pos]
    Pm = np.interp(-fp[::-1], f, P)[::-1]         # PSD at the mirror -f
    return fp, 10 * np.log10(Pp / np.maximum(Pm, 1e-30))


def _band_irr(z, fs, fa, fb):
    return iq.irr_measure_db(z, fa, fb, fs)


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
def main():
    fs = 48000.0
    fa, fb = 2000.0, 20000.0                      # wide band to expose the slope
    N = 1 << 16
    gain_db, phase_deg, delay = 0.5, 2.0, 0.06    # + branch delay skew (samples)

    s = iq.analytic_wanted(N, fa, fb, fs)
    r = apply_fd_imbalance(s, gain_db, phase_deg, delay)

    Ktaps = 11
    y1, w1 = iq.blind_lms(r)                      # single complex weight
    yK, wK, hist = wl_fir_lms(r, K=Ktaps, mu=0.003, w_hist=True)

    print("Frequency-dependent I/Q imbalance: %.1f dB, %.1f deg, %.2f-sample skew\n"
          % (gain_db, phase_deg, delay))
    print("  band-average IRR   before : %5.1f dB" % _band_irr(r, fs, fa, fb))
    print("  band-average IRR   1-tap  : %5.1f dB" % _band_irr(y1[N // 2:], fs, fa, fb))
    print("  band-average IRR   WL-FIR : %5.1f dB" % _band_irr(yK[N // 2:], fs, fa, fb))

    fbi, irr_b = irr_spectrum(r, fs, fa, fb)
    _, irr_1 = irr_spectrum(y1[N // 2:], fs, fa, fb)
    _, irr_K = irr_spectrum(yK[N // 2:], fs, fa, fb)
    print("  worst-in-band IRR  before/1-tap/WL-FIR : %.1f / %.1f / %.1f dB"
          % (irr_b.min(), irr_1.min(), irr_K.min()))

    # learning curves
    L = 2048
    starts = np.arange(0, N - L, L)
    lc1 = np.array([_band_irr(y1[a:a + L], fs, fa, fb) for a in starts])
    lcK = np.array([_band_irr(yK[a:a + L], fs, fa, fb) for a in starts])
    lct = starts / fs * 1000

    _plots(fs, fa, fb, r, yK, fbi, irr_b, irr_1, irr_K, lct, lc1, lcK, wK,
           gain_db, phase_deg, delay)


def _plots(fs, fa, fb, r, yK, fbi, irr_b, irr_1, irr_K, lct, lc1, lcK, wK,
           gain_db, phase_deg, delay):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) IRR vs frequency: the key comparison
    ax[0, 0].plot(fbi / 1000, irr_b, color="C3", lw=1.2, label="before (sloped)")
    ax[0, 0].plot(fbi / 1000, irr_1, color="C1", lw=1.2, label="1-tap LMS")
    ax[0, 0].plot(fbi / 1000, irr_K, color="C0", lw=1.2, label="WL-FIR LMS (K=%d)" % wK.size)
    ax[0, 0].set_ylim(0, max(90, np.nanmax(irr_K) + 5))
    ax[0, 0].set_title("Image rejection vs frequency — FIR fixes the slope")
    ax[0, 0].set_xlabel("Frequency [kHz]"); ax[0, 0].set_ylabel("IRR [dB]")
    ax[0, 0].grid(True, alpha=0.4); ax[0, 0].legend(fontsize=8)

    # (b) spectra before vs WL-FIR
    def spec(z):
        Z = np.fft.fftshift(np.fft.fft(z * np.hanning(z.size)))
        m = 20 * np.log10(np.maximum(np.abs(Z), 1e-9)); return m - m.max()
    f = np.fft.fftshift(np.fft.fftfreq(r.size, 1 / fs)) / 1000
    ax[0, 1].plot(f, spec(r), color="C3", lw=0.7, label="imbalanced r")
    ax[0, 1].plot(f, spec(yK), color="C0", lw=0.7, label="WL-FIR corrected")
    ax[0, 1].axvspan(fa / 1000, fb / 1000, color="C2", alpha=0.08)
    ax[0, 1].axvspan(-fb / 1000, -fa / 1000, color="C3", alpha=0.08)
    ax[0, 1].set_xlim(-fs / 2000, fs / 2000); ax[0, 1].set_ylim(-100, 5)
    ax[0, 1].set_title("Spectrum — image band cleared")
    ax[0, 1].set_xlabel("Frequency [kHz]"); ax[0, 1].set_ylabel("Level [dB]")
    ax[0, 1].grid(True, alpha=0.4); ax[0, 1].legend(fontsize=8, loc="lower center")

    # (c) learning curves
    ax[1, 0].plot(lct, lc1, color="C1", label="1-tap LMS")
    ax[1, 0].plot(lct, lcK, color="C0", label="WL-FIR LMS")
    ax[1, 0].set_title("Learning curves (band-average IRR)")
    ax[1, 0].set_xlabel("Time [ms]"); ax[1, 0].set_ylabel("IRR [dB]")
    ax[1, 0].grid(True, alpha=0.4); ax[1, 0].legend(fontsize=8, loc="lower right")

    # (d) corrector FIR taps
    K = wK.size
    ax[1, 1].stem(np.arange(K) - K // 2, np.abs(wK))
    ax[1, 1].set_title("WL-FIR corrector taps |w_k| (K=%d)" % K)
    ax[1, 1].set_xlabel("tap (relative to centre)"); ax[1, 1].set_ylabel("|w_k|")
    ax[1, 1].grid(True, alpha=0.4)

    fig.suptitle("Frequency-dependent I/Q imbalance + WL-FIR LMS  "
                 "(%.1f dB / %.1f° / %.2f-sample skew)" % (gain_db, phase_deg, delay),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "iq_fd_imbalance_wlfir.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
