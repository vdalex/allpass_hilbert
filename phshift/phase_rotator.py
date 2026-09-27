"""phase_rotator.py

Audio **all-pass phase rotator** -- the "asymmetry eliminator" of speech
processors (Kahn *Symmetra-Peak*; e.g. the CZH-labs / Audiowind MD-A110 board:
an 8-pole "lead" all-pass with its pole near 340 Hz).

Speech is strongly *asymmetric*: the positive and negative excursions differ, so
the waveform has a high crest factor and one polarity clips first.  A cascade of
first-order all-pass sections leaves the **magnitude spectrum untouched** (it is
all-pass) but scrambles the phase, which **redistributes the peaks about the zero
axis** -- the waveform becomes symmetric and its crest factor drops.  Feeding
that into the transmitter clipper/limiter then buys more average (talk) power for
the same peak deviation and less audible distortion -- the reason phase rotators
sit in "almost every broadcast audio processor" and in SSB/AM/NBFM speech chains.

This is a *single* all-pass cascade (one leg), unlike the two-leg 90-degree
splitter used for SSB generation in `ssb_phasing.py` -- but it is built from the
same first-order all-pass section  H(z) = (z^-1 - c)/(1 - c z^-1).

Run:  python phase_rotator.py
"""

import numpy as np
from scipy.signal import lfilter

import digital_phshift as dp


def phase_rotator(x, f0, fs, n_sections=8):
    """Cascade of n_sections identical first-order 'lead' all-pass stages at f0."""
    c = float(dp.f0_to_coeff(f0, fs))
    y = np.asarray(x, dtype=float)
    for _ in range(n_sections):
        y = lfilter([-c, 1.0], [1.0, -c], y)     # (z^-1 - c)/(1 - c z^-1)
    return y, c


def _asymmetric_speech(fs, pitch=150.0, fmax=3400.0, periods=400):
    """Synthetic voiced-speech-like signal: harmonics in cosine phase -> a tall,
    one-sided (asymmetric) glottal-pulse waveform."""
    K = int(fmax // pitch)
    n = np.arange(int(periods * fs / pitch))
    t = n / fs
    x = np.zeros(t.size)
    for k in range(1, K + 1):
        x += np.cos(2 * np.pi * k * pitch * t) / k ** 0.5   # mild 1/sqrt(k) roll-off
    return x, pitch


def _stats(x):
    rms = np.sqrt(np.mean(x ** 2))
    pk_pos, pk_neg = x.max(), -x.min()
    crest = max(pk_pos, pk_neg) / rms
    asym = pk_pos / pk_neg                       # 1.0 == symmetric
    return dict(rms=rms, pk_pos=pk_pos, pk_neg=pk_neg, crest=crest, asym=asym)


def main():
    fs = 8000.0
    f0 = 340.0
    n_sections = 8

    x, pitch = _asymmetric_speech(fs)
    y, c = phase_rotator(x, f0, fs, n_sections)

    # discard IIR start-up transient before measuring
    skip = int(0.5 * fs)
    xs, ys = x[skip:], y[skip:]
    sx, sy = _stats(xs), _stats(ys)

    print("Audio phase rotator: %d-section 'lead' all-pass, pole f0=%g Hz (c=%.5f)\n"
          % (n_sections, f0, c))
    print("  %-22s %10s %10s" % ("", "input", "rotated"))
    print("  %-22s %10.3f %10.3f" % ("+peak", sx["pk_pos"], sy["pk_pos"]))
    print("  %-22s %10.3f %10.3f" % ("-peak", sx["pk_neg"], sy["pk_neg"]))
    print("  %-22s %10.3f %10.3f  (1.0 = symmetric)"
          % ("asymmetry +pk/-pk", sx["asym"], sy["asym"]))
    print("  %-22s %10.3f %10.3f  (unchanged: all-pass)"
          % ("RMS", sx["rms"], sy["rms"]))
    print("  %-22s %9.2fdB %9.2fdB"
          % ("crest factor", 20 * np.log10(sx["crest"]), 20 * np.log10(sy["crest"])))
    print("  peak reduction = %.2f dB   -> ~%.2f dB more average power after clipping"
          % (20 * np.log10(max(sx["pk_pos"], sx["pk_neg"])
                           / max(sy["pk_pos"], sy["pk_neg"])),
             20 * np.log10(max(sx["pk_pos"], sx["pk_neg"])
                           / max(sy["pk_pos"], sy["pk_neg"]))))

    _plots(xs, ys, fs, f0, n_sections, sx, sy, pitch)


def _plots(x, y, fs, f0, n_sections, sx, sy, pitch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # --- (a) waveform, a few pitch periods ---------------------------------
    ns = int(3 * fs / pitch)
    t = np.arange(ns) / fs * 1000
    ax[0, 0].plot(t, x[:ns], color="C3", lw=1.0, label="input (asymmetric)")
    ax[0, 0].plot(t, y[:ns], color="C0", lw=1.0, label="phase-rotated")
    ax[0, 0].axhline(0, color="0.6", lw=0.6)
    ax[0, 0].set_title("Waveform — peaks redistributed about zero")
    ax[0, 0].set_xlabel("Time [ms]"); ax[0, 0].set_ylabel("Amplitude")
    ax[0, 0].grid(True, alpha=0.4); ax[0, 0].legend(fontsize=8, loc="upper right")

    # --- (b) amplitude distribution ---------------------------------------
    bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 80)
    ax[0, 1].hist(x, bins=bins, color="C3", alpha=0.55, density=True,
                  label="input  (+pk/−pk %.2f)" % sx["asym"])
    ax[0, 1].hist(y, bins=bins, color="C0", alpha=0.55, density=True,
                  label="rotated (+pk/−pk %.2f)" % sy["asym"])
    ax[0, 1].axvline(0, color="0.5", lw=0.8)
    ax[0, 1].set_title("Amplitude distribution — asymmetry removed")
    ax[0, 1].set_xlabel("Amplitude"); ax[0, 1].set_ylabel("Density")
    ax[0, 1].grid(True, alpha=0.4); ax[0, 1].legend(fontsize=8)

    # --- (c) magnitude spectrum (identical) -------------------------------
    def mag(sig):
        w = np.hanning(sig.size)
        X = np.fft.rfft(sig * w)
        return 20 * np.log10(np.maximum(np.abs(X), 1e-9))
    faxis = np.fft.rfftfreq(x.size, 1 / fs)
    ax[1, 0].semilogx(faxis, mag(x) - mag(x).max(), color="C3", lw=1.0,
                      label="input")
    ax[1, 0].semilogx(faxis, mag(y) - mag(y).max(), color="C0", lw=0.9, ls="--",
                      label="rotated")
    ax[1, 0].set_xlim(50, fs / 2); ax[1, 0].set_ylim(-80, 5)
    ax[1, 0].set_title("Magnitude spectrum — unchanged (all-pass)")
    ax[1, 0].set_xlabel("Frequency [Hz]"); ax[1, 0].set_ylabel("Level [dB]")
    ax[1, 0].grid(True, which="both", alpha=0.4); ax[1, 0].legend(fontsize=8)

    # --- (d) all-pass phase response --------------------------------------
    fph = np.logspace(np.log10(20), np.log10(fs / 2), 500)
    c = float(dp.f0_to_coeff(f0, fs))
    z1 = np.exp(-1j * 2 * np.pi * fph / fs)
    H = ((z1 - c) / (1 - c * z1)) ** n_sections
    ax[1, 1].semilogx(fph, np.degrees(np.unwrap(np.angle(H))), color="C4")
    ax[1, 1].axvline(f0, color="0.5", ls="--", lw=0.8, label="pole f0=%g Hz" % f0)
    ax[1, 1].set_title("Phase of the %d-section all-pass" % n_sections)
    ax[1, 1].set_xlabel("Frequency [Hz]"); ax[1, 1].set_ylabel("Phase [deg]")
    ax[1, 1].grid(True, which="both", alpha=0.4); ax[1, 1].legend(fontsize=8)

    crest_in = 20 * np.log10(sx["crest"])
    crest_out = 20 * np.log10(sy["crest"])
    fig.suptitle("Audio phase rotator (asymmetry eliminator) — crest factor "
                 "%.1f dB → %.1f dB" % (crest_in, crest_out), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "phase_rotator.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
