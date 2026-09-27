"""ssb_phasing.py

Single-sideband (SSB) generation by the **phasing / Hartley method**, driven by
the all-pass 90-degree audio phase splitter designed in this repo -- the classic
scheme of a phasing HF-transceiver exciter.

Chain
-----
    audio m[n] --> two all-pass legs (I, Q, 90 deg apart across the audio band)
               --> quadrature up-conversion at the carrier fc
               --> USB / LSB

    USB:  s[n] = I[n]*cos(wc n) - Q[n]*sin(wc n)      (wanted at fc+f_audio)
    LSB:  s[n] = I[n]*cos(wc n) + Q[n]*sin(wc n)      (wanted at fc-f_audio)

Because both legs are all-pass (|H| = 1), the amplitude balance is perfect and
the unwanted-sideband suppression is set purely by the phase error eps(f) of the
splitter:

    suppression(f) = -20*log10| tan(eps(f)/2) |   =  the IRR of the design.

The demo feeds a coherent multi-tone across the audio band, up-converts, and
reads the wanted vs. image lines straight off one FFT, then overlays the
theoretical IRR curve.  In a real rig the quadrature mix happens at RF; here fc
is a low IF purely so both sidebands are visible in one spectrum.

Run:  python ssb_phasing.py
"""

import numpy as np
from scipy.signal import lfilter

import digital_phshift as dp
import elliptic_phshift as ep


# ---------------------------------------------------------------------------
#  All-pass leg and phasing modulator
# ---------------------------------------------------------------------------
def allpass_leg(x, c_col):
    """Run x through a cascade of first-order all-pass sections H=(z^-1-c)/(1-c z^-1)."""
    y = np.asarray(x, dtype=float)
    for c in np.atleast_1d(c_col):
        y = lfilter([-c, 1.0], [1.0, -c], y)     # b = -c + z^-1, a = 1 - c z^-1
    return y


def ssb_modulate(audio, c, fc, fs, sideband="USB"):
    """Phasing-method SSB. c is the (N,2) coefficient matrix from the design;
    column 0 = leg I, column 1 = leg Q (leads by 90 deg)."""
    n = np.arange(audio.size)
    wc = 2 * np.pi * fc / fs
    I = allpass_leg(audio, c[:, 0])
    Q = allpass_leg(audio, c[:, 1])
    car_c, car_s = np.cos(wc * n), np.sin(wc * n)
    if sideband.upper() == "USB":
        return I * car_c - Q * car_s
    return I * car_c + Q * car_s                 # LSB


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
def _multitone(Nfft, tone_bins):
    """Coherent multi-tone: 1 period pre-roll + 1 analysed period."""
    n = np.arange(2 * Nfft)
    audio = np.zeros(n.size)
    for k in tone_bins:
        audio += np.sin(2 * np.pi * k * n / Nfft)
    return audio / len(tone_bins)


def evaluate_N(N, fa, fb, fs, fc, audio, Nfft, tone_bins, fc_bin, df):
    """Design N sections/leg, run the SSB chain, return measured + theoretical
    suppression and the output spectrum."""
    d = ep.design_equiripple(N, fa, fb, fs)
    block = ssb_modulate(audio, d["c"], fc, fs, "USB")[Nfft:2 * Nfft]
    X = np.fft.rfft(block) / Nfft * 2.0
    mag_db = 20 * np.log10(np.maximum(np.abs(X), 1e-12))

    f_meas = tone_bins * df
    supp_meas = 20 * np.log10(np.abs(X[fc_bin + tone_bins])
                              / np.maximum(np.abs(X[fc_bin - tone_bins]), 1e-15))

    ftheo = np.logspace(np.log10(fa), np.log10(fb), 500)
    ph = dp.phfunc(d["params"], ftheo, fs)
    irr_theo = ep.phase_error_to_irr_db(ph[1] - ph[0] - 90.0)
    return {"N": N, "mag_db": mag_db, "f_meas": f_meas, "supp_meas": supp_meas,
            "ftheo": ftheo, "irr_theo": irr_theo, "worst": supp_meas.min()}


def main():
    fs = 48000.0            # audio / IF sample rate
    fa, fb = 300.0, 3000.0  # HF communications audio band
    fc = 9000.0             # carrier (low IF, for visualisation)
    Ns = [2, 3, 4, 6, 8]    # sections per leg to compare (2N all-pass stages)

    Nfft = 1 << 16
    df = fs / Nfft
    tone_bins = np.unique(np.round(np.arange(400, 3001, 160.0) / df).astype(int))
    fc_bin = int(round(fc / df))
    fc = fc_bin * df
    audio = _multitone(Nfft, tone_bins)

    print("Phasing-method SSB, unwanted-sideband suppression vs. section count:")
    results = []
    for N in Ns:
        r = evaluate_N(N, fa, fb, fs, fc, audio, Nfft, tone_bins, fc_bin, df)
        results.append(r)
        print("  N=%d/leg (2N=%2d stages):  worst-case suppression = %6.1f dB"
              % (N, 2 * N, r["worst"]))

    _plots(results, fc, fa, fb, df)


def _plots(results, fc, fa, fb, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faxis = np.arange(results[0]["mag_db"].size) * df
    cmap = plt.get_cmap("viridis")
    cols = [cmap(x) for x in np.linspace(0.0, 0.85, len(results))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

    # --- image-band spectra: image floor drops with section count ---------
    for r, col in zip(results, cols):
        ax1.plot(faxis / 1000, r["mag_db"], lw=0.9, color=col,
                 label="N=%d" % r["N"])
    ax1.axvspan((fc - fb) / 1000, (fc - fa) / 1000, color="C3", alpha=0.08)
    ax1.axhline(-25, color="0.6", ls=":", lw=0.8)
    ax1.text((fc - fb) / 1000 + 0.05, -20, "wanted level (≈ −25 dB)", fontsize=8,
             color="0.4")
    ax1.set_xlim((fc - fb) / 1000 - 0.2, fc / 1000 + 0.1)
    ax1.set_ylim(-170, 5)
    ax1.set_title("Suppressed image (LSB) region — more sections push the image deeper")
    ax1.set_xlabel("Frequency [kHz]"); ax1.set_ylabel("Level [dB]")
    ax1.grid(True, alpha=0.4); ax1.legend(loc="lower right", ncol=len(results),
                                          fontsize=8, title="sections/leg")

    # --- suppression vs audio frequency, one curve per N ------------------
    for r, col in zip(results, cols):
        ax2.semilogx(r["ftheo"], r["irr_theo"], color=col,
                     label="N=%d  (worst %.0f dB)" % (r["N"], r["worst"]))
        ax2.semilogx(r["f_meas"], r["supp_meas"], "o", color=col, ms=3)
    ax2.axvspan(fa, fb, color="C2", alpha=0.06)
    ax2.set_ylim(30, 150)
    ax2.set_title("Unwanted-sideband suppression vs. section count "
                  "(lines = theory, dots = measured)")
    ax2.set_xlabel("Audio frequency [Hz]"); ax2.set_ylabel("Suppression [dB]")
    ax2.grid(True, which="both", alpha=0.4); ax2.legend(loc="upper center",
                                                        ncol=len(results), fontsize=8)

    fig.tight_layout()
    out = "ssb_phasing.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
