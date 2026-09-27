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
def main():
    fs = 48000.0            # audio / IF sample rate
    fa, fb = 300.0, 3000.0  # HF communications audio band
    fc = 9000.0             # carrier (low IF, for visualisation)
    target_irr = 50.0       # design spec: unwanted-sideband suppression

    # --- design the 90-degree splitter from the IRR spec -------------------
    N, d, irr = ep.min_sections(fa, fb, fs, target_irr, verbose=True)
    c = d["c"]
    print("\nUsing N=%d sections/leg  (2N=%d all-pass stages), worst-case IRR %.1f dB"
          % (N, 2 * N, irr))

    # --- coherent multi-tone test signal (tones on exact FFT bins) ---------
    Nfft = 1 << 16
    df = fs / Nfft
    tone_hz = np.arange(400, 3001, 160.0)
    tone_bins = np.round(tone_hz / df).astype(int)
    tone_bins = np.unique(tone_bins)
    fc_bin = int(round(fc / df))
    fc = fc_bin * df

    n = np.arange(2 * Nfft)                       # 1 period pre-roll + 1 analysed
    audio = np.zeros(n.size)
    for k in tone_bins:
        audio += np.sin(2 * np.pi * k * n / Nfft)
    audio /= len(tone_bins)

    usb = ssb_modulate(audio, c, fc, fs, "USB")
    block = usb[Nfft:2 * Nfft]                    # steady-state, coherent window
    X = np.fft.rfft(block) / Nfft * 2.0
    faxis = np.arange(X.size) * df
    mag_db = 20 * np.log10(np.maximum(np.abs(X), 1e-12))

    # measured suppression per tone: wanted (fc+f) vs image (fc-f)
    supp_meas, f_meas = [], []
    for k in tone_bins:
        f = k * df
        w = np.abs(X[fc_bin + k])
        img = np.abs(X[fc_bin - k])
        supp_meas.append(20 * np.log10(w / max(img, 1e-12)))
        f_meas.append(f)
    supp_meas = np.array(supp_meas)
    f_meas = np.array(f_meas)

    # theoretical IRR from the splitter's phase error
    ftheo = np.logspace(np.log10(fa), np.log10(fb), 400)
    ph = dp.phfunc(d["params"], ftheo, fs)
    eps = ph[1] - ph[0] - 90.0
    irr_theo = ep.phase_error_to_irr_db(eps)

    worst = supp_meas.min()
    print("Measured worst-case sideband suppression in band: %.1f dB" % worst)

    _plots(faxis, mag_db, fc, fa, fb, f_meas, supp_meas, ftheo, irr_theo,
           worst, N, target_irr)


def _plots(faxis, mag_db, fc, fa, fb, f_meas, supp_meas, ftheo, irr_theo,
           worst, N, target_irr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    # --- SSB output spectrum ----------------------------------------------
    ax1.plot(faxis / 1000, mag_db, lw=0.8, color="C0")
    ax1.axvspan((fc + fa) / 1000, (fc + fb) / 1000, color="C2", alpha=0.12,
                label="wanted sideband (USB)")
    ax1.axvspan((fc - fb) / 1000, (fc - fa) / 1000, color="C3", alpha=0.12,
                label="suppressed image (LSB)")
    ax1.axvline(fc / 1000, color="0.5", ls="--", lw=0.8, label="carrier fc")
    ax1.set_xlim(5, 13)
    ax1.set_ylim(-120, 5)
    ax1.set_title("Phasing SSB output spectrum (multi-tone audio, USB selected)")
    ax1.set_xlabel("Frequency [kHz]"); ax1.set_ylabel("Level [dB]")
    ax1.grid(True, alpha=0.4); ax1.legend(loc="upper right", fontsize=8)

    # --- suppression vs audio frequency -----------------------------------
    ax2.semilogx(ftheo, irr_theo, color="C0",
                 label="theoretical IRR = −20·log₁₀|tan(ε/2)|")
    ax2.semilogx(f_meas, supp_meas, "o", color="C3", ms=5,
                 label="measured (from output FFT)")
    ax2.axhline(target_irr, color="0.5", ls="--", lw=0.8,
                label="design target %.0f dB" % target_irr)
    ax2.axvspan(fa, fb, color="C2", alpha=0.08)
    ax2.set_ylim(30, max(90, irr_theo.max() + 5))
    ax2.set_title("Unwanted-sideband suppression  (N=%d sections/leg, worst %.1f dB)"
                  % (N, worst))
    ax2.set_xlabel("Audio frequency [Hz]"); ax2.set_ylabel("Suppression [dB]")
    ax2.grid(True, which="both", alpha=0.4); ax2.legend(loc="lower center", fontsize=8)

    fig.tight_layout()
    out = "ssb_phasing.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
