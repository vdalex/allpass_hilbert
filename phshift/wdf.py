"""wdf.py

Wave-Digital-Filter (lattice / one-multiplier) realisation of the all-pass
Hilbert phase-shifter, and a fixed-point sensitivity comparison against the
Direct-Form biquad realisation (see digital_phshift.py).

Why bother
----------
Both structures realise the *same* ideal all-pass response.  The difference
shows up under **coefficient quantisation** (fixed point, e.g. Q15 on an MCU):

* The first-order **WDF / one-multiplier lattice** stores the pole ``c`` itself
  as its single coefficient.  For any quantised ``c`` with |c| < 1 the section
  stays *exactly* all-pass (|H| = 1) and the pole merely shifts by one
  quantisation step -- a structurally lossless, minimum-sensitivity structure.

* The **Direct-Form biquad** stores ``a1 = -(c1+c2)`` and ``a2 = c1*c2``.  Near
  the unit circle (low-frequency sections, |a1| ~ 2, a2 ~ 1) the mapping from
  (a1, a2) to the pole pair is ill-conditioned: a tiny error in a2 can even make
  the discriminant negative, collapsing two real poles into a complex pair and
  wrecking the phase.  Plus a1 needs an extra integer bit (|a1| < 2), costing one
  fractional bit at equal word length.

First-order lattice all-pass (stored coefficient = pole c)
----------------------------------------------------------
    e[n] = x[n] + c * e[n-1]
    y[n] = -c * e[n] + e[n-1]        -->  H(z) = (z^-1 - c)/(1 - c z^-1)

Direct-Form II biquad all-pass
------------------------------
    w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
    y[n] = a2*w[n] + a1*w[n-1] + w[n-2]
                                     -->  H(z) = (a2 + a1 z^-1 + z^-2)
                                                 / (1 + a1 z^-1 + a2 z^-2)
"""

import numpy as np
import digital_phshift as dp


# ---------------------------------------------------------------------------
#  Fixed-point coefficient quantiser
# ---------------------------------------------------------------------------
def quantize(x, frac_bits):
    """Round coefficient(s) to a Q-format with ``frac_bits`` fractional bits.

    ``frac_bits=None`` returns the value unchanged (ideal / float).
    """
    x = np.asarray(x, dtype=float)
    if frac_bits is None:
        return x
    step = 2.0 ** (-frac_bits)
    return np.round(x / step) * step


# ---------------------------------------------------------------------------
#  Time-domain filters (real, usable implementations)
# ---------------------------------------------------------------------------
def wdf_allpass_filter(x, c):
    """First-order WDF / one-multiplier lattice all-pass, coefficient = pole c."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    e1 = 0.0
    for n in range(x.size):
        e = x[n] + c * e1
        y[n] = -c * e + e1
        e1 = e
    return y


def biquad_allpass_filter(x, a1, a2):
    """Second-order Direct-Form II all-pass biquad."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    w1 = w2 = 0.0
    for n in range(x.size):
        w0 = x[n] - a1 * w1 - a2 * w2
        y[n] = a2 * w0 + a1 * w1 + w2
        w2, w1 = w1, w0
    return y


def filter_leg(x, sections, structure):
    """Run a signal through one leg's cascade of sections.

    structure : "wdf"    -> every section as a first-order lattice (poles only)
                "biquad" -> 2nd-order sections as DF-II biquads, 1st-order direct
    """
    y = np.asarray(x, dtype=float)
    for s in sections:
        if structure == "wdf":
            for c in _section_poles(s):
                y = wdf_allpass_filter(y, c)
        elif structure == "biquad":
            if s["order"] == 2:
                y = biquad_allpass_filter(y, s["a1"], s["a2"])
            else:
                y = wdf_allpass_filter(y, s["c"])   # 1st-order == lattice == direct
        else:
            raise ValueError("structure must be 'wdf' or 'biquad'")
    return y


def _section_poles(s):
    return list(s["poles"]) if s["order"] == 2 else [s["c"]]


# ---------------------------------------------------------------------------
#  Frequency response with (optionally quantised) coefficients
# ---------------------------------------------------------------------------
def _ap1_tf(c, z1):
    return (z1 - c) / (1.0 - c * z1)


def _biquad_tf(a1, a2, z1):
    z2 = z1 * z1
    return (a2 + a1 * z1 + z2) / (1.0 + a1 * z1 + a2 * z2)


def leg_response(sections, freq, fs, structure, frac_bits=None):
    """Complex response of one leg, coefficients quantised as the structure
    would actually store them.

    "wdf"    quantises the pole ``c`` of every first-order section.
    "biquad" quantises ``a1, a2`` of every 2nd-order section (a1 keeps its
             integer bit; here the same fractional step is used for both).
    """
    w = 2.0 * np.pi * np.asarray(freq, dtype=float) / fs
    z1 = np.exp(-1j * w)
    H = np.ones(np.size(freq), dtype=complex)
    for s in sections:
        if structure == "wdf":
            for c in _section_poles(s):
                H = H * _ap1_tf(quantize(c, frac_bits), z1)
        elif structure == "biquad":
            if s["order"] == 2:
                a1 = quantize(s["a1"], frac_bits)
                a2 = quantize(s["a2"], frac_bits)
                H = H * _biquad_tf(a1, a2, z1)
            else:
                H = H * _ap1_tf(quantize(s["c"], frac_bits), z1)
        else:
            raise ValueError("structure must be 'wdf' or 'biquad'")
    return H


# ---------------------------------------------------------------------------
#  Image-rejection metric
# ---------------------------------------------------------------------------
def image_rejection_db(phase_diff_deg, phdesired=90.0):
    """Image-rejection ratio (dB) of an I/Q splitter with perfect amplitude
    match and phase error ``phase_diff - phdesired`` (deg).

        IRR = -20*log10|tan(err/2)|            (err in radians)
    """
    err = np.radians(np.asarray(phase_diff_deg) - phdesired)
    t = np.abs(np.tan(err / 2.0))
    t = np.maximum(t, 1e-18)          # avoid log(0) where error is exactly 0
    return -20.0 * np.log10(t)


def evaluate(d, structure, frac_bits=None, nf=4000):
    """Worst-case & RMS phase error and worst-case IRR of a design ``d`` when
    realised with ``structure`` at ``frac_bits`` fractional bits.
    """
    fs = d["fs"]
    freq = np.logspace(np.log10(d["fa"]), np.log10(d["fb"]), nf)
    leg1, leg2 = dp.biquads(d)
    H1 = leg_response(leg1, freq, fs, structure, frac_bits)
    H2 = leg_response(leg2, freq, fs, structure, frac_bits)
    pdiff = -np.degrees(np.angle(H2 * np.conj(H1)))
    err = ((pdiff - d["phdesired"] + 180) % 360) - 180
    irr = image_rejection_db(pdiff, d["phdesired"])
    return {
        "freq": freq, "phase_diff": pdiff, "error": err, "irr": irr,
        "max_err": np.max(np.abs(err)), "rms_err": np.sqrt(np.mean(err ** 2)),
        "min_irr": np.min(irr),
    }


# ---------------------------------------------------------------------------
#  Comparison report + plot
# ---------------------------------------------------------------------------
def compare(d, bits_list=(15, 12, 10, 8, 6), verbose=True):
    """Print a table of worst-case phase error / IRR vs word length for both
    structures, and return the collected results.
    """
    rows = []
    ideal_w = evaluate(d, "wdf", None)
    ideal_b = evaluate(d, "biquad", None)
    if verbose:
        print("\nFixed-point sensitivity: WDF lattice vs Direct-Form biquad")
        print("  Design: %g deg, N=%d, %g-%g Hz, Fs=%g Hz\n"
              % (d["phdesired"], d["N"], d["fa"], d["fb"], d["fs"]))
        print("  %-8s | %-26s | %-26s" % ("", "WDF lattice", "Direct-Form biquad"))
        print("  %-8s | %12s %12s | %12s %12s"
              % ("frac", "max err[deg]", "min IRR[dB]", "max err[deg]", "min IRR[dB]"))
        print("  " + "-" * 74)
        print("  %-8s | %12.4f %12.1f | %12.4f %12.1f"
              % ("ideal", ideal_w["max_err"], ideal_w["min_irr"],
                 ideal_b["max_err"], ideal_b["min_irr"]))
    for B in bits_list:
        w = evaluate(d, "wdf", B)
        b = evaluate(d, "biquad", B)
        rows.append((B, w, b))
        if verbose:
            print("  Q%-7d | %12.4f %12.1f | %12.4f %12.1f"
                  % (B, w["max_err"], w["min_irr"], b["max_err"], b["min_irr"]))
    if verbose:
        print("\n  (IRR = worst-case image rejection across the band; higher is better)")
    return {"ideal_wdf": ideal_w, "ideal_biquad": ideal_b, "rows": rows}


def plot_compare(d, bits=8, save_prefix=None, show=True):
    """Plot IRR vs frequency: ideal, WDF@bits, biquad@bits."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ideal = evaluate(d, "wdf", None)
    w = evaluate(d, "wdf", bits)
    b = evaluate(d, "biquad", bits)
    f = ideal["freq"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.semilogx(f, ideal["irr"], "k-", lw=1.0, label="ideal (float)")
    ax1.semilogx(f, w["irr"], "C2-", label="WDF lattice, Q%d" % bits)
    ax1.semilogx(f, b["irr"], "C3-", label="Direct-Form biquad, Q%d" % bits)
    ax1.set_title("Image rejection vs frequency  (higher = better)")
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("IRR [dB]")
    ax1.grid(True, which="both"); ax1.legend()

    ax2.semilogx(f, ideal["error"], "k-", lw=1.0, label="ideal (float)")
    ax2.semilogx(f, w["error"], "C2-", label="WDF lattice, Q%d" % bits)
    ax2.semilogx(f, b["error"], "C3-", label="Direct-Form biquad, Q%d" % bits)
    ax2.axhline(0, color="0.6", lw=0.6)
    ax2.set_title("Phase error (phase difference - %g deg)" % d["phdesired"])
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Error [deg]")
    ax2.grid(True, which="both"); ax2.legend()

    fig.tight_layout()
    if save_prefix:
        out = save_prefix + ".png"
        fig.savefig(out, dpi=110)
        print("  Saved plot -> %s" % out)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = dp.design(90, 6, 270, 3600, 22050, pnorm=2, method="digital", verbose=False)

    # Sanity: time-domain filters reproduce the analytic transfer function.
    imp = np.zeros(64); imp[0] = 1.0
    leg1, _ = dp.biquads(d)
    y_wdf = filter_leg(imp, leg1, "wdf")
    y_bq = filter_leg(imp, leg1, "biquad")
    print("time-domain WDF vs biquad leg impulse match: max diff = %.2e"
          % np.max(np.abs(y_wdf - y_bq)))

    compare(d)
    plot_compare(d, bits=8, save_prefix="wdf_vs_biquad", show=False)
