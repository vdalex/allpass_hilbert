"""digital_phshift.py

Design of a DIGITAL all-pass phase-shift network (e.g. a Hilbert / 90-degree
phase splitter), using the same optimisation methodology as the original MATLAB
``phshift.m`` that designed the *analog* network.

Methodology (identical to the MATLAB original)
----------------------------------------------
The network has two "legs" (outputs I and Q).  Each leg is a cascade of ``N``
first-order all-pass sections.  The free parameters are the "90-degree
frequencies" ``f0`` of every section (one per section, per leg).  The optimiser
(Nelder-Mead simplex, ``scipy.optimize.fmin`` -- the equivalent of MATLAB's
``fminsearch``) adjusts these frequencies so that the phase *difference* between
leg 2 and leg 1 equals the desired value (90 deg for a Hilbert transformer)
across the frequency band ``[fa, fb]``.  The cost is the p-norm of the phase
error (p = 2 -> least squares, p ~ 10 -> equal-ripple / minimax).

Difference from the analog original
-----------------------------------
The analog code evaluates the phase of an analog first-order all-pass section:

    phi_analog(f) = (360/pi) * atan(f / f0)          [deg, 0 -> 180]

Here we instead evaluate the phase of the equivalent *digital* first-order
all-pass section.  Each analog 90-degree frequency ``f0`` is converted to a
digital all-pass coefficient using the bilinear transform (this is exactly the
formula used in ``analog_to_digital_allpass.ods``):

    c = (1 - w0*T/2) / (1 + w0*T/2),   w0 = 2*pi*f0,   T = 1/Fs

which gives the standard real first-order all-pass section

    H(z) = (z^-1 - c) / (1 - c*z^-1)                 (pole at z = c, |c| < 1)

whose difference equation is

    y[n] = -c*x[n] + x[n-1] + c*y[n-1]

Because the optimiser works on the *digital* phase response directly, it
automatically compensates for the bilinear frequency warping, so the 90-degree
accuracy holds all the way up towards Nyquist.

Author: port of Michael Spencer's phshift.m (1992/2000) to a digital design.
"""

import numpy as np
from scipy.optimize import fmin


# ---------------------------------------------------------------------------
#  Core all-pass phase model
# ---------------------------------------------------------------------------
def f0_to_coeff(f0, fs):
    """Analog 90-degree frequency ``f0`` (Hz) -> digital all-pass pole ``c``.

    Bilinear transform of the analog all-pass H(s) = (w0 - s)/(w0 + s).
    Identical to the ``B1`` formula in analog_to_digital_allpass.ods.
    """
    w0 = 2.0 * np.pi * np.asarray(f0, dtype=float)
    T = 1.0 / fs
    x = w0 * T / 2.0
    return (1.0 - x) / (1.0 + x)


def phfunc(params, freq, fs):
    """Total phase of the digital all-pass network, per leg.

    Parameters
    ----------
    params : (N, 2) array
        log10 of the angular 90-degree frequency (log10(2*pi*f0)) for each of
        the N sections in each of the 2 legs.  (Same log-space parametrisation
        as the MATLAB original, which keeps the magnitudes well conditioned.)
    freq : (M,) array
        Frequency vector in Hz (must be < fs/2).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    phase : (2, M) array
        Positive phase (deg, 0 -> 180 per section) of each leg, summed over the
        N cascaded sections.  Sign convention matches the analog phfunc.m
        (each section's inversion is ignored, phase increases with frequency).
    """
    params = np.asarray(params, dtype=float).reshape(-1, 2)  # (N, 2)
    w0 = 10.0 ** params                                      # angular freq (N, 2)
    T = 1.0 / fs
    x = w0 * T / 2.0
    c = (1.0 - x) / (1.0 + x)                                # pole / coeff (N, 2)

    w = 2.0 * np.pi * np.asarray(freq, dtype=float) / fs     # digital rad/sample (M,)
    z1 = np.exp(-1j * w)                                      # z^-1 on unit circle (M,)

    c = c[:, :, None]                                        # (N, 2, 1)
    z1 = z1[None, None, :]                                   # (1, 1, M)
    H = (z1 - c) / (1.0 - c * z1)                            # (N, 2, M)

    # per-section phase, positive & increasing 0 -> 180 deg (matches analog sign)
    sect_phase = -np.degrees(np.angle(H))                    # (N, 2, M)
    phase = sect_phase.sum(axis=0)                           # (2, M)
    return phase


def phcost(params_flat, freq, fs, phdesired, pnorm):
    """p-norm cost of the phase-difference error (matches phcost.m)."""
    M = np.size(freq)
    phase = phfunc(params_flat, freq, fs)
    phaseerr = phase[1, :] - phase[0, :] - phdesired
    return (np.sum(np.abs(phaseerr) ** pnorm) ** (1.0 / pnorm)) / (2 * M)


def phfunc_analog(params, freq):
    """Total phase of the *analog* all-pass network, per leg (exactly phfunc.m).

    Per section:  phi(f) = (360/pi) * atan(f / f0)   [deg, 0 -> 180]
    where f0 = 10**params / (2*pi).  No sampling frequency involved -- this is
    the pure analog model used by the two-step design mode.
    """
    params = np.asarray(params, dtype=float).reshape(-1, 2)  # (N, 2)
    w0 = 10.0 ** params                                      # angular freq (N, 2)
    freq = np.asarray(freq, dtype=float)                     # (M,)
    ratio = (2.0 * np.pi / w0)[:, :, None] * freq[None, None, :]   # f/f0 (N,2,M)
    sect_phase = (360.0 / np.pi) * np.arctan(ratio)          # (N, 2, M)
    return sect_phase.sum(axis=0)                            # (2, M)


def phcost_analog(params_flat, freq, phdesired, pnorm):
    """p-norm cost using the analog phase model (for the two-step mode)."""
    M = np.size(freq)
    phase = phfunc_analog(params_flat, freq)
    phaseerr = phase[1, :] - phase[0, :] - phdesired
    return (np.sum(np.abs(phaseerr) ** pnorm) ** (1.0 / pnorm)) / (2 * M)


# ---------------------------------------------------------------------------
#  Second-order (biquad) grouping
# ---------------------------------------------------------------------------
def to_biquads(poles):
    """Group a leg's first-order all-pass poles into 2nd-order all-pass biquads.

    Two cascaded first-order all-pass sections with real poles c1, c2

        (z^-1 - c1)/(1 - c1 z^-1) * (z^-1 - c2)/(1 - c2 z^-1)

    combine *exactly* into one 2nd-order all-pass biquad

        H(z) = (a2 + a1 z^-1 + z^-2) / (1 + a1 z^-1 + a2 z^-2)
        a1 = -(c1 + c2),   a2 = c1 * c2

    (numerator = mirror of denominator -> unit magnitude, phase doubled).
    This is the "dual stage" of the AES paper / the 2-Order table in the .ods,
    and halves the number of filter blocks.  Grouping does not change the
    overall phase response.

    Adjacent (frequency-sorted) sections are paired.  With an odd number of
    sections the highest one is left as a first-order section.

    Parameters
    ----------
    poles : (N,) array of real first-order all-pass coefficients c (|c| < 1).

    Returns
    -------
    list of dicts, each either
        {"order": 2, "a1": a1, "a2": a2, "poles": (c1, c2)}   or
        {"order": 1, "c": c}
    """
    c = list(np.asarray(poles, dtype=float).ravel())
    sections, i = [], 0
    while i + 1 < len(c):
        c1, c2 = c[i], c[i + 1]
        sections.append({"order": 2, "a1": -(c1 + c2), "a2": c1 * c2,
                         "poles": (c1, c2)})
        i += 2
    if i < len(c):
        sections.append({"order": 1, "c": c[i]})
    return sections


def biquads(d):
    """Return the biquad section list for each leg of a design ``d``.

    Returns a tuple ``(leg1_sections, leg2_sections)`` from :func:`to_biquads`.
    """
    c = d["c"]
    return to_biquads(c[:, 0]), to_biquads(c[:, 1])


def section_response(section, freq, fs):
    """Complex frequency response of one first- or second-order all-pass section."""
    w = 2.0 * np.pi * np.asarray(freq, dtype=float) / fs
    z1 = np.exp(-1j * w)
    if section["order"] == 1:
        c = section["c"]
        return (z1 - c) / (1.0 - c * z1)
    a1, a2 = section["a1"], section["a2"]
    z2 = z1 * z1
    return (a2 + a1 * z1 + z2) / (1.0 + a1 * z1 + a2 * z2)


def biquad_phase(leg_sections, freq, fs):
    """Total positive phase (deg, matches phfunc) of a cascade of sections."""
    H = np.ones(np.size(freq), dtype=complex)
    for s in leg_sections:
        H = H * section_response(s, freq, fs)
    return -np.degrees(np.angle(H))


# ---------------------------------------------------------------------------
#  Design driver
# ---------------------------------------------------------------------------
def design(phdesired, N, fa, fb, fs, pnorm=2, method="digital", verbose=True):
    """Run the Nelder-Mead optimisation and return the design.

    Parameters
    ----------
    method : {"digital", "analog"}
        ``"digital"`` (default, one-step) -- the optimiser works directly on the
        digital all-pass phase, so it compensates for bilinear warping and the
        90-degree accuracy holds up towards Nyquist.

        ``"analog"`` (two-step) -- reproduces the original workflow exactly: the
        optimiser designs the *analog* network (phfunc.m model, no Fs), then each
        analog 90-degree frequency is converted to a digital coefficient with the
        bilinear transform.  This is what ``analog_to_digital_allpass.ods`` does.
        Simpler and Fs-independent in the search, but the digital phase drifts
        from 90 degrees near Nyquist because of frequency warping.

    Returns a dict with the optimised parameters and derived quantities.
    """
    if method not in ("digital", "analog"):
        raise ValueError("method must be 'digital' or 'analog', got %r" % method)
    if fb >= fs / 2.0:
        raise ValueError(
            "End frequency fb=%g Hz must be below Nyquist fs/2=%g Hz" % (fb, fs / 2.0)
        )

    # Frequency grid used by the optimiser (log spaced), same density as MATLAB.
    freqopt = np.logspace(np.log10(fa), np.log10(fb), N * 40 + 1)

    # Initial guess -- ad hoc, taken directly from phshift.m.  Leg 1 sits at
    # higher corner frequencies, leg 2 at lower ones, so their phase curves are
    # offset by ~90 deg to begin with.  Corners are clipped just below Nyquist.
    fnyq = fs / 2.0
    leg1 = np.clip(np.logspace(np.log10(0.75 * 2 * fa), np.log10(0.85 * 2 * fb), N),
                   None, 0.95 * fnyq)
    leg2 = np.clip(np.logspace(np.log10(0.75 * fa), np.log10(0.85 * fb), N),
                   None, 0.95 * fnyq)
    params = 2.0 * np.pi * np.column_stack([leg1, leg2])     # angular (N, 2)
    params = np.log10(params)                                # log space

    # Pick the phase model / cost for the requested mode.
    if method == "digital":
        costfun, args = phcost, (freqopt, fs, phdesired, pnorm)
    else:  # analog two-step -- optimise the analog network, convert afterwards
        costfun, args = phcost_analog, (freqopt, phdesired, pnorm)

    # Repeatedly restart the simplex until it terminates cleanly several times
    # in a row (mirrors the while-loop in phshift.m).
    donecount = 4
    count = donecount
    x0 = params.ravel()
    while count:
        x0, fval, iters, funcalls, warnflag = fmin(
            costfun, x0, args=args,
            maxiter=4000, maxfun=5000, xtol=1e-9, ftol=1e-12,
            full_output=True, disp=verbose,
        )
        if warnflag == 0:      # converged normally
            count -= 1
        else:                  # hit an iteration/eval limit -> keep going
            count = donecount

    params = x0.reshape(-1, 2)
    params = np.sort(params, axis=0)          # frequency-order the sections

    w0 = 10.0 ** params                        # analog angular 90-deg freq (N,2)
    f0 = w0 / (2.0 * np.pi)                     # analog 90-deg freq in Hz (N,2)
    c = f0_to_coeff(f0, fs)                     # digital all-pass poles (N,2)
    # actual digital 90-deg frequency (bilinear un-warping), Hz
    fd = (fs / np.pi) * np.arctan(np.pi * f0 / fs)

    return {
        "phdesired": phdesired, "N": N, "fa": fa, "fb": fb, "fs": fs,
        "pnorm": pnorm, "fval": fval, "method": method,
        "params": params, "f0": f0, "c": c, "fd": fd,
        "freqopt": freqopt,
    }


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------
def report(d):
    N, fs = d["N"], d["fs"]
    f0, c, fd = d["f0"], d["c"], d["fd"]

    method = d.get("method", "digital")
    method_desc = {"digital": "digital (one-step, warping compensated)",
                   "analog": "analog + bilinear (two-step, matches .ods)"}[method]

    print("\nDigital all-pass phase shifter design results:")
    print("  Design method            = %s" % method_desc)
    print("  Desired phase difference = %g deg." % d["phdesired"])
    print("  Sections per leg         = %d" % N)
    print("  Band                     = %g .. %g Hz" % (d["fa"], d["fb"]))
    print("  Sampling frequency Fs    = %g Hz  (Nyquist = %g Hz)" % (fs, fs / 2))
    print("  p-norm                   = %g" % d["pnorm"])
    print("  final cost               = %.4e" % d["fval"])

    leg_names = ("Leg 1 (I)", "Leg 2 (Q)")
    print("\nDigital first-order all-pass sections:  H(z) = (z^-1 - c)/(1 - c*z^-1)")
    print("  difference equation:  y[n] = -c*x[n] + x[n-1] + c*y[n-1]\n")
    header = "  %-11s %-4s %14s %14s %14s" % (
        "Leg", "Sec", "coeff c", "analog f0[Hz]", "digital f90[Hz]")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for l in range(2):
        for n in range(N):
            print("  %-11s %-4d %14.9f %14.2f %14.2f"
                  % (leg_names[l], n + 1, c[n, l], f0[n, l], fd[n, l]))
        print()

    # Second-order (biquad) grouping -- the efficient "dual stage" form.
    leg_bq = biquads(d)
    print("Second-order all-pass biquads:  H(z) = (a2 + a1 z^-1 + z^-2)/(1 + a1 z^-1 + a2 z^-2)")
    print("  difference equation:  y[n] = a2*x[n] + a1*x[n-1] + x[n-2]"
          " - a1*y[n-1] - a2*y[n-2]\n")
    bqhead = "  %-11s %-6s %14s %14s   %s" % (
        "Leg", "Biquad", "a1", "a2", "from poles")
    print(bqhead)
    print("  " + "-" * (len(bqhead) - 2))
    for l in range(2):
        for k, s in enumerate(leg_bq[l]):
            if s["order"] == 2:
                print("  %-11s %-6d %14.9f %14.9f   c=(%.5f, %.5f)"
                      % (leg_names[l], k + 1, s["a1"], s["a2"], *s["poles"]))
            else:
                print("  %-11s %-6d %14s %14s   c=%.5f  (1st-order leftover)"
                      % (leg_names[l], k + 1, "-", "-", s["c"]))
        print()

    # Worst-case phase error over the design band (sanity check).
    fcheck = np.logspace(np.log10(d["fa"]), np.log10(d["fb"]), 2000)
    ph = phfunc(d["params"], fcheck, fs)
    err = ph[1] - ph[0] - d["phdesired"]
    print("  Max |phase error| in band = %.4f deg   (RMS = %.4f deg)"
          % (np.max(np.abs(err)), np.sqrt(np.mean(err ** 2))))

    # Confirm the biquad grouping reproduces the first-order phase exactly.
    def _leg_H(sections):
        H = np.ones_like(fcheck, dtype=complex)
        for s in sections:
            H = H * section_response(s, fcheck, fs)
        return H
    dbq = -np.degrees(np.angle(_leg_H(leg_bq[1]) * np.conj(_leg_H(leg_bq[0]))))
    mism = np.abs(((dbq - (ph[1] - ph[0]) + 180) % 360) - 180)  # wrap-safe
    print("  Biquad vs 1st-order phase mismatch = %.2e deg  (should be ~0)"
          % np.max(mism))


def plot(d, save_prefix=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmin_, fmax_ = d["fa"] / 2.0, min(d["fb"] * 2.0, 0.499 * d["fs"])
    freq = np.logspace(np.log10(fmin_), np.log10(fmax_), 1000)
    ph = phfunc(d["params"], freq, d["fs"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.semilogx(freq, ph[1] - ph[0], "b-")
    ax1.axhline(d["phdesired"], color="k", ls="--", lw=0.8)
    ax1.axvspan(d["fa"], d["fb"], color="green", alpha=0.10, label="design band")
    ax1.set_title("Phase difference (Leg 2 - Leg 1)")
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("Phase [deg]")
    ax1.grid(True, which="both"); ax1.legend()

    ax2.semilogx(freq, ph[0], "b-", label="Leg 1 (I)")
    ax2.semilogx(freq, ph[1], "r-", label="Leg 2 (Q)")
    ax2.set_title("Total phase of each leg")
    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Phase [deg]")
    ax2.grid(True, which="both"); ax2.legend()

    fig.tight_layout()
    if save_prefix:
        out = save_prefix + ".png"
        fig.savefig(out, dpi=110)
        print("  Saved plot -> %s" % out)
    if show:
        plt.show()
    return fig


def plot_compare(designs, labels=None, save_prefix=None, show=True):
    """Overlay the phase response of several designs for comparison.

    Parameters
    ----------
    designs : list of design dicts (from :func:`design`), same fa/fb/fs/phdesired.
    labels  : optional list of legend labels (defaults to each design's method).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d0 = designs[0]
    fa, fb, fs, pd = d0["fa"], d0["fb"], d0["fs"], d0["phdesired"]
    if labels is None:
        labels = [d.get("method", "design") for d in designs]

    fwide = np.logspace(np.log10(fa / 2.0),
                        np.log10(min(fb * 2.0, 0.499 * fs)), 1500)
    fband = np.logspace(np.log10(fa), np.log10(fb), 2000)
    colors = ["C0", "C3", "C2", "C1"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    for d, lab, col in zip(designs, labels, colors):
        pw = phfunc(d["params"], fwide, fs)
        pb = phfunc(d["params"], fband, fs)
        emax = np.max(np.abs(pb[1] - pb[0] - pd))
        ax1.semilogx(fwide, pw[1] - pw[0], col, label="%s" % lab)
        ax2.semilogx(fband, pb[1] - pb[0] - pd, col,
                     label="%s (max %.3f deg)" % (lab, emax))

    ax1.axhline(pd, color="k", ls="--", lw=0.8)
    ax1.axvspan(fa, fb, color="green", alpha=0.10, label="design band")
    ax1.set_title("Phase difference (Leg 2 - Leg 1)")
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("Phase [deg]")
    ax1.grid(True, which="both"); ax1.legend()

    ax2.axhline(0.0, color="k", ls="--", lw=0.8)
    ax2.set_title("Phase error in design band (phase difference - %g deg)" % pd)
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
#  Interactive entry point (mirrors phshift.m)
# ---------------------------------------------------------------------------
def _ask(prompt, cast, default):
    s = input("%s [%s]: " % (prompt, default)).strip()
    return cast(s) if s else cast(default)


def main():
    print("Digital all-pass phase-shifter designer")
    print("(same optimisation methodology as the analog phshift.m)\n")
    phdesired = _ask("Enter phase difference between outputs (deg, 90 = Hilbert)",
                     float, 90)
    N = _ask("Enter number of sections in each leg", int, 4)
    fs = _ask("Enter sampling frequency Fs in Hz", float, 22050)
    fa = _ask("Enter start frequency of phase shifter in Hz", float, 270)
    fb = _ask("Enter end frequency of phase shifter in Hz", float, 3600)
    pnorm = _ask("Enter p-norm (2 = least squares, 10 = equiripple)", float, 2)
    method = _ask("Enter design mode (digital = 1-step / analog = 2-step, .ods)",
                  str, "digital").strip().lower()
    if method not in ("digital", "analog"):
        print("  Unknown mode %r, falling back to 'digital'." % method)
        method = "digital"

    d = design(phdesired, N, fa, fb, fs, pnorm=pnorm, method=method, verbose=True)
    report(d)
    try:
        plot(d, save_prefix="digital_phshift_result", show=True)
    except Exception as e:  # headless / no display -> just save
        print("  (plot display skipped: %s)" % e)
        plot(d, save_prefix="digital_phshift_result", show=False)


if __name__ == "__main__":
    main()
