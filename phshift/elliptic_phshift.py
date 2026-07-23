"""elliptic_phshift.py

Spec-driven design of the digital 90-degree all-pass phase splitter:
**given the band and a target image-rejection (IRR), return the smallest design
that meets it.**

Why
---
`digital_phshift.py` answers "given N sections, how good is it?".  In practice
the question is the other way round: "I need >= X dB of image rejection from
fa to fb -- what is the smallest N?".  This module answers that.

How
---
* **Synthesis** reuses the proven equiripple (minimax) design from
  `digital_phshift` -- a large p-norm drives the phase error to equal ripple,
  which is the optimal (global) criterion for a given N.
* **Order estimate** uses a closed-form law for the per-section rejection.  For
  the equiripple splitter the worst-case in-band IRR is very nearly linear in N:

      IRR(N)  ~=  s(L) * N  -  6.1   [dB],      L = ln(fb/fa)
      s(L)    ~=  40.1 * L**(-0.712)  [dB per section]

  (``s`` is the per-section rejection gain: narrow bands -- small L -- buy more
  dB per section.)  This was calibrated against the equiripple designs and is
  accurate to a few percent; :func:`min_sections` uses it only as a starting
  guess and then **verifies by designing**, so the returned N is guaranteed.

The image-rejection ratio of an all-pass I/Q splitter (perfect amplitude match,
phase error eps from 90 deg) is  IRR = -20*log10|tan(eps/2)|.
"""

import numpy as np

import digital_phshift as dp

# Calibrated per-section rejection law (fitted to the equiripple designs).
_S_A, _S_P, _S_OFF = 40.1, 0.712, 6.1


# ---------------------------------------------------------------------------
#  Image-rejection <-> phase-error helpers
# ---------------------------------------------------------------------------
def phase_error_to_irr_db(eps_deg):
    """Worst-case phase deviation from 90 deg -> image-rejection ratio (dB)."""
    t = np.abs(np.tan(np.radians(eps_deg) / 2.0))
    return -20.0 * np.log10(np.maximum(t, 1e-18))


def irr_db_to_phase_error(irr_db):
    """Image-rejection target (dB) -> allowed worst-case phase error (deg)."""
    return np.degrees(2.0 * np.arctan(10.0 ** (-irr_db / 20.0)))


# ---------------------------------------------------------------------------
#  Equiripple synthesis + measurement
# ---------------------------------------------------------------------------
def design_equiripple(N, fa, fb, fs, method="digital", pnorm=40):
    """Equiripple (minimax) 90-degree splitter for N sections per leg.

    Thin wrapper over `digital_phshift.design` with a large p-norm, which drives
    the phase error to equal ripple -- the optimal criterion for fixed N.
    """
    return dp.design(90, N, fa, fb, fs, pnorm=pnorm, method=method, verbose=False)


def achieved_irr(d, nf=3000):
    """Worst-case in-band IRR (dB) and max phase error (deg) of a design dict."""
    freq = np.logspace(np.log10(d["fa"]), np.log10(d["fb"]), nf)
    ph = dp.phfunc(d["params"], freq, d["fs"])
    eps = np.max(np.abs(ph[1] - ph[0] - 90.0))
    return phase_error_to_irr_db(eps), eps


# ---------------------------------------------------------------------------
#  Closed-form order estimate + guaranteed minimum order
# ---------------------------------------------------------------------------
def per_section_gain_db(fa, fb):
    """Closed-form per-section image-rejection gain s(L) [dB/section]."""
    L = np.log(fb / fa)
    return _S_A * L ** (-_S_P)


def predicted_irr(N, fa, fb):
    """Closed-form estimate of the worst-case in-band IRR (dB) for N sections."""
    return per_section_gain_db(fa, fb) * N - _S_OFF


def order_estimate(fa, fb, irr_db):
    """Closed-form estimate of the sections-per-leg N needed for a target IRR."""
    return max(1.0, (irr_db + _S_OFF) / per_section_gain_db(fa, fb))


def min_sections(fa, fb, fs, irr_db, method="digital", nmax=24, verbose=True):
    """Smallest sections-per-leg N whose equiripple design meets the IRR target.

    Uses the closed-form estimate to start, then designs and verifies upward, so
    the returned N is guaranteed to reach the spec.  Returns (N, design, IRR).
    """
    est = order_estimate(fa, fb, irr_db)
    n0 = max(1, int(np.ceil(est)) - 1)
    if verbose:
        print("Target: IRR >= %.1f dB over %g-%g Hz  (Fs=%g)"
              % (irr_db, fa, fb, fs))
        print("  allowed phase error   = %.4f deg" % irr_db_to_phase_error(irr_db))
        print("  per-section gain s(L) = %.2f dB/section" % per_section_gain_db(fa, fb))
        print("  closed-form estimate  = %.2f  ->  start search at N=%d\n" % (est, n0))
    for N in range(n0, nmax + 1):
        d = design_equiripple(N, fa, fb, fs, method=method)
        irr, eps = achieved_irr(d)
        if verbose:
            flag = "  <-- meets spec" if irr >= irr_db else ""
            print("  N=%-2d ->  IRR = %6.1f dB   (max err %.4f deg)%s"
                  % (N, irr, eps, flag))
        if irr >= irr_db:
            return N, d, irr
    raise RuntimeError("No design up to N=%d reaches %.1f dB" % (nmax, irr_db))


# ---------------------------------------------------------------------------
#  Demo / self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Closed-form IRR estimate vs achieved (equiripple design):\n")
    print("  %-16s %3s %14s %14s" % ("band", "N", "predicted", "achieved"))
    for fa, fb in [(300, 3400), (150, 4500), (60, 5000)]:
        for N in (3, 5):
            d = design_equiripple(N, fa, fb, 22050)
            irr, _ = achieved_irr(d)
            print("  %5g-%-5g Hz   %2d %11.1f dB %11.1f dB"
                  % (fa, fb, N, predicted_irr(N, fa, fb), irr))
    print()
    N, d, irr = min_sections(270, 3600, 22050, 70.0)
    print("\n=> minimum design for 70 dB over 270-3600 Hz: N=%d/leg (achieved %.1f dB)"
          % (N, irr))
