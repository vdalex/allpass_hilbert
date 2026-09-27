"""polyphase_complex.py

Complex (asymmetric) all-pass / polyphase **image-reject filter** -- the digital
counterpart of the Gingell / Behbahani RC polyphase network.

Idea
----
A real filter has |H(-f)| = |H(+f)| and cannot tell +f from -f.  Combining the
two real all-pass legs of the 90-degree splitter into a single *complex* filter

    C(z) = A0(z) + j * A1(z)          (A0, A1 = the two all-pass legs)

breaks that symmetry: across the design band it adds constructively for +f and
cancels for -f, so its magnitude response is **asymmetric about DC** -- a
passband on +f and a deep notch on the mirror -f.  Applied to a real signal it
outputs the analytic (one-sided) signal; the residual -f leakage is the image,
suppressed by exactly the design IRR.  This is the same image rejection as the
phasing SSB mixer, folded into one complex filter (no separate quadrature mix).

More sections -> deeper notch -> higher image rejection, staggered across the
band by the same geometric-symmetric corner placement used for the RC values of
a multistage RC polyphase filter.

Run:  python polyphase_complex.py
"""

import numpy as np
from scipy.signal import lfilter

import digital_phshift as dp


def _leg_response(c_col, w):
    """Complex frequency response of one all-pass leg at angular freqs w."""
    z1 = np.exp(-1j * w)
    H = np.ones_like(w, dtype=complex)
    for c in np.atleast_1d(c_col):
        H = H * (z1 - c) / (1.0 - c * z1)
    return H


def complex_response(d, w, s):
    """C(w) = A0(w) + s*j*A1(w) for the design d (s = +/-1 picks the passed side)."""
    return _leg_response(d["c"][:, 0], w) + s * 1j * _leg_response(d["c"][:, 1], w)


def _orient(d, wband):
    """Pick s so the +f band is the passband (not the -f band)."""
    p = np.mean(np.abs(complex_response(d, wband, +1)))
    m = np.mean(np.abs(complex_response(d, wband, -1)))
    return +1 if p >= m else -1


def leg_filter(x, c_col):
    y = np.asarray(x, dtype=float)
    for c in np.atleast_1d(c_col):
        y = lfilter([-c, 1.0], [1.0, -c], y)
    return y


def main():
    fs = 48000.0
    fa, fb = 300.0, 3400.0        # band passed on +f, rejected on -f
    Ns = [2, 3, 4, 6]

    designs = {N: dp.design(90, N, fa, fb, fs, pnorm=40, method="digital",
                            verbose=False) for N in Ns}
    wband = 2 * np.pi * np.logspace(np.log10(fa), np.log10(fb), 400) / fs
    s = _orient(designs[Ns[0]], wband)

    print("Complex all-pass polyphase image-reject filter (fs=%g, band %g-%g Hz)\n"
          % (fs, fa, fb))
    worst = {}
    for N in Ns:
        Cp = np.abs(complex_response(designs[N], wband, s))
        Cm = np.abs(complex_response(designs[N], -wband, s))
        irr = 20 * np.log10(Cp / np.maximum(Cm, 1e-15))
        worst[N] = irr.min()
        print("  N=%d/leg (2N=%2d stages):  worst-case image rejection = %6.1f dB"
              % (N, 2 * N, worst[N]))

    # --- process a real multi-tone: show +f passes, -f image suppressed ----
    Nfft = 1 << 15
    df = fs / Nfft
    tone_bins = np.unique(np.round(np.arange(400, 3401, 200.0) / df).astype(int))
    n = np.arange(2 * Nfft)
    audio = np.zeros(n.size)
    for k in tone_bins:
        audio += np.sin(2 * np.pi * k * n / Nfft)
    audio /= len(tone_bins)

    dN = designs[4]
    yI = leg_filter(audio, dN["c"][:, 0])
    yQ = leg_filter(audio, dN["c"][:, 1])
    yc = (yI + s * 1j * yQ)[Nfft:2 * Nfft]
    Y = np.fft.fftshift(np.fft.fft(yc)) / Nfft
    fax = np.fft.fftshift(np.fft.fftfreq(Nfft, 1 / fs))
    Ydb = 20 * np.log10(np.maximum(np.abs(Y), 1e-12))
    Ydb -= Ydb.max()

    _plots(fs, fa, fb, designs, Ns, s, worst, fax, Ydb)


def _plots(fs, fa, fb, designs, Ns, s, worst, fax, Ydb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    cols = {N: cmap(x) for N, x in zip(Ns, np.linspace(0.0, 0.82, len(Ns)))}

    wfull = np.linspace(-np.pi, np.pi, 4000)
    ffull = wfull * fs / (2 * np.pi)
    fband = np.logspace(np.log10(fa), np.log10(fb), 400)
    wband = 2 * np.pi * fband / fs

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # --- (a) asymmetric magnitude response --------------------------------
    for N in (Ns[0], Ns[-1]):
        C = complex_response(designs[N], wfull, s)
        mag = 20 * np.log10(np.maximum(np.abs(C), 1e-9))
        mag -= mag.max()
        ax[0, 0].plot(ffull / 1000, mag, color=cols[N], lw=1.0, label="N=%d" % N)
    ax[0, 0].axvspan(fa / 1000, fb / 1000, color="C2", alpha=0.12, label="pass (+f)")
    ax[0, 0].axvspan(-fb / 1000, -fa / 1000, color="C3", alpha=0.12, label="reject (−f)")
    ax[0, 0].axvline(0, color="0.5", lw=0.8)
    ax[0, 0].set_xlim(-6, 6); ax[0, 0].set_ylim(-110, 5)
    ax[0, 0].set_title("Asymmetric magnitude |C(f)| — passes +f, notches −f")
    ax[0, 0].set_xlabel("Frequency [kHz]"); ax[0, 0].set_ylabel("Level [dB]")
    ax[0, 0].grid(True, alpha=0.4); ax[0, 0].legend(fontsize=8, loc="lower right")

    # --- (b) image rejection vs frequency, per N --------------------------
    for N in Ns:
        Cp = np.abs(complex_response(designs[N], wband, s))
        Cm = np.abs(complex_response(designs[N], -wband, s))
        irr = 20 * np.log10(Cp / np.maximum(Cm, 1e-15))
        ax[0, 1].semilogx(fband, irr, color=cols[N],
                          label="N=%d (worst %.0f dB)" % (N, worst[N]))
    ax[0, 1].axvspan(fa, fb, color="C2", alpha=0.06)
    ax[0, 1].set_ylim(20, 140)
    ax[0, 1].set_title("Image rejection = |C(+f)| / |C(−f)|")
    ax[0, 1].set_xlabel("Frequency [Hz]"); ax[0, 1].set_ylabel("IRR [dB]")
    ax[0, 1].grid(True, which="both", alpha=0.4); ax[0, 1].legend(fontsize=8)

    # --- (c) real multi-tone processed: +f passed, −f image suppressed ----
    ax[1, 0].plot(fax / 1000, Ydb, color="C0", lw=0.8)
    ax[1, 0].axvspan(fa / 1000, fb / 1000, color="C2", alpha=0.12, label="wanted (+f)")
    ax[1, 0].axvspan(-fb / 1000, -fa / 1000, color="C3", alpha=0.12, label="image (−f)")
    ax[1, 0].axvline(0, color="0.5", lw=0.8)
    ax[1, 0].set_xlim(-4, 4); ax[1, 0].set_ylim(-120, 5)
    ax[1, 0].set_title("Real multi-tone through C(z) (N=4): image (−f) suppressed")
    ax[1, 0].set_xlabel("Frequency [kHz]"); ax[1, 0].set_ylabel("Level [dB]")
    ax[1, 0].grid(True, alpha=0.4); ax[1, 0].legend(fontsize=8, loc="upper left")

    # --- (d) worst-case IRR vs section count ------------------------------
    Nsv = np.array(Ns)
    ax[1, 1].plot(Nsv, [worst[N] for N in Ns], "o-", color="C4")
    for N in Ns:
        ax[1, 1].annotate("%.0f dB" % worst[N], (N, worst[N]),
                          textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax[1, 1].set_title("Worst-case image rejection vs. sections (≈22 dB/section)")
    ax[1, 1].set_xlabel("sections per leg N"); ax[1, 1].set_ylabel("worst IRR [dB]")
    ax[1, 1].grid(True, alpha=0.4); ax[1, 1].set_xticks(Nsv)

    fig.suptitle("Complex all-pass polyphase image-reject filter  C(z) = A0(z) + j·A1(z)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "polyphase_complex.png"
    fig.savefig(out, dpi=110)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
