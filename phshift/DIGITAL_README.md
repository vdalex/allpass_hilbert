# digital_phshift.py — Digital all-pass phase-shifter designer (Python)

Python port of the analog `phshift.m` methodology, producing coefficients for a
**digital** all-pass phase-shift network (Hilbert / 90° phase splitter).

## Methodology

Identical optimisation to the MATLAB original:

- Two legs (outputs **I** and **Q**), each a cascade of `N` first-order all-pass sections.
- Free parameters: the 90°-frequencies `f0` of every section (in log space).
- **Nelder-Mead simplex** (`scipy.optimize.fmin`, the equivalent of MATLAB `fminsearch`)
  adjusts the `f0` so that `phase(leg2) − phase(leg1) = phdesired` across the band `[fa, fb]`.
- Cost = **p-norm** of the phase error (`p=2` → least squares, `p≈10` → equal-ripple / minimax).

Every analog 90°-frequency `f0` is mapped to a digital first-order all-pass coefficient
with the **bilinear transform** (the same formula used in `../analog_to_digital_allpass.ods`):

```
c = (1 − ω₀·T/2) / (1 + ω₀·T/2),    ω₀ = 2π·f0,    T = 1/Fs
```

giving the standard real first-order all-pass section

```
H(z) = (z⁻¹ − c) / (1 − c·z⁻¹)          pole at z = c,  |c| < 1
y[n] = −c·x[n] + x[n−1] + c·y[n−1]      difference equation
```

## Second-order (biquad) sections

Two cascaded first-order sections combine **exactly** into one 2nd-order all-pass
biquad (the "dual stage" of the AES paper / the *2-Order All-Pass Filter* table in the
`.ods`). Grouping does not change the phase response but halves the number of filter
blocks — handy for DSP hardware / libraries built around biquads:

```
H(z) = (a2 + a1·z⁻¹ + z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
a1 = −(c1 + c2),   a2 = c1·c2                    (c1, c2 = the two first-order poles)
y[n] = a2·x[n] + a1·x[n−1] + x[n−2] − a1·y[n−1] − a2·y[n−2]
```

`report()` prints both the first-order coefficients and the biquad form (and verifies the
grouping reproduces the phase to ~1e-12°). Adjacent frequency-sorted sections are paired;
with an odd `N` the highest section is left first-order. Programmatic access:

```python
leg1, leg2 = dp.biquads(d)          # list of {"order":2,"a1":..,"a2":..} / {"order":1,"c":..}
```

## Two design modes

The `method` argument selects **where** the bilinear conversion happens:

| Mode | `method` | What it does | 90° accuracy near Nyquist |
|------|----------|--------------|---------------------------|
| **One-step** (default) | `"digital"` | Optimiser evaluates the **digital** phase directly, so it compensates for bilinear warping. | best |
| **Two-step** | `"analog"` | Optimiser designs the **analog** network (`phfunc.m` model, no Fs), then each `f0` is bilinear-converted. Exactly the `../analog_to_digital_allpass.ods` workflow. | drifts (warping) |

The two-step mode reproduces the original MATLAB analog design frequencies (e.g. for
90°, N=4, 270–3600 Hz it gives Leg 1 = 224.35 / 756.79 / 2232.80 / 14070 Hz, matching the
repo README). It is Fs-independent in the search and simpler, but because the analog phase
is fitted (not the digital one) the digital phase difference drifts from 90° toward the
upper band edge. The one-step mode fits the digital phase itself and is more accurate.

Example (90°, N=4, 270–3600 Hz, Fs=22050), max phase error in band:

```
digital (one-step):  0.030°     analog (two-step, .ods):  0.110°
```

### Mode comparison

![Mode comparison](digital_phshift_compare.png)

Both modes are near-identical through most of the band (~±0.015° equiripple). The
difference appears at the **upper band edge**: the two-step design (red) fits the *analog*
phase, so after bilinear warping its digital phase error grows to ≈0.11° near 3600 Hz,
while the one-step design (blue) fits the *digital* phase and stays within ±0.03°. The gap
widens the closer `fb` is pushed toward Nyquist.

Regenerate this figure with:

```python
import digital_phshift as dp
dd = dp.design(90, 4, 270, 3600, 22050, method="digital")
da = dp.design(90, 4, 270, 3600, 22050, method="analog")
dp.plot_compare([dd, da],
                labels=["digital (one-step)", "analog+bilinear (two-step, .ods)"],
                save_prefix="digital_phshift_compare", show=False)
```

## Requirements

`python -m pip install numpy scipy matplotlib`

## Usage

Interactive (like the MATLAB script):

```
python digital_phshift.py
```

It asks for: desired phase (90 = Hilbert), sections per leg `N`, sampling frequency `Fs`,
band `[fa, fb]` (must be < Fs/2), the p-norm, and the design mode (`digital` or `analog`).
It prints the coefficients and saves a phase plot (`digital_phshift_result.png`).

Programmatic:

```python
import digital_phshift as dp

# one-step (default)
d = dp.design(phdesired=90, N=4, fa=270, fb=3600, fs=22050, pnorm=2)
# two-step (matches analog_to_digital_allpass.ods)
d = dp.design(phdesired=90, N=4, fa=270, fb=3600, fs=22050, pnorm=2, method="analog")

dp.report(d)          # print coefficients + worst-case phase error
dp.plot(d, save_prefix="result", show=False)

d["c"]    # (N, 2) digital all-pass coefficients (poles), leg 1 and leg 2
d["f0"]   # (N, 2) analog 90° design frequencies (Hz)
d["fd"]   # (N, 2) actual digital 90° frequencies (Hz)
```

## Example (90°, N=4, 270–3600 Hz, Fs=22050)

Max phase error ≈ **0.03°** across the band. Coefficients `c` for each section
(`H(z) = (z⁻¹−c)/(1−c·z⁻¹)`):

| Leg      | c₁      | c₂      | c₃      | c₄       |
|----------|---------|---------|---------|----------|
| 1 (I)    | 0.93645 | 0.79827 | 0.49666 | −0.36323 |
| 2 (Q)    | 0.98006 | 0.87962 | 0.67794 | 0.20715  |
