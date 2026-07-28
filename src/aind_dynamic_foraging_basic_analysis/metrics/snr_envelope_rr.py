"""
Envelope + rise-rate (Env+RR) signal-to-noise estimator for 1D
fluorescence (dF/F) traces.

:class:`EnvelopeRRSNR` estimates SNR by:

1. Tracking a **tonic baseline** beneath the trace (``tonic_method``:
   ``'arpls'`` (default, adaptive asymmetric least squares), ``'als'``
   (fixed-asymmetry ALS), or ``'envelope'`` (valley/local-minima
   tracking)).
2. Subtracting it to get a **phasic residual**.
3. Estimating the noise floor from that residual (``noise_method``:
   ``'aind_mad'`` (default), ``'folded_iqr'``, or ``'mad_iqr_avg'``).
4. Detecting phasic peaks above ``peak_threshold_sd`` and refining them
   with a rise-rate gate that rejects slow drifts.
5. Reporting **total SNR** as ``snr_tonic + snr_phasic``: ``snr_tonic``
   is the tonic baseline's own peak-to-peak range divided by the noise
   floor, and ``snr_phasic`` is a suprathreshold peak-amplitude
   statistic divided by the noise floor, optionally bias-corrected
   (``bias_correction``; fit against ``snr_phasic`` specifically, see
   ``EnvelopeRRResult``).

Class API
---------
Construct once, call :meth:`~EnvelopeRRSNR.fit` per trace for the full
result (``snr_`` (total), ``snr_tonic_``, ``snr_phasic_``, ``noise_``,
``peaks_``, ``tonic_``, ``residual_``, ...), :meth:`~EnvelopeRRSNR.estimate`
for a one-shot ``(snr, noise, peaks)`` tuple (``snr`` is the total;
drop-in compatible with a plain ``estimate_snr(trace, fps)`` function),
:meth:`~EnvelopeRRSNR.estimate_components` for the one-shot
``(snr_total, snr_tonic, snr_phasic)`` breakdown, or
:meth:`~EnvelopeRRSNR.decompose` for just ``(tonic, phasic)``.

Notes
-----
- Feed a dF/F preprocessed trace (peak height is interpreted from zero).
- Default ``fps`` is 20 Hz; NaNs are filled with the trace median.
- Needs only numpy/scipy -- no other dependencies for any ``noise_method``.

Example: random noise with sinusoidal tonic baseline (SNR ~ 10)
and no real phasic transients (SNR ~ 0)
-------

>>> import numpy as np
>>> from snr_envelope_rr import EnvelopeRRSNR
>>> rng = np.random.default_rng(2)
>>> t = np.arange(1200) / 20.0
>>> noise_floor = 0.01
>>> tonic_amp = 5 * noise_floor # 10 * noise floor
>>> y = tonic_amp * np.sin(2 * np.pi * t / 40.0) + noise_floor * rng.standard_normal(1200)
>>> estimator = EnvelopeRRSNR(fps=20.0)
>>> result = estimator.fit(y)
>>> snr, noise, peaks = estimator.estimate(y)        # one-shot form
>>> print(f"total SNR: {snr}")
total SNR: 12.965802243331636
>>> print(f"noise estimate (true=0.01): {noise}")
noise estimate (true=0.01): 0.009333765351685406
>>> print(f"detected peaks: {peaks}")
detected peaks: [ 56 105 291 348 604 724]
>>> print(f"tonic SNR (true=10.0): {result.snr_tonic:.2f}")
tonic SNR (true=10.0): 10.17
>>> print(f"phasic SNR (true=0.0): {result.snr_phasic:.2f}")
phasic SNR (true=0.0): 2.80
True

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, ndimage, signal

__all__ = ["EnvelopeRRSNR", "EnvelopeRRResult"]

# Tuned defaults at the reference frame rate below. peak_threshold_sd is a
# sigma multiplier (unitless, does not scale with fps). The other three are
# window sizes in *samples*; scale_window() rescales them to preserve their
# real-world duration at a different fps.
_REFERENCE_FPS = 20.0
_TUNED_DEFAULTS = {
    # Grid search (tonic_method='arpls', the default) found
    # peak_threshold_sd=2.0 optimal for noise_method='aind_mad'
    # (score=0.762; arpls_lam=1e7, arpls_n_iter=15, rise_window=5).
    "peak_threshold_sd": 2.0,
    # lower_min_distance/lower_smooth_window only apply if
    # tonic_method='envelope' is explicitly chosen (default is 'arpls',
    # which uses arpls_lam/arpls_n_iter/arpls_ratio instead).
    "lower_min_distance": 20,  # 1.00 s at 20 fps
    "lower_smooth_window": 31,  # 1.55 s at 20 fps
    "rise_window": 5,  # 0.25 s at 20 fps
}

# Linear bias correction fit against ground-truth SNR for the tuned
# defaults above (noise_method='aind_mad', peak_threshold_sd=2.0,
# tonic_method='arpls'): snr_corrected = (snr_raw - intercept) / slope.
# R^2=0.9462. Auto-applied as the constructor's default bias_correction
# only when the effective config matches these defaults exactly (see
# __init__); pass bias_correction=None to disable, or refit via
# fit_bias_correction_from_benchmark for a different configuration.
_TUNED_BIAS_CORRECTION: Tuple[float, float] = (1.8494, -0.5508)

# Only used if tonic_range_method='robust' -- samples at the reference
# fps above (~5.0s); not tuned/validated like _TUNED_DEFAULTS, just a
# reasonable starting point (see _robust_tonic_range's docstring). Scaled
# to the instance's actual fps in __init__, same as _TUNED_DEFAULTS.
_DEFAULT_TONIC_ROBUST_MIN_DISTANCE = 100


class _UnsetType:
    """Sentinel distinguishing "bias_correction not specified" (auto-
    resolve) from an explicit ``bias_correction=None`` (disable)."""

    def __repr__(self) -> str:
        """Return a short, unambiguous marker for debugging/repr output."""
        return "<unset>"


_UNSET = _UnsetType()

# Noise-floor estimator for sigma_fit. 'aind_mad' (default): median-
# filtered residual + twice-trimmed scaled MAD (see _mad_noise_std;
# reproduces aind-ophys-utils' noise_std(method='mad') exactly, cloned
# locally so this module has no dependency on that package). 'folded_iqr':
# folds the residual around its own half-sample mode, scales the IQR.
# 'mad_iqr_avg': mean of the two.
# 'mad' is a deprecated alias for 'aind_mad' (DeprecationWarning).
_VALID_NOISE_METHODS = ("aind_mad", "folded_iqr", "mad_iqr_avg")
_DEFAULT_NOISE_METHOD = "aind_mad"
_DEPRECATED_NOISE_METHOD_ALIASES = {"mad": "aind_mad"}

# Remaining envelope-decomposition parameters; overridable via `config`.
_DEFAULT_CONFIG: Dict = {
    "interp_kind": "pchip",  # 'pchip' or 'linear' envelope interpolation
    # 'arpls' (default): envelope's local-minima tonic tracking collapses
    # at higher SNR (large transients distort where minima land); arPLS
    # held up across the full SNR range tested. _TUNED_BIAS_CORRECTION
    # is fit against this default -- switching tonic_method disables it
    # automatically (see bias_correction's resolution in __init__).
    "tonic_method": "arpls",  # 'envelope' (valley-tracked), 'als', or 'arpls'
    "als_lam": 1e7,  # ALS smoothness (only if tonic_method='als')
    "als_p": 0.01,  # ALS asymmetry, rides under peaks
    "als_n_iter": 10,
    "arpls_lam": 1e7,  # arPLS smoothness (only if tonic_method='arpls')
    "arpls_n_iter": 15,
    "arpls_ratio": 1e-6,
    "midpoint_window": 101,  # smoothing window for the midpoint reference curve
    "midpoint_polyorder": 1,
    "lower_order": 2,  # local-minimum order for valley detection
    "upper_min_distance": 3,  # min spacing enforced by find_peaks on the residual
}


# ======================================================================
# Private helpers
# ======================================================================


def _half_sample_mode(x: NDArray[np.floating]) -> float:
    """Robust mode estimator (half-sample mode; Bickel & Fruhwirth-
    Schnatter 2006). Recursively narrows to the shortest contiguous half
    of the sorted data until <=3 points remain, then returns their
    center. No external dependency; O(n log n).
    """
    data = np.sort(np.asarray(x, dtype=np.float64))
    while len(data) > 3:
        n = len(data)
        half_n = (n + 1) // 2
        widths = data[half_n - 1:] - data[:n - half_n + 1]
        i = int(np.argmin(widths))
        data = data[i:i + half_n]
    if len(data) == 1:
        return float(data[0])
    if len(data) == 2:
        return float(np.mean(data))
    d1, d2 = data[1] - data[0], data[2] - data[1]
    if d1 < d2:
        return float(np.mean(data[:2]))
    elif d2 < d1:
        return float(np.mean(data[1:]))
    else:
        return float(data[1])


def _folded_iqr_noise_std(residual: NDArray[np.floating]) -> float:
    """Robust noise sigma via folded (one-sided) IQR, mode-anchored.

    Reflects the below-mode half of the residual around its own
    half-sample mode (:func:`_half_sample_mode`) and scales the IQR by
    1.349. Only ever looks at the below-mode half, so it never sees
    above-mode phasic transients -- unlike a two-sided estimator (e.g.
    :func:`_mad_noise_std`).
    """
    residual = np.asarray(residual, dtype=np.float64)
    anchor = _half_sample_mode(residual)
    below = residual[residual < anchor]
    if len(below) >= 20:
        reflected = anchor - below
        q75r, q25r = float(np.percentile(reflected, 75)), float(np.percentile(reflected, 25))
        return (q75r - q25r) / 1.349
    return float(np.std(residual[residual <= np.median(residual)]))


def _robust_std(x: NDArray[np.floating]) -> float:
    """Scaled median absolute deviation, assuming near-Gaussian noise.

    Local clone of aind-ophys-utils' ``robust_std`` (1D, no-NaN case) --
    see :func:`_mad_noise_std`.
    """
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _mad_noise_std(residual: NDArray[np.floating], filter_length: int = 31) -> float:
    """Robust noise sigma via a median-filtered residual + twice-trimmed
    scaled MAD.

    Local, dependency-free clone of aind-ophys-utils'
    ``noise_std(x, method='mad')`` (the 1D, ``skipna=False`` case, which
    is all this module needs) -- reproduces it exactly (validated
    against the original numerically) without requiring that package,
    which pulls in pytorch even for this method (a module-level default
    argument elsewhere in that package evaluates ``torch.cuda.is_available()``
    at import time).

    Median-filters the residual to remove any remaining slow baseline,
    takes what's left, trims positive-peak outliers, then trims any
    remaining outliers on either side, and returns the scaled MAD of
    that twice-trimmed remainder.

    Falls back to a less-trimmed robust std (rather than NaN) if a
    trimming step empties out completely -- this happens on short or
    heavily-anomalous windows (e.g. one chunk of a chunked/windowed
    estimate landing squarely on a large artifact) where the first
    trim's own scale estimate collapses to 0. Silently returning NaN
    there would poison any downstream aggregation across windows
    (``np.median`` propagates a single NaN to the whole result).
    """
    residual = np.asarray(residual, dtype=np.float64)
    if np.any(np.isnan(residual)):
        return float("nan")
    baseline = ndimage.percentile_filter(residual, 50, size=filter_length, mode="reflect")
    noise = residual - baseline
    if noise.size == 0:
        return float("nan")
    filtered_0 = noise[noise < 1.5 * np.abs(noise.min())]
    rstd = _robust_std(filtered_0)
    filtered_1 = filtered_0[np.abs(filtered_0) < 2.5 * rstd] if rstd > 0 else np.array([])
    if filtered_1.size > 0:
        result = _robust_std(filtered_1)
        if result > 0:
            return result
    if rstd > 0:
        return rstd
    return _robust_std(noise)


def _moving_average(
    x: NDArray[np.floating], window: int, polyorder: int = 1
) -> NDArray[np.floating]:
    """Savitzky-Golay smoothed reference curve, used as the midpoint."""
    window = window if window % 2 == 1 else window + 1
    window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    return signal.savgol_filter(x, window_length=window, polyorder=polyorder)


def _interpolate_envelope(idx_anchored, val_anchored, n, interp_kind: str = "pchip"):
    """Interpolate anchor points (e.g. tonic valleys) to a full-length curve."""
    query = np.arange(n)
    if interp_kind == "linear":
        return np.interp(query, idx_anchored, val_anchored)
    elif interp_kind == "pchip":
        return interpolate.PchipInterpolator(idx_anchored, val_anchored, extrapolate=True)(query)
    else:
        raise ValueError(f"Unknown interp_kind: {interp_kind!r}. Use 'linear' or 'pchip'.")


def _lower_envelope(x, midpoint, smooth_window, order, min_distance, interp_kind="pchip"):
    """Valley-tracked tonic (slow baseline) envelope beneath ``x``."""
    if smooth_window % 2 == 0:
        smooth_window += 1
    x_smooth = signal.savgol_filter(x, window_length=smooth_window, polyorder=3)

    minima_idx = signal.argrelmin(x_smooth, order=order)[0]
    if len(minima_idx) > 0:
        minima_idx = minima_idx[x[minima_idx] < midpoint[minima_idx]]

    if len(minima_idx) > 1:
        filtered = [minima_idx[0]]
        for idx in minima_idx[1:]:
            if idx - filtered[-1] >= min_distance:
                filtered.append(idx)
        minima_idx = np.array(filtered)

    if len(minima_idx) < 2:
        return np.full_like(x, np.min(x), dtype=float), minima_idx

    idx_anchored = np.concatenate([[0], minima_idx, [len(x) - 1]])
    val_anchored = x[idx_anchored]
    return _interpolate_envelope(idx_anchored, val_anchored, len(x), interp_kind), minima_idx


def _als_baseline(y, lam: float = 1e7, p: float = 0.01, n_iter: int = 10):
    """Asymmetric Least Squares (ALS) smooth curve, ridden under peaks
    when ``p`` is small (``tonic_method='als'``).

    Has a sharp trade-off in its single fixed ``p``: small p (~0.01)
    resists large transients but is biased at low SNR; larger p
    (~0.2-0.3) is good at low SNR but collapses at high SNR. See
    ``tonic_method='arpls'`` (:func:`_arpls_baseline`), which adapts the
    asymmetry per trace instead of fixing it in advance.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _arpls_baseline(y, lam: float = 1e7, n_iter: int = 15, ratio: float = 1e-6):
    """Asymmetrically Reweighted Penalized Least Squares (arPLS) smooth
    curve (Baek et al. 2015; ``tonic_method='arpls'``).

    Like :func:`_als_baseline`, but the asymmetry weights are re-derived
    from the current residual's own below-baseline noise statistics at
    every iteration (same one-sided logic as
    :func:`_folded_iqr_noise_std`) instead of one fixed `p` chosen in
    advance -- resolving ALS's low/high-SNR trade-off in testing.
    Iterates until the relative weight change drops below `ratio` or
    `n_iter` is reached.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        d = y - z
        d_neg = d[d < 0]
        m = float(np.mean(d_neg)) if len(d_neg) else 0.0
        s = float(np.std(d_neg)) if len(d_neg) else 1e-12
        s = s if s > 1e-12 else 1e-12
        w_new = 1.0 / (1.0 + np.exp(np.clip(2 * (d - (2 * s - m)) / s, -500, 500)))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return z


def _detect_peaks_rise_rate(residual, candidates, sigma, rise_window: int = 3):
    """Reject candidate peaks whose approach isn't fast enough to be a
    real transient onset (vs. a slow drift crossing threshold).

    Calibrates its slope threshold from a robust (MAD-based) scale
    estimate of the below-median ("noise-like") candidate slopes, not
    an ordinary standard deviation (``scipy.stats.norm.fit``, what this
    used before). Ordinary std is itself not robust: a single moderate
    anomaly elsewhere in the trace can spawn a few large-magnitude
    candidate slopes from its own decay tail, and even just one or two
    of those landing in the "below-median" bucket can inflate an
    ordinary std by 2x+ -- which pushes slope_thresh above genuine
    events' own slopes and silently suppresses their detection
    (measured: one moderate, localized anomaly cut detected event count
    from 73 to 19 out of ~80 genuine events, purely through this
    threshold-calibration side effect, though the anomaly itself was
    nowhere near most of the suppressed events). The MAD-based estimate
    used here left detection completely unchanged (73 to 73) on that
    same case.
    """
    if len(candidates) < 5:
        return candidates

    slopes = []
    for idx in candidates:
        lo = max(0, idx - rise_window)
        window = residual[lo:idx]
        if len(window) >= 2:
            slopes.append(float(np.mean(np.diff(window))))
        else:
            slopes.append(0.0)
    slopes = np.array(slopes)

    lower = slopes[slopes <= np.median(slopes)]
    if len(lower) >= 5:
        sigma_slope = 1.4826 * np.median(np.abs(lower - np.median(lower)))
        slope_thresh = max(sigma_slope * 3.0, sigma * 0.1)
    else:
        slope_thresh = np.percentile(slopes, 50)

    return candidates[slopes > slope_thresh]


def _despike_interpolate(
    x: NDArray[np.floating], window: int, k: float
) -> Tuple[NDArray[np.floating], NDArray[np.bool_], float]:
    """Flag samples far from a local (rolling-median) baseline and
    replace them via linear interpolation from the nearest un-flagged
    samples on either side -- run *before* tonic fitting, not as a
    replacement for it.

    Why this and not a fixed-ceiling clip: clipping a decaying artifact
    to a hard ceiling turns it into a flat plateau, which a smoothness-
    penalized fit (arPLS/ALS) can end up tracking *more* readily than
    the original decay, since a sustained flat elevation looks more
    like genuine slow drift than a decaying transient does -- clipping
    can make tonic contamination worse, not better (measured directly:
    a naive global clip at 50x a robust noise estimate roughly
    tripled downstream tonic contamination in one test case). Removing
    the flagged span and interpolating over it avoids introducing that
    new, more-trackable shape.

    Parameters
    ----------
    x : ndarray
    window : int
        Rolling-median window (samples) for the local baseline and
        local MAD estimate.
    k : float
        Flag samples where ``|x - rolling_median| > k * local_MAD``.

    Returns
    -------
    x_clean : ndarray
        Copy of ``x`` with flagged samples replaced.
    flagged : ndarray of bool
    local_noise_est : float
        Median of the local MAD estimate across the trace (diagnostic).
    """
    rolling_med = ndimage.median_filter(x, size=window, mode="reflect")
    dev = x - rolling_med
    local_mad = 1.4826 * ndimage.median_filter(np.abs(dev), size=window, mode="reflect")
    local_mad = np.maximum(local_mad, 1e-12)
    flagged = np.abs(dev) > k * local_mad

    x_clean = x.copy()
    if flagged.any() and not flagged.all():
        idx = np.arange(len(x))
        x_clean[flagged] = np.interp(idx[flagged], idx[~flagged], x[~flagged])

    return x_clean, flagged, float(np.median(local_mad))


def _robust_tonic_range(tonic: NDArray[np.floating], min_distance: int) -> Tuple[float, int]:
    """Median peak-to-valley swing amplitude of the tonic curve --
    mirrors how phasic amplitude is estimated (detect individual
    events, take the median across them) instead of ``ptp(tonic)``'s
    single global max-minus-min.

    Detects local maxima and minima in ``tonic``
    (``scipy.signal.argrelextrema``, ``order=min_distance`` -- the
    number of samples on each side a point must exceed to count as a
    local extremum), takes the absolute difference between each pair of
    temporally-adjacent extrema (each one a single peak-to-valley or
    valley-to-peak "swing"), and returns the median swing amplitude.

    A one-off contaminated region typically produces only one or two
    anomalous swings among many genuine ones, so the median has a real
    breakdown point -- unlike ``ptp(tonic)``, which *is* the
    contaminated region's own extreme value whenever that region
    happens to contain the global max or min. Unlike a chunked/windowed
    estimate (:meth:`EnvelopeRRSNR.fit_chunked`), this doesn't impose an
    arbitrary fixed-duration window that can slice a genuine slow cycle
    in half and underestimate it by construction -- it finds extrema
    wherever the curve's own structure actually puts them, the same
    principle ``find_peaks`` already uses for phasic events.

    The trade-off: robustness here scales with how many genuine swings
    get detected, same as phasic's median scales with how many genuine
    events get detected. A short recording relative to the tonic's own
    drift period (few genuine cycles) gives the median little to work
    with, same limitation phasic would have with only a handful of true
    transients.

    Parameters
    ----------
    tonic : ndarray
        Fitted tonic curve.
    min_distance : int
        Minimum spacing (samples) between detected extrema -- should be
        well below the tonic's own characteristic drift period, but
        well above any remaining fast wiggle in the fitted curve.

    Returns
    -------
    tonic_range : float
        Median swing amplitude, or ``ptp(tonic)`` if fewer than 2
        extrema are detected (not enough structure for a meaningful
        median -- e.g. a very short trace or an almost perfectly flat
        tonic).
    n_swings : int
        Number of swings the median was computed over (0 if it fell
        back to ``ptp``).
    """
    peaks = signal.argrelextrema(tonic, np.greater, order=min_distance)[0]
    valleys = signal.argrelextrema(tonic, np.less, order=min_distance)[0]
    extrema_idx = np.sort(np.concatenate([peaks, valleys]))
    if len(extrema_idx) < 2:
        return float(np.ptp(tonic)), 0
    swings = np.abs(np.diff(tonic[extrema_idx]))
    return float(np.median(swings)), len(swings)


def _apply_pre_despike(
    x: NDArray[np.floating], cfg: Dict
) -> Tuple[NDArray[np.floating], int, float]:
    """Optionally despike ``x`` before it's used for tonic fitting.

    Off by default (``cfg['pre_despike_window']`` is ``None``),
    preserving prior behavior exactly unless opted in. Scoped to
    protecting the tonic fit only -- callers should run residual/peak
    detection on the ORIGINAL ``x``, not this function's output, so a
    genuine large artifact still surfaces as an inspectable outlier
    rather than silently vanishing. Even with despiking on, a large
    enough outlier relative to the true tonic's own dynamic range can
    still leak through a smoothness-penalized fit;
    ``n_extreme_samples``/``frac_extreme_samples`` are diagnostics for
    exactly that residual risk, not a guarantee despiking fully
    removed it.

    Returns
    -------
    x_for_tonic : ndarray
        ``x`` unchanged if pre-despiking is off, else the despiked copy.
    n_extreme_samples : int
    frac_extreme_samples : float
    """
    pre_despike_window = cfg.get("pre_despike_window", None)
    if pre_despike_window is None:
        return x, 0, 0.0
    x_for_tonic, flagged, _local_noise = _despike_interpolate(
        x, window=pre_despike_window, k=cfg.get("pre_despike_k", 5.0)
    )
    n_extreme_samples = int(np.sum(flagged))
    frac_extreme_samples = float(n_extreme_samples / len(x))
    return x_for_tonic, n_extreme_samples, frac_extreme_samples


def _fit_tonic_curve(
    x_for_tonic: NDArray[np.floating],
    midpoint: NDArray[np.floating],
    tonic_method: str,
    cfg: Dict,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Fit the tonic (slow baseline) curve with the requested tracker.

    Parameters
    ----------
    x_for_tonic : ndarray
        Trace to fit (already despiked if pre-despiking was applied).
    midpoint : ndarray
        Smoothed reference curve; only used by ``tonic_method='envelope'``.
    tonic_method : {'als', 'arpls', 'envelope'}
    cfg : dict
        Method-specific tuning knobs (``als_lam``/``als_p``/``als_n_iter``,
        ``arpls_lam``/``arpls_n_iter``/``arpls_ratio``, or
        ``lower_smooth_window``/``lower_order``/``lower_min_distance``/
        ``interp_kind``).

    Returns
    -------
    tonic : ndarray
    tonic_minima : ndarray of int
        Valley indices tracked by the envelope method; empty for
        ``'als'``/``'arpls'`` (they have no discrete "minima" concept).

    Raises
    ------
    ValueError
        If ``tonic_method`` isn't one of the three supported values.
    """
    if tonic_method == "als":
        tonic = _als_baseline(
            x_for_tonic,
            lam=cfg.get("als_lam", 1e7),
            p=cfg.get("als_p", 0.01),
            n_iter=cfg.get("als_n_iter", 10),
        )
        return tonic, np.array([], dtype=int)
    elif tonic_method == "arpls":
        tonic = _arpls_baseline(
            x_for_tonic,
            lam=cfg.get("arpls_lam", 1e7),
            n_iter=cfg.get("arpls_n_iter", 15),
            ratio=cfg.get("arpls_ratio", 1e-6),
        )
        return tonic, np.array([], dtype=int)
    elif tonic_method == "envelope":
        return _lower_envelope(
            x_for_tonic,
            midpoint,
            cfg["lower_smooth_window"],
            cfg["lower_order"],
            cfg["lower_min_distance"],
            interp_kind=cfg["interp_kind"],
        )
    raise ValueError(f"tonic_method must be 'envelope', 'als', or 'arpls'; got {tonic_method!r}.")


def _resolve_noise_method(noise_method: str) -> str:
    """Resolve a deprecated ``noise_method`` alias (e.g. ``'mad'``) to
    its canonical name (``'aind_mad'``); returns unrecognized names
    unchanged so the caller's own validation can reject them."""
    return _DEPRECATED_NOISE_METHOD_ALIASES.get(noise_method, noise_method)


def _estimate_residual_noise(residual: NDArray[np.floating], noise_method: str) -> float:
    """Dispatch to the requested noise-floor estimator on the tonic-
    subtracted residual.

    Parameters
    ----------
    residual : ndarray
    noise_method : {'aind_mad', 'folded_iqr', 'mad_iqr_avg'}
        Deprecated aliases (e.g. ``'mad'``) are resolved first via
        :func:`_resolve_noise_method`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``noise_method`` (after alias resolution) isn't one of the
        three supported values.
    """
    noise_method = _resolve_noise_method(noise_method)
    if noise_method == "folded_iqr":
        return _folded_iqr_noise_std(residual)
    elif noise_method == "aind_mad":
        return _mad_noise_std(residual)
    elif noise_method == "mad_iqr_avg":
        return 0.5 * (_folded_iqr_noise_std(residual) + _mad_noise_std(residual))
    raise ValueError(f"noise_method must be one of {_VALID_NOISE_METHODS}; got {noise_method!r}.")


def _estimate_sigma_minima(
    residual: NDArray[np.floating],
    tonic_minima: NDArray[np.intp],
    fallback_sigma: float,
) -> float:
    """Noise estimate from the residual at valley-to-valley midpoints --
    only meaningful for ``tonic_method='envelope'``, which tracks an
    explicit list of minima. Falls back to ``fallback_sigma`` if there
    aren't enough minima (or valid midpoints) for a stable estimate.
    """
    if len(tonic_minima) < 5:
        return fallback_sigma
    n = len(residual)
    mid_idx = ((tonic_minima[:-1] + tonic_minima[1:]) // 2).astype(int)
    mid_idx = mid_idx[(mid_idx >= 0) & (mid_idx < n)]
    if len(mid_idx) < 5:
        return fallback_sigma
    return float(np.std(residual[mid_idx]))


def _detect_phasic_events(
    residual: NDArray[np.floating], sigma_fit: float, cfg: Dict
) -> Tuple[NDArray[np.intp], float, NDArray[np.floating]]:
    """Detect suprathreshold phasic peaks and gate them by rise rate
    (see :func:`_detect_peaks_rise_rate`).

    Returns
    -------
    event_maxima : ndarray of int
    threshold : float
        ``peak_threshold_sd * sigma_fit`` -- the height cutoff used.
    peak_amps : ndarray
        ``residual[event_maxima]``; empty if no events were detected.
    """
    threshold = cfg["peak_threshold_sd"] * sigma_fit
    raw_peaks, _ = signal.find_peaks(
        residual,
        height=threshold,
        distance=cfg.get("upper_min_distance", 3),
    )
    event_maxima = _detect_peaks_rise_rate(
        residual,
        raw_peaks,
        sigma_fit,
        rise_window=cfg.get("rise_window", 3),
    )
    event_maxima = np.asarray(event_maxima, dtype=int)
    event_maxima = event_maxima[(event_maxima >= 0) & (event_maxima < len(residual))]
    peak_amps = residual[event_maxima] if len(event_maxima) > 0 else np.array([])
    return event_maxima, threshold, peak_amps


def _summarize_phasic_amplitudes(
    peak_amps: NDArray[np.floating],
) -> Tuple[float, float, float]:
    """95th percentile / median / std of detected peak amplitudes; all
    zero if no events were detected.

    Returns
    -------
    (phasic_p95, phasic_median, phasic_sd)
    """
    if len(peak_amps) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(peak_amps, 95)),
        float(np.median(peak_amps)),
        float(np.std(peak_amps)),
    )


def _compute_tonic_range(
    tonic: NDArray[np.floating], tonic_range_method: str, cfg: Dict
) -> Tuple[float, int]:
    """Dispatch to the requested tonic-amplitude summary statistic.

    ``ptp(tonic)`` (``max - min``) has a breakdown point of exactly one
    sample -- a single large excursion the tonic fit only partially
    absorbs (e.g. from a sustained artifact arPLS's reweighting doesn't
    fully reject) inflates it directly, with nothing to average it out.
    ``'percentile'`` trims ``tonic_range_trim_pct`` from each tail
    before taking the range, at the cost of also clipping any genuine
    tonic dynamic range that happens to live in that trimmed fraction --
    choose ``tonic_range_trim_pct`` to comfortably exceed the fraction
    of the trace you expect a real artifact to occupy (e.g. a 5s glitch
    in a 150s trace is ~3.3% one-sided; ``trim_pct=5`` gives a 5%
    one-sided margin above that). ``'robust'`` (the default) instead
    takes the median peak-to-valley swing amplitude across detected
    local extrema in the tonic curve (see :func:`_robust_tonic_range`),
    mirroring how phasic amplitude is estimated (detect individual
    events, take the median across them) rather than reading off a
    single global extreme value.

    Returns
    -------
    tonic_range : float
    n_tonic_swings : int
        Only nonzero for ``tonic_range_method='robust'``.

    Raises
    ------
    ValueError
        If ``tonic_range_method`` isn't one of the three supported
        values.
    """
    if tonic_range_method == "ptp":
        return float(np.ptp(tonic)), 0
    elif tonic_range_method == "percentile":
        trim_pct = cfg.get("tonic_range_trim_pct", 5.0)
        tonic_range = float(np.percentile(tonic, 100 - trim_pct) - np.percentile(tonic, trim_pct))
        return tonic_range, 0
    elif tonic_range_method == "robust":
        min_distance = cfg.get("tonic_robust_min_distance", _DEFAULT_TONIC_ROBUST_MIN_DISTANCE)
        return _robust_tonic_range(tonic, min_distance)
    raise ValueError(
        f"tonic_range_method must be 'ptp', 'percentile', or 'robust'; "
        f"got {tonic_range_method!r}."
    )


def _decompose_envelope_rr(x: NDArray[np.floating], config: Dict) -> Dict:
    """Tonic/phasic decomposition + rise-rate gated peak detection.

    A thin orchestrator: optional pre-despiking
    (:func:`_apply_pre_despike`) -> tonic fit (:func:`_fit_tonic_curve`)
    -> noise-floor estimate (:func:`_estimate_residual_noise`,
    :func:`_estimate_sigma_minima`) -> phasic peak detection
    (:func:`_detect_phasic_events`, :func:`_summarize_phasic_amplitudes`)
    -> tonic-amplitude summary statistic (:func:`_compute_tonic_range`).
    See each helper's own docstring for the mechanism and trade-offs of
    the strategy it dispatches between; this function itself makes no
    strategy decisions of its own.
    """
    cfg = config

    x_for_tonic, n_extreme_samples, frac_extreme_samples = _apply_pre_despike(x, cfg)

    midpoint = _moving_average(
        x_for_tonic,
        window=cfg["midpoint_window"],
        polyorder=cfg["midpoint_polyorder"],
    )

    tonic_method = cfg.get("tonic_method", "envelope")
    tonic, tonic_minima = _fit_tonic_curve(x_for_tonic, midpoint, tonic_method, cfg)

    # residual/peak-detection intentionally use the ORIGINAL x, not
    # x_for_tonic -- despiking is scoped to protecting the tonic fit; a
    # genuine large artifact should still surface in the residual/phasic
    # stats as an inspectable outlier rather than being silently erased.
    residual = x - tonic

    noise_method = cfg.get("noise_method", _DEFAULT_NOISE_METHOD)
    sigma_iqr = _estimate_residual_noise(residual, noise_method)
    sigma_minima = _estimate_sigma_minima(residual, tonic_minima, sigma_iqr)
    sigma_fit = sigma_iqr

    event_maxima, thresh, peak_amps = _detect_phasic_events(residual, sigma_fit, cfg)
    phasic_p95, phasic_median, phasic_sd = _summarize_phasic_amplitudes(peak_amps)

    tonic_range_method = cfg.get("tonic_range_method", "robust")
    tonic_range, n_tonic_swings = _compute_tonic_range(tonic, tonic_range_method, cfg)

    return {
        "tonic": tonic,
        "tonic_minima": tonic_minima,
        "event_maxima": event_maxima,
        "midpoint": midpoint,
        "residual": residual,
        "peak_amps": peak_amps,
        "noise_sigma": float(sigma_fit),
        "noise_sigma_iqr": float(sigma_iqr),
        "noise_sigma_minima": float(sigma_minima),
        "peak_threshold": float(thresh),
        "detection_method": "rise_rate",
        "tonic_method": tonic_method,
        "tonic_range": tonic_range,
        "phasic_p95": phasic_p95,
        "phasic_median": phasic_median,
        "phasic_amplitude": phasic_sd,
        "phasic_snr_p95": float(phasic_p95 / sigma_fit) if sigma_fit > 0 else float("nan"),
        "phasic_snr_median": float(phasic_median / sigma_fit) if sigma_fit > 0 else float("nan"),
        "phasic_snr_sd": float(phasic_sd / sigma_fit) if sigma_fit > 0 else float("nan"),
        "tonic_snr_sd": float(tonic_range / sigma_fit) if sigma_fit > 0 else float("nan"),
        "n_extreme_samples": n_extreme_samples,
        "n_tonic_swings": n_tonic_swings,
        "frac_extreme_samples": frac_extreme_samples,
        "config": cfg,
    }


@dataclass
class EnvelopeRRResult:
    """Fitted result of :meth:`EnvelopeRRSNR.fit`.

    Attributes
    ----------
    snr : float
        **Total** SNR: ``snr_tonic + snr_phasic``, before bias
        correction. (Changed from phasic-only in earlier versions of
        this class -- see ``snr_phasic`` for that quantity on its own.)
    snr_tonic : float
        Tonic-baseline SNR: the tonic's own peak-to-peak range divided
        by the noise floor. How much the slow baseline itself varies
        relative to the noise, independent of any phasic events.
    snr_phasic : float
        Phasic SNR: ``signal / noise``, using ``signal_statistic`` on
        the suprathreshold phasic peak amplitudes. This is what ``snr``
        meant in earlier versions of this class, and is the quantity
        ``bias_correction`` was actually fit against (see
        ``snr_phasic_corrected``).
    snr_phasic_corrected : float or None
        ``snr_phasic`` after applying ``bias_correction`` (``None`` if
        none configured, or if no phasic peaks were detected).
    snr_corrected : float or None
        **Total**, bias-corrected: ``snr_tonic + snr_phasic_corrected``
        (``None`` if ``bias_correction`` isn't configured). Since
        ``bias_correction`` was fit only against ``snr_phasic`` (see
        above), this is a derived combination, not itself a directly
        validated correction of the total -- treat it as an estimate
        built from one validated piece and one uncorrected piece.
    noise : float
        Estimated noise floor of the tonic-subtracted residual, via
        ``config['noise_method']``.
    peaks : numpy.ndarray
        Indices of detected phasic event peaks (rise-rate gated).
    tonic : numpy.ndarray
        Fitted tonic (slow baseline), same length as the input trace.
    residual : numpy.ndarray
        Tonic-subtracted residual (``trace - tonic``).
    signal : float
        Suprathreshold peak-amplitude statistic used as
        ``snr_phasic``'s numerator (median or 95th percentile, per
        ``signal_statistic``).
    config : dict
        The resolved configuration used for this fit.
    n_extreme_samples : int
        Count of samples flagged and interpolated over by pre-despiking
        before the tonic fit (0 if ``pre_despike_window`` wasn't set in
        ``config``). A nonzero count doesn't mean the tonic fit is now
        clean -- see the module docstring's note on ``tonic_range`` and
        despiking's actual, partial effectiveness against large,
        sustained artifacts.
    frac_extreme_samples : float
        ``n_extreme_samples / len(trace)``.
    n_chunks : int or None
        Number of chunks actually used to compute ``snr_tonic``, if
        this result came from :meth:`EnvelopeRRSNR.fit_chunked` rather
        than :meth:`EnvelopeRRSNR.fit`. ``None`` after a plain ``fit``.
    chunk_duration_s : float or None
        Chunk length (seconds) used by :meth:`fit_chunked`, if
        applicable.
    per_chunk_tonic_snr : list of float or None
        Each chunk's own ``snr_tonic`` before aggregation, if this
        result came from :meth:`fit_chunked`.
    n_tonic_swings : int
        Number of peak-to-valley swings the median was computed over,
        if ``config['tonic_range_method'] == 'robust'`` (0 otherwise,
        including the fallback-to-``ptp`` case when fewer than 2
        extrema were detected -- see :func:`_robust_tonic_range`).
    """

    snr: float
    snr_tonic: float
    snr_phasic: float
    noise: float
    peaks: NDArray[np.intp]
    tonic: NDArray[np.floating]
    residual: NDArray[np.floating]
    signal: float
    config: Dict = field(repr=False)
    snr_corrected: Optional[float] = None
    snr_phasic_corrected: Optional[float] = None
    n_extreme_samples: int = 0
    frac_extreme_samples: float = 0.0
    n_chunks: Optional[int] = None
    chunk_duration_s: Optional[float] = None
    per_chunk_tonic_snr: Optional[list] = None
    n_tonic_swings: int = 0


class EnvelopeRRSNR:
    """Envelope + rise-rate SNR estimator for 1D fluorescence traces.

    Parameters
    ----------
    fps : float, optional
        Sampling frequency (frames per second), by default ``20.0``.
    peak_threshold_sd : float, optional
        Detection threshold for candidate phasic peaks, in units of the
        estimated noise sigma (unitless; does not scale with ``fps``).
        Defaults to ``2.0``, grid-search-optimal for
        ``noise_method='aind_mad'`` with ``tonic_method='arpls'``
        (score=0.762). ``folded_iqr`` also optimizes at ``2.0``;
        ``mad_iqr_avg`` optimizes at ``2.5`` -- don't assume one
        ``noise_method``'s tuned threshold transfers to another.
    signal_statistic : {'median', 'p95'}, optional
        Statistic of the suprathreshold peak amplitudes used as the SNR
        numerator, by default ``'median'``.
    noise_method : {'aind_mad', 'folded_iqr', 'mad_iqr_avg'}, optional
        Noise-floor estimator on the tonic-subtracted residual, by
        default ``'aind_mad'``.

        - ``'aind_mad'``: local clone of aind-ophys-utils'
          ``noise_std(method='mad')`` (see :func:`_mad_noise_std`) --
          reproduces it exactly without depending on that package.
        - ``'folded_iqr'``: folds the below-mode half of the residual
          (mode via :func:`_half_sample_mode`) and scales the IQR.
        - ``'mad_iqr_avg'``: mean of the two above.

        ``'mad'`` is a deprecated alias for ``'aind_mad'`` (raises
        ``DeprecationWarning``).
    bias_correction : tuple of (slope, intercept), or None, optional
        If given, ``snr_phasic_corrected = (snr_phasic - intercept) /
        slope`` is computed on every fit -- this was fit against
        ``snr_phasic`` specifically (the only quantity ever validated
        against ground truth), not the total ``snr``; ``snr_corrected``
        (the corrected total) is then ``snr_tonic + snr_phasic_corrected``,
        a derived combination rather than an independently-validated
        correction. Defaults to a tuned fit
        (``slope=1.8494, intercept=-0.5508, R^2=0.9462``), but only
        auto-applies when ``noise_method``, ``peak_threshold_sd``, and
        ``tonic_method`` all still match the defaults it was fit
        against (``'aind_mad'``, ``2.0``, ``'arpls'``); change any of
        those and this silently resolves to ``None`` instead of
        misapplying a correction fit for a different configuration.
        Pass ``bias_correction=None`` explicitly to disable it even
        with the tuned defaults. Fit your own via
        :meth:`fit_bias_correction_from_benchmark` or
        ``numpy.polyfit(true_snr, snr_est, 1)`` for other
        configurations -- each sits at its own scale.
    config : dict, optional
        Overrides merged on top of the fps-scaled tuned defaults
        (``lower_min_distance``, ``lower_smooth_window``,
        ``rise_window``) and this module's fixed defaults -- see
        ``_DEFAULT_CONFIG``. Notably ``config={'tonic_method': 'als'}``
        or ``'envelope'`` to change the tonic tracker (default
        ``'arpls'`` -- see :func:`_arpls_baseline`), or
        ``config={'noise_method': ...}``, which takes precedence over
        the ``noise_method`` argument.

        ``tonic_range_method`` (default ``'robust'``) controls how
        ``snr_tonic`` is computed from the fitted tonic curve:
        ``'robust'`` -- median peak-to-valley swing amplitude across
        detected local extrema, mirroring how ``snr_phasic`` is
        estimated (detect individual events, take the median across
        them), spacing controlled by ``tonic_robust_min_distance``
        (default 100 samples at 20fps, fps-scaled) -- or ``'ptp'``
        (``max(tonic) - min(tonic)``, the previous default; a fragile,
        single-sample-breakdown-point statistic -- kept for backward
        compatibility, and still useful as a plain, unadorned baseline
        to compare against), or ``'percentile'`` (trims
        ``tonic_range_trim_pct`` from each tail before taking the
        range). See :func:`_robust_tonic_range` for the full mechanism
        and its own trade-off (robustness scales with how many genuine
        tonic swings actually get detected, same as ``snr_phasic``'s
        robustness scales with detected event count).

    Notes
    -----
    - Noise is estimated from the tonic-subtracted residual, not the
      raw derivative, so it's robust to slow drift and phasic events.
    - If fewer than one phasic peak is found, ``snr_phasic`` is ``NaN``
      (and so is the total ``snr``, via propagation) and a
      ``RuntimeWarning`` is issued; ``snr_tonic`` is still computed.
    - **Window defaults scale with fps.** ``lower_min_distance``,
      ``lower_smooth_window``, and ``rise_window`` are tuned at 20 fps
      and rescaled by :meth:`scale_window` to preserve real-world
      duration at other rates (``lower_smooth_window`` forced odd).
      Pass an explicit value in ``config`` to opt out.
    """

    def __init__(
        self,
        fps: float = 20.0,
        peak_threshold_sd: float = _TUNED_DEFAULTS["peak_threshold_sd"],
        signal_statistic: str = "median",
        noise_method: str = _DEFAULT_NOISE_METHOD,
        bias_correction: Union[Tuple[float, float], None, "_UnsetType"] = _UNSET,
        config: Optional[Dict] = None,
    ) -> None:
        """Construct an estimator; see the class docstring for parameter details."""
        if signal_statistic not in ("median", "p95"):
            raise ValueError(
                f"signal_statistic must be 'median' or 'p95', got {signal_statistic!r}."
            )
        # config={'noise_method': ...} takes precedence over the argument.
        effective_noise_method = (config or {}).get("noise_method", noise_method)
        if effective_noise_method in _DEPRECATED_NOISE_METHOD_ALIASES:
            new_name = _DEPRECATED_NOISE_METHOD_ALIASES[effective_noise_method]
            effective_noise_method = new_name
        if effective_noise_method not in _VALID_NOISE_METHODS:
            raise ValueError(
                f"noise_method must be one of {_VALID_NOISE_METHODS} "
                f"(or the deprecated alias 'mad' for 'aind_mad'), "
                f"got {effective_noise_method!r}."
            )

        # Same precedence, for the other two settings _TUNED_BIAS_CORRECTION
        # was fit against -- needed to decide if it's safe to auto-apply.
        effective_peak_threshold_sd = (config or {}).get("peak_threshold_sd", peak_threshold_sd)
        effective_tonic_method = (config or {}).get("tonic_method", _DEFAULT_CONFIG["tonic_method"])

        if bias_correction is _UNSET:
            # Auto-apply the tuned correction only if noise_method,
            # peak_threshold_sd, and tonic_method all match what it was
            # fit against -- otherwise it'd silently misapply a
            # correction fit for a different configuration.
            matches_tuned_config = (
                effective_noise_method == _DEFAULT_NOISE_METHOD
                and effective_peak_threshold_sd == _TUNED_DEFAULTS["peak_threshold_sd"]
                and effective_tonic_method == _DEFAULT_CONFIG["tonic_method"]
            )
            bias_correction = _TUNED_BIAS_CORRECTION if matches_tuned_config else None

        self.fps = fps
        self.peak_threshold_sd = effective_peak_threshold_sd
        self.signal_statistic = signal_statistic
        self.noise_method = effective_noise_method
        self.bias_correction = bias_correction

        scaled_defaults = {
            "lower_min_distance": self.scale_window(_TUNED_DEFAULTS["lower_min_distance"], fps),
            "lower_smooth_window": self.scale_window(
                _TUNED_DEFAULTS["lower_smooth_window"], fps, make_odd=True
            ),
            "rise_window": self.scale_window(_TUNED_DEFAULTS["rise_window"], fps),
            "tonic_robust_min_distance": self.scale_window(_DEFAULT_TONIC_ROBUST_MIN_DISTANCE, fps),
        }
        self.config = {
            **_DEFAULT_CONFIG,
            "fps": fps,
            "peak_threshold_sd": peak_threshold_sd,
            **scaled_defaults,
            **(config or {}),
            # Set last (resolved, canonical name) so a raw alias in
            # `config` can't override it back to the deprecated spelling.
            "noise_method": effective_noise_method,
        }

        # populated by fit()
        self.result_: Optional[EnvelopeRRResult] = None

    @staticmethod
    def scale_window(
        base_samples: int,
        fps: float,
        reference_fps: float = _REFERENCE_FPS,
        make_odd: bool = False,
    ) -> int:
        """Rescale a sample-count window to a new fps, preserving duration.

        Returns ``round(base_samples * fps / reference_fps)``, floored
        at 1. If ``make_odd``, increments by 1 if the result is even
        (required by e.g. Savitzky-Golay windows).

        Example
        -------
        >>> EnvelopeRRSNR.scale_window(20, fps=40.0)
        40
        """
        n = int(round(base_samples * fps / reference_fps))
        n = max(1, n)
        if make_odd and n % 2 == 0:
            n += 1
        return n

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, trace: NDArray[np.floating]) -> EnvelopeRRResult:
        """Decompose ``trace`` and estimate its SNR.

        NaNs in ``trace`` are replaced with its median first. Returns
        an :class:`EnvelopeRRResult`, also stored on ``self.result_``
        for the convenience properties (``self.snr_``, etc.).
        """
        trace = np.nan_to_num(trace, nan=float(np.nanmedian(trace)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = _decompose_envelope_rr(trace, self.config)

        snr_key = "phasic_snr_p95" if self.signal_statistic == "p95" else "phasic_snr_median"
        sig_key = "phasic_p95" if self.signal_statistic == "p95" else "phasic_median"

        peaks = np.asarray(raw["event_maxima"], dtype=int)
        snr_phasic = float(raw[snr_key])
        snr_tonic = float(raw["tonic_snr_sd"])
        noise = float(raw["noise_sigma"])
        signal = float(raw[sig_key])

        if len(peaks) == 0:
            warnings.warn(
                "No phasic peaks detected above threshold. Returning NaN for snr_phasic.",
                RuntimeWarning,
                stacklevel=2,
            )
            snr_phasic = float("nan")

        snr_total = snr_tonic + snr_phasic  # NaN-propagates if snr_phasic is NaN

        snr_phasic_corrected = None
        snr_corrected = None
        if self.bias_correction is not None and not np.isnan(snr_phasic):
            # bias_correction was fit against snr_phasic specifically (the
            # only quantity ever validated against ground truth) -- see
            # EnvelopeRRResult's docstring for why snr_corrected (a total)
            # is therefore a derived combination, not itself validated.
            slope, intercept = self.bias_correction
            snr_phasic_corrected = float((snr_phasic - intercept) / slope)
            snr_corrected = snr_tonic + snr_phasic_corrected

        self.result_ = EnvelopeRRResult(
            snr=snr_total,
            snr_tonic=snr_tonic,
            snr_phasic=snr_phasic,
            snr_corrected=snr_corrected,
            snr_phasic_corrected=snr_phasic_corrected,
            noise=noise,
            peaks=peaks,
            tonic=raw["tonic"],
            residual=raw["residual"],
            signal=signal,
            config=raw["config"],
            n_extreme_samples=raw["n_extreme_samples"],
            frac_extreme_samples=raw["frac_extreme_samples"],
            n_tonic_swings=raw["n_tonic_swings"],
        )
        return self.result_

    def fit_chunked(
        self,
        trace: NDArray[np.floating],
        chunk_duration_s: Optional[float] = None,
        min_chunk_duration_s: float = 30.0,
        chunk_fraction: float = 0.20,
        aggregate: str = "median",
    ) -> EnvelopeRRResult:
        """Like :meth:`fit`, but computes ``snr_tonic`` from a chunked,
        per-window median instead of the single global fit's
        ``ptp(tonic)``.

        Why: ``ptp(tonic)`` has a breakdown point of exactly one sample
        -- a single large excursion the tonic fit only partially
        absorbs (e.g. a sustained artifact arPLS's reweighting doesn't
        fully reject) inflates it directly, with nothing to average it
        out. Splitting the trace into independent chunks and fitting
        each one separately means a one-off artifact can only poison
        the (hopefully minority of) chunks it actually overlaps; the
        median across chunks then has a real breakdown point, rather
        than relying on one global fit's shape being trustworthy in
        the first place. Measured on synthetic one-off sustained
        artifacts: brings tonic_snr inflation from ~6x (a flat true
        baseline, 5000x-noise_sigma artifact) down to ~1.04x, versus
        ~15-20% reductions from pre-despiking or percentile-trimmed
        ``tonic_range`` alone -- this changes the failure mode instead
        of just softening it.

        The trade-off: a chunk shorter than the recording's own genuine
        tonic drift period will only see a fraction of that drift,
        biasing ``snr_tonic`` down even on a clean trace (measured:
        ~3x underestimate using 15s chunks against a 120s-period
        drift). Default chunk sizing (``chunk_duration_s=None``)
        balances this with ``max(min_chunk_duration_s, chunk_fraction *
        recording_duration)`` -- a fixed floor for short recordings,
        scaling up for longer ones so chunk size tracks a long
        recording's own timescale rather than staying fixed at the
        floor. Pass ``chunk_duration_s`` explicitly to match your own
        recordings' known drift timescale instead of relying on this
        heuristic -- it's a reasonable default, not a substitute for
        knowing your own data.

        Everything else (phasic peak detection, noise floor, residual,
        the returned ``tonic`` curve) is unchanged from :meth:`fit` --
        only ``snr_tonic`` (and the ``snr``/``snr_corrected`` totals
        derived from it) differ. New fields not populated by a plain
        :meth:`fit` call: ``n_chunks``, ``chunk_duration_s``,
        ``per_chunk_tonic_snr``.

        Parameters
        ----------
        trace : ndarray
        chunk_duration_s : float, optional
            Chunk length in seconds. If None (default), uses
            ``max(min_chunk_duration_s, chunk_fraction * len(trace)/fps)``.
        min_chunk_duration_s : float
            Floor on the default chunk duration (seconds). Ignored if
            ``chunk_duration_s`` is given explicitly.
        chunk_fraction : float
            Fraction of the recording's total duration used for the
            default chunk duration, before the floor is applied.
            Ignored if ``chunk_duration_s`` is given explicitly.
        aggregate : {'median', 'mean'}
            How to combine per-chunk ``snr_tonic`` values. ``'median'``
            (default) is what gives this its outlier robustness;
            ``'mean'`` has no breakdown point and defeats the purpose --
            provided mainly for comparison/diagnostics.

        Returns
        -------
        EnvelopeRRResult
            Same structure as :meth:`fit`'s return value, with
            ``snr_tonic``/``snr``/``snr_corrected`` computed from the
            chunked-median tonic estimate, plus ``n_chunks``,
            ``chunk_duration_s``, ``per_chunk_tonic_snr``. Also stored
            on ``self.result_``.
        """
        trace = np.nan_to_num(trace, nan=float(np.nanmedian(trace)))

        # Global fit first: phasic peaks/signal/noise/residual/the
        # returned tonic curve all come from here, unchanged from a
        # plain fit() -- only snr_tonic (and totals derived from it)
        # get overridden below.
        result = self.fit(trace)

        n = len(trace)
        if chunk_duration_s is None:
            chunk_duration_s = max(min_chunk_duration_s, chunk_fraction * (n / self.fps))
        chunk_len = max(1, int(round(chunk_duration_s * self.fps)))
        n_chunks_requested = max(1, n // chunk_len)
        chunks = np.array_split(trace, n_chunks_requested)

        min_samples = max(50, int(self.config.get("midpoint_window", 101)))
        per_chunk_tonic_snr = []
        for c in chunks:
            if len(c) < min_samples:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = _decompose_envelope_rr(c, self.config)
            per_chunk_tonic_snr.append(float(r["tonic_snr_sd"]))

        if aggregate == "median":
            agg_fn = np.nanmedian
        elif aggregate == "mean":
            agg_fn = np.nanmean
        else:
            raise ValueError(f"aggregate must be 'median' or 'mean'; got {aggregate!r}.")

        if per_chunk_tonic_snr:
            snr_tonic_chunked = float(agg_fn(per_chunk_tonic_snr))
        else:
            warnings.warn(
                "No chunk was long enough to fit (trace too short for "
                "chunk_duration_s/min_chunk_duration_s) -- falling back "
                "to the global (non-chunked) snr_tonic.",
                RuntimeWarning,
                stacklevel=2,
            )
            snr_tonic_chunked = result.snr_tonic

        snr_total_chunked = snr_tonic_chunked + result.snr_phasic  # NaN-propagates

        snr_corrected_chunked = None
        if self.bias_correction is not None and result.snr_phasic_corrected is not None:
            snr_corrected_chunked = snr_tonic_chunked + result.snr_phasic_corrected

        chunked_result = replace(
            result,
            snr=snr_total_chunked,
            snr_tonic=snr_tonic_chunked,
            snr_corrected=snr_corrected_chunked,
            n_chunks=len(per_chunk_tonic_snr),
            chunk_duration_s=float(chunk_duration_s),
            per_chunk_tonic_snr=per_chunk_tonic_snr,
        )
        self.result_ = chunked_result
        return chunked_result

    def estimate(
        self, trace: NDArray[np.floating], apply_correction: bool = False
    ) -> Tuple[float, float, NDArray[np.intp]]:
        """One-shot functional interface: ``fit`` and return a 3-tuple.

        Returns ``(snr, noise, peaks)``, the same shape as a plain
        derivative-based ``estimate_snr(trace, fps)`` function, for
        drop-in comparison. ``snr`` is the **total**
        (``snr_tonic + snr_phasic``). Use :meth:`estimate_components`
        for the tonic/phasic breakdown without giving up this shape.

        Returns the raw (uncorrected) total by default even if
        ``bias_correction`` is configured; pass ``apply_correction=True``
        to get ``snr_corrected`` instead. Use ``.fit(trace)`` directly
        for repeated calls to avoid recomputing.
        """
        result = self.fit(trace)
        if apply_correction and result.snr_corrected is not None:
            snr = result.snr_corrected
        else:
            snr = result.snr
        return snr, result.noise, result.peaks

    def estimate_components(
        self, trace: NDArray[np.floating], apply_correction: bool = False
    ) -> Tuple[float, float, float]:
        """One-shot SNR breakdown: ``fit`` and return
        ``(snr_total, snr_tonic, snr_phasic)``.

        Companion to :meth:`estimate`, which only returns the total (to
        keep that method's 3-tuple shape drop-in compatible with a
        plain ``estimate_snr(trace, fps)`` function). ``snr_total``
        here always equals :meth:`estimate`'s first element under the
        same ``apply_correction`` setting.

        Returns raw (uncorrected) values by default; pass
        ``apply_correction=True`` to get ``snr_corrected`` and
        ``snr_phasic_corrected`` for the 1st/3rd elements (``snr_tonic``
        has no correction to apply, so it's unaffected either way).
        """
        result = self.fit(trace)
        if apply_correction and result.snr_corrected is not None:
            return result.snr_corrected, result.snr_tonic, result.snr_phasic_corrected
        return result.snr, result.snr_tonic, result.snr_phasic

    def decompose(
        self, trace: Optional[NDArray[np.floating]] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the (tonic, phasic) decomposition of a trace.

        Convenience wrapper around :meth:`fit`. "Phasic" is the tonic-
        subtracted residual (``trace - tonic``, same as
        ``EnvelopeRRResult.residual``). If ``trace`` is omitted, returns
        the decomposition from the most recent :meth:`fit`/
        :meth:`estimate` call (raises if not yet fit).

        Example
        -------
        >>> import numpy as np
        >>> y = 0.01 * np.random.default_rng(0).standard_normal(1200)
        >>> tonic, phasic = EnvelopeRRSNR(fps=20.0).decompose(y)
        >>> np.allclose(phasic, y - tonic)
        True
        """
        if trace is not None:
            result = self.fit(trace)
        else:
            self._check_fitted()
            result = self.result_
        return result.tonic, result.residual

    # ------------------------------------------------------------------
    # Convenience read-only views onto the last fit
    # ------------------------------------------------------------------

    @property
    def snr_(self) -> float:
        """Total SNR (``snr_tonic_ + snr_phasic_``) from the most recent
        :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr

    @property
    def snr_corrected_(self) -> Optional[float]:
        """Bias-corrected total SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_corrected

    @property
    def snr_tonic_(self) -> float:
        """Tonic-baseline SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_tonic

    @property
    def snr_phasic_(self) -> float:
        """Phasic SNR from the most recent :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr_phasic

    @property
    def noise_(self) -> float:
        """Noise floor from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.noise

    @property
    def peaks_(self) -> NDArray[np.intp]:
        """Detected phasic peak indices from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.peaks

    @property
    def tonic_(self) -> NDArray[np.floating]:
        """Fitted tonic envelope from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.tonic

    @property
    def residual_(self) -> NDArray[np.floating]:
        """Tonic-subtracted residual from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.residual

    def _check_fitted(self) -> None:
        """Raise RuntimeError if fit()/estimate()/decompose() hasn't been called yet."""
        if self.result_ is None:
            raise RuntimeError(
                "This EnvelopeRRSNR instance has not been fit yet. "
                "Call `.fit(trace)`, `.estimate(trace)`, or `.decompose(trace)` first."
            )

    # ------------------------------------------------------------------
    # Bias-correction helper
    # ------------------------------------------------------------------

    @staticmethod
    def fit_bias_correction_from_benchmark(
        true_snr: NDArray[np.floating], snr_est: NDArray[np.floating]
    ) -> Tuple[float, float]:
        """Fit a linear bias correction from benchmark ground truth.

        Convenience wrapper around ``numpy.polyfit`` for correcting a
        consistent linear bias found by sweeping known SNR levels. Fit
        this against ``snr_phasic`` (not the total ``snr``), separately
        per configuration (``noise_method``, ``peak_threshold_sd``,
        ``tonic_method``) -- each sits at its own scale. Returns
        ``(slope, intercept)``; pass directly as
        ``bias_correction=(slope, intercept)`` to the constructor.

        Example
        -------
        >>> import numpy as np
        >>> true_snr = np.array([5.0, 10.0, 20.0, 40.0])
        >>> snr_est  = 0.9 * true_snr + 1.5   # simulated linear bias
        >>> slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(
        ...     true_snr, snr_est)
        """
        true_snr = np.asarray(true_snr, dtype=float)
        snr_est = np.asarray(snr_est, dtype=float)
        ok = np.isfinite(true_snr) & np.isfinite(snr_est)
        slope, intercept = np.polyfit(true_snr[ok], snr_est[ok], 1)
        return float(slope), float(intercept)
