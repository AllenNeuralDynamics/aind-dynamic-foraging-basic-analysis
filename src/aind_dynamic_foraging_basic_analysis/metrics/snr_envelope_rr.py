"""
Envelope + rise-rate (Env+RR) signal-to-noise estimator for 1D
fluorescence (dF/F) traces.

This module provides :class:`EnvelopeRRSNR`, an alternative to a plain
derivative-based estimator (see e.g. ``estimate_snr`` in
``snr_kurtosis.py``). Rather than estimating noise from the sample-to-
sample derivative and signal from raw peak heights, it:

1. Tracks a valley-anchored **tonic envelope** beneath the trace, which
   is robust to slow drift/bleaching that would otherwise bias a
   derivative-based noise estimate.
2. Subtracts the tonic envelope to obtain a **phasic residual**.
3. Estimates the noise floor robustly from the below-median half of the
   residual (IQR-based sigma), which resists inflation by the phasic
   events themselves.
4. Detects candidate phasic peaks above ``peak_threshold_sd`` and
   refines them with a **rise-rate gate** that rejects slow drifts and
   keeps only fast-onset transients.
5. Reports SNR as a suprathreshold peak-amplitude statistic (median or
   95th percentile) divided by the estimated noise floor.

This is the "Env+RR" method benchmarked against a derivative-based
estimator across a synthetic SNR sweep in
``BWNM_Signal_Quality_Benchmark.ipynb``. That benchmark found Env+RR
detects events at least as well as the derivative method, but with a
small, consistent linear bias in its SNR estimate at high true SNR;
:class:`EnvelopeRRSNR` can optionally apply the fitted linear correction
for that bias (see ``bias_correction`` below and
``fit_bias_correction_from_benchmark``).

This module is **self-contained**: it has no dependency on
``bwnm_signal_utils`` or any other file in the benchmark repo (only
``numpy`` and ``scipy``), so it can be copied into another project on
its own. It inlines the subset of the envelope-decomposition machinery
that Env+RR actually exercises (valley-tracked or ALS tonic baseline +
rise-rate gated detection); the benchmark repo's
``bwnm_signal_utils.py`` additionally supports other detection methods
(``two_gate``, ``als_ceiling``, ``matched_filter``) and distribution-
fitting utilities not needed here, and is unaffected by this module.

Class API
---------
:class:`EnvelopeRRSNR`
    Stateful estimator: construct once with configuration, call
    :meth:`~EnvelopeRRSNR.fit` per trace to populate result attributes
    (``snr_``, ``noise_``, ``peaks_``, ``tonic_``, ``residual_``, ...),
    call :meth:`~EnvelopeRRSNR.estimate` for a one-shot functional
    interface returning ``(snr, noise, peaks)`` -- the same 3-tuple
    shape as a plain ``estimate_snr(trace, fps)`` function, for drop-in
    comparison -- or call :meth:`~EnvelopeRRSNR.decompose` for just the
    ``(tonic, phasic)`` component arrays.

Notes
-----
- Feed a dF/F preprocessed trace, as the peak height is interpreted
  from zero.
- Default sampling frequency (``fps``) is 20 Hz; adjust it if your data
  differ.
- NaNs are filled with the median of the trace prior to computation.

Example
-------
>>> import numpy as np
>>> from snr_envelope_rr import EnvelopeRRSNR
>>> rng = np.random.default_rng(0)
>>> t = np.arange(1200) / 20.0                      # 60 s @ 20 Hz
>>> y = 0.05 * np.sin(2 * np.pi * t / 40.0) + 0.01 * rng.standard_normal(1200)
>>> estimator = EnvelopeRRSNR(fps=20.0)
>>> result = estimator.fit(y)
>>> isinstance(result.snr, float) and isinstance(result.noise, float)
True
>>> isinstance(result.peaks, np.ndarray)
True
>>> snr, noise, peaks = estimator.estimate(y)        # one-shot form
>>> snr == result.snr
True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, signal, stats

__all__ = ["EnvelopeRRSNR", "EnvelopeRRResult"]

# Tuned defaults at the reference frame rate below. peak_threshold_sd is a
# sigma multiplier (unitless, does not scale with fps). The other three are
# window sizes in *samples*; scale_window() rescales them to preserve their
# real-world duration at a different fps.
_REFERENCE_FPS = 20.0
_TUNED_DEFAULTS = {
    "peak_threshold_sd": 1.5,
    "lower_min_distance": 20,  # 1.00 s at 20 fps
    "lower_smooth_window": 11,  # 0.55 s at 20 fps
    "rise_window": 5,  # 0.25 s at 20 fps
}

# Remaining envelope-decomposition parameters that aren't fps-scaled tuned
# hyperparameters above, but still have sensible fixed defaults. Any of
# these can be overridden via the `config` argument to EnvelopeRRSNR.
_DEFAULT_CONFIG: Dict = {
    "interp_kind": "pchip",  # 'pchip' or 'linear' envelope interpolation
    "tonic_method": "envelope",  # 'envelope' (valley-tracked) or 'als'
    "als_lam": 1e7,  # ALS smoothness (only if tonic_method='als')
    "als_p": 0.01,  # ALS asymmetry, rides under peaks
    "als_n_iter": 10,
    "midpoint_window": 101,  # smoothing window for the midpoint reference curve
    "midpoint_polyorder": 1,
    "lower_order": 2,  # local-minimum order for valley detection
    "upper_min_distance": 3,  # min spacing enforced by find_peaks on the residual
}


# ======================================================================
# Private helpers: envelope decomposition + rise-rate peak detection.
# Inlined and trimmed from bwnm_signal_utils.py to keep this module
# dependency-free; only the code paths EnvelopeRRSNR actually exercises
# (tonic_method in {'envelope', 'als'}, rise-rate detection) are kept.
# ======================================================================


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
    when ``p`` is small (the alternative tonic baseline to the
    valley-tracked envelope above; ``tonic_method='als'``)."""
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


def _detect_peaks_rise_rate(residual, candidates, sigma, rise_window: int = 3):
    """Reject candidate peaks whose approach isn't fast enough to be a
    real transient onset (vs. a slow drift crossing threshold)."""
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
        _, sigma_slope = stats.norm.fit(lower)
        slope_thresh = max(sigma_slope * 3.0, sigma * 0.1)
    else:
        slope_thresh = np.percentile(slopes, 50)

    return candidates[slopes > slope_thresh]


def _decompose_envelope_rr(x: NDArray[np.floating], config: Dict) -> Dict:
    """Tonic/phasic decomposition + rise-rate gated peak detection.

    Trimmed, self-contained equivalent of
    ``bwnm_signal_utils.estimate_snr_components_envelope`` with
    ``detection_method`` fixed to ``'rise_rate'`` (the only mode
    :class:`EnvelopeRRSNR` uses). Returns the same key set that
    ``EnvelopeRRSNR`` and ``EnvelopeRRResult`` read.
    """
    cfg = config

    midpoint = _moving_average(
        x,
        window=cfg["midpoint_window"],
        polyorder=cfg["midpoint_polyorder"],
    )

    tonic_method = cfg.get("tonic_method", "envelope")
    if tonic_method == "als":
        tonic = _als_baseline(
            x,
            lam=cfg.get("als_lam", 1e7),
            p=cfg.get("als_p", 0.01),
            n_iter=cfg.get("als_n_iter", 10),
        )
        tonic_minima = np.array([], dtype=int)
    elif tonic_method == "envelope":
        tonic, tonic_minima = _lower_envelope(
            x,
            midpoint,
            cfg["lower_smooth_window"],
            cfg["lower_order"],
            cfg["lower_min_distance"],
            interp_kind=cfg["interp_kind"],
        )
    else:
        raise ValueError(f"tonic_method must be 'envelope' or 'als'; got {tonic_method!r}.")

    residual = x - tonic

    neg_res = residual[residual < 0]
    if len(neg_res) >= 20:
        reflected = np.abs(neg_res)
        q75r, q25r = float(np.percentile(reflected, 75)), float(np.percentile(reflected, 25))
        sigma_iqr = (q75r - q25r) / 1.349
    else:
        sigma_iqr = float(np.std(residual[residual <= np.median(residual)]))

    if len(tonic_minima) >= 5:
        n = len(residual)
        mid_idx = ((tonic_minima[:-1] + tonic_minima[1:]) // 2).astype(int)
        mid_idx = mid_idx[(mid_idx >= 0) & (mid_idx < n)]
        sigma_minima = float(np.std(residual[mid_idx])) if len(mid_idx) >= 5 else sigma_iqr
    else:
        sigma_minima = sigma_iqr

    sigma_fit = sigma_iqr
    thresh = cfg["peak_threshold_sd"] * sigma_fit

    raw_peaks, _ = signal.find_peaks(
        residual,
        height=thresh,
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

    if len(peak_amps) > 0:
        phasic_p95 = float(np.percentile(peak_amps, 95))
        phasic_median = float(np.median(peak_amps))
        phasic_sd = float(np.std(peak_amps))
    else:
        phasic_p95 = phasic_median = phasic_sd = 0.0

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
        "tonic_range": float(np.ptp(tonic)),
        "phasic_p95": phasic_p95,
        "phasic_median": phasic_median,
        "phasic_amplitude": phasic_sd,
        "phasic_snr_p95": float(phasic_p95 / sigma_fit) if sigma_fit > 0 else float("nan"),
        "phasic_snr_median": float(phasic_median / sigma_fit) if sigma_fit > 0 else float("nan"),
        "phasic_snr_sd": float(phasic_sd / sigma_fit) if sigma_fit > 0 else float("nan"),
        "tonic_snr_sd": float(np.ptp(tonic) / sigma_fit) if sigma_fit > 0 else float("nan"),
        "config": cfg,
    }


@dataclass
class EnvelopeRRResult:
    """Fitted result of :meth:`EnvelopeRRSNR.fit`.

    Attributes
    ----------
    snr : float
        Estimated signal-to-noise ratio (dimensionless), using
        ``signal_statistic`` on the suprathreshold phasic peak
        amplitudes, before any bias correction.
    snr_corrected : float or None
        ``snr`` after applying ``bias_correction`` (``None`` if no
        correction was configured).
    noise : float
        Estimated noise floor (IQR-based sigma of the tonic-subtracted
        residual).
    peaks : numpy.ndarray
        Indices of detected phasic event peaks (rise-rate gated).
    tonic : numpy.ndarray
        The fitted tonic (slow baseline) envelope, same length as the
        input trace.
    residual : numpy.ndarray
        Tonic-subtracted residual (``trace - tonic``).
    signal : float
        Suprathreshold peak-amplitude statistic used as the numerator
        of ``snr`` (median or 95th percentile, per
        ``signal_statistic``).
    config : dict
        The resolved configuration used for this fit (defaults merged
        with any overrides).
    """

    snr: float
    noise: float
    peaks: NDArray[np.intp]
    tonic: NDArray[np.floating]
    residual: NDArray[np.floating]
    signal: float
    config: Dict = field(repr=False)
    snr_corrected: Optional[float] = None


class EnvelopeRRSNR:
    """Envelope + rise-rate SNR estimator for 1D fluorescence traces.

    Parameters
    ----------
    fps : float, optional
        Sampling frequency (frames per second), by default ``20.0``.
    peak_threshold_sd : float, optional
        Detection threshold for candidate phasic peaks, in units of the
        estimated noise sigma. Defaults to ``1.5``, a value tuned by
        grid search in ``BWNM_Signal_Quality_Benchmark.ipynb``. This is
        a unitless sigma multiplier, so it does *not* scale with
        ``fps``.
    signal_statistic : {'median', 'p95'}, optional
        Which statistic of the suprathreshold peak amplitudes to use as
        the SNR numerator, by default ``'median'``.
    bias_correction : tuple of (slope, intercept), optional
        If given, ``snr_corrected = (snr_raw - intercept) / slope`` is
        computed on every fit. Fit this once against ground-truth SNR
        (e.g. from a synthetic benchmark sweep) with
        :meth:`fit_bias_correction_from_benchmark` or
        ``numpy.polyfit(true_snr, snr_est, 1)``, then reuse it here to
        correct a consistent linear bias without re-running the
        benchmark on new data.
    config : dict, optional
        Overrides merged on top of the tuned window defaults (fps-scaled
        ``lower_min_distance``, ``lower_smooth_window``, ``rise_window``)
        and the fixed defaults in this module (envelope interpolation,
        ALS baseline, midpoint smoothing, etc.) -- see
        ``_DEFAULT_CONFIG`` in this file for the full list. Notably,
        pass ``config={'tonic_method': 'als'}`` to swap the valley-
        tracked tonic baseline for an Asymmetric Least Squares fit.

    Notes
    -----
    - Noise is estimated from the IQR of the tonic-subtracted residual,
      not the raw derivative, so it is robust to both slow drift and
      the phasic events themselves.
    - Signal is estimated from suprathreshold, rise-rate-gated peak
      amplitudes in the residual.
    - If fewer than one phasic peak is found, ``snr`` is ``NaN`` and a
      :class:`RuntimeWarning` is issued (mirroring the behaviour of a
      plain derivative-based estimator when too few peaks are found).
    - **Window defaults scale with fps.** ``lower_min_distance``,
      ``lower_smooth_window``, and ``rise_window`` are sample counts,
      tuned at 20 fps (1.00 s, 0.55 s, and 0.25 s respectively). At any
      other ``fps`` they are automatically rescaled by
      :meth:`scale_window` to preserve those durations, then rounded to
      the nearest integer (``lower_smooth_window`` is additionally
      forced odd, as required by the underlying Savitzky-Golay smoothing
      step). Pass an explicit value in ``config`` to opt out of scaling
      for a given parameter.
    """

    def __init__(
        self,
        fps: float = 20.0,
        peak_threshold_sd: float = _TUNED_DEFAULTS["peak_threshold_sd"],
        signal_statistic: str = "median",
        bias_correction: Optional[Tuple[float, float]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """Construct an estimator; see the class docstring for parameter details."""
        if signal_statistic not in ("median", "p95"):
            raise ValueError(
                f"signal_statistic must be 'median' or 'p95', got {signal_statistic!r}."
            )
        self.fps = fps
        self.peak_threshold_sd = peak_threshold_sd
        self.signal_statistic = signal_statistic
        self.bias_correction = bias_correction

        scaled_defaults = {
            "lower_min_distance": self.scale_window(_TUNED_DEFAULTS["lower_min_distance"], fps),
            "lower_smooth_window": self.scale_window(
                _TUNED_DEFAULTS["lower_smooth_window"], fps, make_odd=True
            ),
            "rise_window": self.scale_window(_TUNED_DEFAULTS["rise_window"], fps),
        }
        self.config = {
            **_DEFAULT_CONFIG,
            "fps": fps,
            "peak_threshold_sd": peak_threshold_sd,
            **scaled_defaults,
            **(config or {}),
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

        Parameters
        ----------
        base_samples : int
            Window size in samples at ``reference_fps``.
        fps : float
            Target sampling frequency.
        reference_fps : float, optional
            The fps ``base_samples`` was tuned/specified at, by default
            ``20.0``.
        make_odd : bool, optional
            If True, increment the result by 1 if it comes out even
            (required by e.g. Savitzky-Golay smoothing windows).

        Returns
        -------
        int
            ``round(base_samples * fps / reference_fps)``, floored at 1.

        Example
        -------
        >>> EnvelopeRRSNR.scale_window(20, fps=20.0)
        20
        >>> EnvelopeRRSNR.scale_window(20, fps=40.0)
        40
        >>> EnvelopeRRSNR.scale_window(11, fps=40.0, make_odd=True)
        23
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

        Parameters
        ----------
        trace : numpy.ndarray
            1D input trace (e.g. dF/F). NaNs are replaced with the
            median of ``trace`` before calculation.

        Returns
        -------
        EnvelopeRRResult
            Also stored on ``self.result_`` for later access via the
            convenience properties (``self.snr_``, ``self.noise_``,
            ``self.peaks_``, etc.).
        """
        trace = np.nan_to_num(trace, nan=float(np.nanmedian(trace)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = _decompose_envelope_rr(trace, self.config)

        snr_key = "phasic_snr_p95" if self.signal_statistic == "p95" else "phasic_snr_median"
        sig_key = "phasic_p95" if self.signal_statistic == "p95" else "phasic_median"

        peaks = np.asarray(raw["event_maxima"], dtype=int)
        snr = float(raw[snr_key])
        noise = float(raw["noise_sigma"])
        signal = float(raw[sig_key])

        if len(peaks) == 0:
            warnings.warn(
                "No phasic peaks detected above threshold. Returning NaN for snr.",
                RuntimeWarning,
                stacklevel=2,
            )
            snr = float("nan")

        snr_corrected = None
        if self.bias_correction is not None and not np.isnan(snr):
            slope, intercept = self.bias_correction
            snr_corrected = float((snr - intercept) / slope)

        self.result_ = EnvelopeRRResult(
            snr=snr,
            snr_corrected=snr_corrected,
            noise=noise,
            peaks=peaks,
            tonic=raw["tonic"],
            residual=raw["residual"],
            signal=signal,
            config=raw["config"],
        )
        return self.result_

    def estimate(self, trace: NDArray[np.floating]) -> Tuple[float, float, NDArray[np.intp]]:
        """One-shot functional interface: ``fit`` and return a 3-tuple.

        Mirrors the ``(snr, noise, peaks)`` return signature of a plain
        derivative-based ``estimate_snr(trace, fps)`` function, so this
        class can be dropped in wherever that function is used. Returns
        the bias-corrected SNR if ``bias_correction`` was configured,
        otherwise the raw SNR.

        Parameters
        ----------
        trace : numpy.ndarray
            1D input trace (e.g. dF/F).

        Returns
        -------
        snr : float
        noise : float
        peaks : numpy.ndarray
        """
        result = self.fit(trace)
        snr = result.snr_corrected if result.snr_corrected is not None else result.snr
        return snr, result.noise, result.peaks

    def decompose(
        self, trace: Optional[NDArray[np.floating]] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the (tonic, phasic) decomposition of a trace.

        Convenience wrapper around :meth:`fit` for callers who only want
        the two component arrays rather than the full SNR estimate.
        "Phasic" here is the tonic-subtracted residual (``trace -
        tonic``) — the same array as ``EnvelopeRRResult.residual`` /
        ``self.residual_`` — named ``phasic`` to match the signal/noise
        decomposition terminology used elsewhere (e.g.
        ``phasic_median``, ``phasic_snr_median``).

        Parameters
        ----------
        trace : numpy.ndarray, optional
            1D input trace (e.g. dF/F). If omitted, returns the
            decomposition from the most recent :meth:`fit` /
            :meth:`estimate` call instead of recomputing one; raises if
            this instance hasn't been fit yet.

        Returns
        -------
        tonic : numpy.ndarray
            The fitted tonic (slow baseline) envelope.
        phasic : numpy.ndarray
            The tonic-subtracted residual, i.e. the phasic component.

        Example
        -------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(1200) / 20.0
        >>> y = 0.05 * np.sin(2 * np.pi * t / 40.0) + 0.01 * rng.standard_normal(1200)
        >>> tonic, phasic = EnvelopeRRSNR(fps=20.0).decompose(y)
        >>> tonic.shape == y.shape and phasic.shape == y.shape
        True
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
        """SNR from the most recent :meth:`fit` call (raw, uncorrected)."""
        self._check_fitted()
        return self.result_.snr

    @property
    def snr_corrected_(self) -> Optional[float]:
        """Bias-corrected SNR from the most recent :meth:`fit` call."""
        self._check_fitted()
        return self.result_.snr_corrected

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

        Convenience wrapper around ``numpy.polyfit`` for the common case
        of correcting a consistent linear bias identified by sweeping
        known SNR levels (e.g. via
        ``BWNM_Signal_Quality_Benchmark.ipynb``, Figure 2).

        Parameters
        ----------
        true_snr : numpy.ndarray
            Ground-truth SNR values used in the benchmark sweep.
        snr_est : numpy.ndarray
            Corresponding ``EnvelopeRRSNR``-estimated SNR values.

        Returns
        -------
        (slope, intercept) : tuple of float
            Pass directly as ``bias_correction=(slope, intercept)`` to
            the constructor.

        Example
        -------
        >>> import numpy as np
        >>> true_snr = np.array([5.0, 10.0, 20.0, 40.0])
        >>> snr_est  = 0.9 * true_snr + 1.5   # simulated linear bias
        >>> slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(
        ...     true_snr, snr_est)
        >>> estimator = EnvelopeRRSNR(bias_correction=(slope, intercept))
        """
        true_snr = np.asarray(true_snr, dtype=float)
        snr_est = np.asarray(snr_est, dtype=float)
        ok = np.isfinite(true_snr) & np.isfinite(snr_est)
        slope, intercept = np.polyfit(true_snr[ok], snr_est[ok], 1)
        return float(slope), float(intercept)


# ======================================================================
# Self-test
#
# Runs a synthetic (numpy-only, no external dependencies) sanity check
# of the public API: `python snr_envelope_rr.py`. This is a quick
# correctness smoke test, not a substitute for the full benchmark in
# BWNM_Signal_Quality_Benchmark.ipynb -- it checks that the class
# behaves as documented, not that its SNR estimates are accurate.
# ======================================================================


def _make_synthetic_trace(
    n_samples: int = 3000,
    fps: float = 20.0,
    n_events: int = 40,
    event_amp: float = 0.15,
    noise_sigma: float = 0.01,
    drift_amp: float = 0.05,
    seed: int = 0,
) -> NDArray[np.floating]:
    """Build a synthetic dF/F-like trace: slow sinusoidal drift (tonic)
    + sparse exponential-decay transients (phasic) + Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fps

    tonic = drift_amp * np.sin(2 * np.pi * t / (n_samples / fps / 2))

    trace = tonic.copy()
    event_starts = rng.choice(
        np.arange(int(0.5 * fps), n_samples - int(2 * fps)), size=n_events, replace=False
    )
    decay_kernel = event_amp * np.exp(-np.arange(int(1.5 * fps)) / (0.3 * fps))
    for start in event_starts:
        end = min(start + len(decay_kernel), n_samples)
        trace[start:end] += decay_kernel[: end - start]

    trace += noise_sigma * rng.standard_normal(n_samples)
    return trace


def _run_case(name: str, fn) -> Tuple[str, bool, str]:
    """Run one self-test case, capturing pass/fail (and any exception) by name."""
    try:
        fn()
        return (name, True, "")
    except Exception as exc:  # noqa: BLE001 - want to catch/report everything here
        return (name, False, f"{type(exc).__name__}: {exc}")


def _case_fit_basic(trace: NDArray[np.floating]) -> None:
    """fit() should return a well-formed result with at least one detected peak."""
    result = EnvelopeRRSNR().fit(trace)
    assert isinstance(result.snr, float)
    assert isinstance(result.noise, float) and result.noise > 0
    assert isinstance(result.peaks, np.ndarray)
    assert len(result.peaks) > 0, "expected to detect at least one event"
    assert result.tonic.shape == trace.shape
    assert result.residual.shape == trace.shape


def _case_estimate_matches_fit(trace: NDArray[np.floating]) -> None:
    """estimate()'s one-shot tuple should match a separate fit() call exactly."""
    est = EnvelopeRRSNR()
    result = est.fit(trace)
    snr, noise, peaks = est.estimate(trace)
    assert snr == result.snr
    assert noise == result.noise
    assert np.array_equal(peaks, result.peaks)


def _case_decompose_with_and_without_trace(trace: NDArray[np.floating]) -> None:
    """decompose() with vs. without a trace argument should agree, and tonic+phasic
    should reconstruct the original trace."""
    est = EnvelopeRRSNR()
    tonic1, phasic1 = est.decompose(trace)
    tonic2, phasic2 = est.decompose()  # reuse last fit, no recompute
    assert np.array_equal(tonic1, tonic2)
    assert np.array_equal(phasic1, phasic2)
    assert np.allclose(trace, tonic1 + phasic1), "trace should equal tonic + phasic"


def _case_not_fitted_raises() -> None:
    """Accessing a result property before any fit() call should raise RuntimeError."""
    est = EnvelopeRRSNR()
    try:
        _ = est.snr_
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError before any fit")


def _case_invalid_signal_statistic_raises() -> None:
    """An unrecognized signal_statistic should raise ValueError at construction."""
    try:
        EnvelopeRRSNR(signal_statistic="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid signal_statistic")


def _case_scale_window_reference_fps_is_identity() -> None:
    """scale_window() at the 20 fps reference rate should return its input unchanged."""
    for base in (5, 11, 20):
        assert EnvelopeRRSNR.scale_window(base, fps=20.0) == base


def _case_scale_window_doubles_at_2x_fps() -> None:
    """scale_window() should double sample counts when fps doubles."""
    assert EnvelopeRRSNR.scale_window(20, fps=40.0) == 40
    assert EnvelopeRRSNR.scale_window(5, fps=40.0) == 10


def _case_scale_window_make_odd() -> None:
    """scale_window(make_odd=True) should always return an odd result."""
    assert EnvelopeRRSNR.scale_window(11, fps=40.0, make_odd=True) % 2 == 1


def _case_fps_scaling_reaches_config() -> None:
    """Window params in the resolved config should scale with fps; peak_threshold_sd
    should not."""
    cfg20 = EnvelopeRRSNR(fps=20.0).config
    cfg40 = EnvelopeRRSNR(fps=40.0).config
    assert cfg20["lower_min_distance"] * 2 == cfg40["lower_min_distance"]
    assert cfg20["rise_window"] * 2 == cfg40["rise_window"]
    # peak_threshold_sd is not a window -- must NOT scale with fps
    assert cfg20["peak_threshold_sd"] == cfg40["peak_threshold_sd"]


def _case_config_override_takes_precedence() -> None:
    """An explicit config override should win over the fps-scaled tuned default."""
    est = EnvelopeRRSNR(config={"rise_window": 999})
    assert est.config["rise_window"] == 999


def _case_bias_correction_applied(trace: NDArray[np.floating]) -> None:
    """snr_corrected should equal (snr - intercept) / slope for the configured correction."""
    slope, intercept = 0.8, 2.0
    est = EnvelopeRRSNR(bias_correction=(slope, intercept))
    result = est.fit(trace)
    assert result.snr_corrected is not None
    expected = (result.snr - intercept) / slope
    assert abs(result.snr_corrected - expected) < 1e-9


def _case_fit_bias_correction_from_benchmark_recovers_known_fit() -> None:
    """fit_bias_correction_from_benchmark() should exactly recover a noiseless linear fit."""
    true_snr = np.array([5.0, 10.0, 20.0, 40.0])
    snr_est = 0.8 * true_snr + 2.0  # exact, noiseless linear relationship
    slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(true_snr, snr_est)
    assert abs(slope - 0.8) < 1e-9
    assert abs(intercept - 2.0) < 1e-9


def _case_als_tonic_method_runs(trace: NDArray[np.floating]) -> None:
    """The tonic_method='als' config override should run end-to-end without error."""
    result = EnvelopeRRSNR(config={"tonic_method": "als"}).fit(trace)
    assert isinstance(result.snr, float)
    assert result.tonic.shape == trace.shape


def _case_no_peaks_gives_nan_snr_with_warning() -> None:
    """Pure-noise input with an unreachable threshold should give NaN snr + a warning."""
    flat = 0.001 * np.random.default_rng(1).standard_normal(500)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = EnvelopeRRSNR(peak_threshold_sd=100.0).fit(flat)
    assert np.isnan(result.snr)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def _case_doctests_pass() -> None:
    """All doctests embedded in this module's docstrings should pass."""
    import doctest

    this_module = __import__(__name__)
    fail_count, _ = doctest.testmod(this_module, optionflags=doctest.ELLIPSIS)
    assert fail_count == 0, f"{fail_count} doctest(s) failed"


def _self_test() -> bool:
    """Run all self-test cases; print PASS/FAIL per case. Returns True iff every
    case passed."""
    trace = _make_synthetic_trace()

    cases = [
        ("fit() returns sane result", lambda: _case_fit_basic(trace)),
        ("estimate() matches fit()", lambda: _case_estimate_matches_fit(trace)),
        (
            "decompose() with/without trace agree",
            lambda: _case_decompose_with_and_without_trace(trace),
        ),
        ("accessing result before fit() raises", _case_not_fitted_raises),
        ("invalid signal_statistic raises", _case_invalid_signal_statistic_raises),
        (
            "scale_window() is identity at reference fps",
            _case_scale_window_reference_fps_is_identity,
        ),
        ("scale_window() doubles at 2x fps", _case_scale_window_doubles_at_2x_fps),
        ("scale_window() forces odd when requested", _case_scale_window_make_odd),
        ("fps scaling reaches resolved config", _case_fps_scaling_reaches_config),
        ("explicit config override takes precedence", _case_config_override_takes_precedence),
        ("bias_correction is applied correctly", lambda: _case_bias_correction_applied(trace)),
        (
            "fit_bias_correction_from_benchmark recovers fit",
            _case_fit_bias_correction_from_benchmark_recovers_known_fit,
        ),
        ("tonic_method='als' override runs", lambda: _case_als_tonic_method_runs(trace)),
        (
            "no detected peaks -> NaN snr + RuntimeWarning",
            _case_no_peaks_gives_nan_snr_with_warning,
        ),
        ("module doctests pass", _case_doctests_pass),
    ]
    results = [_run_case(name, fn) for name, fn in cases]

    name_width = max(len(name) for name, _, _ in results)
    n_passed = 0
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {name:<{name_width}}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        n_passed += int(passed)

    print(f"\n{n_passed}/{len(results)} checks passed.")
    return n_passed == len(results)


if __name__ == "__main__":
    import sys as _sys

    print(f"Running self-test for {__name__} ...\n")
    ok = _self_test()
    _sys.exit(0 if ok else 1)
