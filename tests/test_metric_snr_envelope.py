"""Test the envelope + rise-rate (Env+RR) SNR estimator.

To run the test, execute "python -m unittest tests/test_metric_snr_envelope.py".
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from aind_dynamic_foraging_basic_analysis.metrics.snr_envelope_rr import (
    _UNSET,
    EnvelopeRRSNR,
    _detect_peaks_rise_rate,
    _mad_noise_std,
    _robust_std,
    _robust_tonic_range,
    _UnsetType,
)

# Shared simulation parameters, reused across every test so intent stays visible at
# each call site (e.g. `event_amplitude=0.2` reads as "20x NOISE_SIGMA") without
# repeating magic numbers.
FPS = 20.0
N_SAMPLES = 3000  # 150 s at FPS
NOISE_SIGMA = 0.01


def make_tonic_only_trace(seed=0, tonic_amp=0.05, tonic_period_s=40.0):
    """Build a trace with a known sinusoidal tonic component and NO phasic events.

    Ground truth is exact: the tonic component is returned separately, so
    ``np.ptp(tonic_true) / NOISE_SIGMA`` is the true tonic SNR to compare estimates
    against, and the true phasic amplitude is exactly zero.

    Returns
    -------
    trace : np.ndarray
    tonic_true : np.ndarray
        The trace's only signal component (noise aside).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FPS
    tonic_true = tonic_amp * np.sin(2 * np.pi * t / tonic_period_s)
    noise = NOISE_SIGMA * rng.standard_normal(N_SAMPLES)
    return tonic_true + noise, tonic_true


def make_phasic_only_trace(seed=1, event_amplitude=0.2, n_events=20, tau_rise=2, tau_decay=8):
    """Build a flat-tonic trace with a known number of phasic transients of a known
    amplitude, evenly spaced (no overlap), plus Gaussian noise.

    Returns
    -------
    trace : np.ndarray
    event_indices : np.ndarray
        True sample index of each transient's onset.
    """
    rng = np.random.default_rng(seed)
    trace = NOISE_SIGMA * rng.standard_normal(N_SAMPLES)

    spacing = N_SAMPLES // (n_events + 1)
    event_indices = np.arange(1, n_events + 1) * spacing

    kernel_len = 60  # 3 s -- well under `spacing`, so events never overlap
    kernel_t = np.arange(kernel_len)
    kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
    kernel = kernel / kernel.max()

    for idx in event_indices:
        end = min(N_SAMPLES, idx + kernel_len)
        trace[idx:end] += event_amplitude * kernel[: end - idx]

    return trace, event_indices


def inject_artifact(trace, amplitude, onset_s, width_s=2.0):
    """Add one large, brief artifact to ``trace`` and return the modified trace plus
    the sample-index range it occupies.

    Uses a raised-cosine bump: exactly zero outside its window, and both its value
    and slope reach zero at each edge, so it has no long tail to distort a tonic fit
    with -- a genuinely transient artifact (seconds long, decays rapidly to
    baseline), not an unbounded one.

    Returns
    -------
    trace_with_artifact : np.ndarray
    onset_idx, end_idx : int
        Sample-index range the artifact occupies (for checking it gets detected).
    """
    n = len(trace)
    t = np.arange(n) / FPS
    onset_idx = int(round(onset_s * FPS))
    end_idx = onset_idx + int(round(width_s * FPS))
    rel_t = t - onset_s
    in_window = (rel_t >= 0) & (rel_t <= width_s)
    bump = np.where(in_window, amplitude * 0.5 * (1 - np.cos(2 * np.pi * rel_t / width_s)), 0.0)
    return trace + bump, onset_idx, end_idx


class TestEnvelopeRRSNRTonic(unittest.TestCase):
    """Test snr_tonic / noise estimation on traces with a known sinusoidal tonic and
    no phasic events."""

    def test_tonic_snr_matches_ground_truth(self):
        """snr_tonic should track ptp(tonic_true) / NOISE_SIGMA for a clean sinusoid."""
        trace, tonic_true = make_tonic_only_trace()
        true_tonic_snr = np.ptp(tonic_true) / NOISE_SIGMA

        result = EnvelopeRRSNR(fps=FPS).fit(trace)

        self.assertAlmostEqual(result.snr_tonic / true_tonic_snr, 1.0, delta=0.3)

    def test_noise_estimate_close_to_true_sigma(self):
        """noise should track the true injected NOISE_SIGMA."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit(trace)
        self.assertAlmostEqual(result.noise / NOISE_SIGMA, 1.0, delta=0.3)

    def test_no_phasic_events_returns_nan_and_warns(self):
        """A strict enough threshold on a no-phasic-signal trace should detect zero
        peaks, return NaN for snr_phasic (and the total snr, via propagation), warn,
        and leave snr_tonic unaffected."""
        trace, _ = make_tonic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS, peak_threshold_sd=50.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(trace)
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

        self.assertEqual(len(result.peaks), 0)
        self.assertTrue(np.isnan(result.snr_phasic))
        self.assertTrue(np.isnan(result.snr))
        self.assertFalse(np.isnan(result.snr_tonic))


class TestEnvelopeRRSNREmptyTrace(unittest.TestCase):
    """Test that empty and too-short traces return all-NaN results and warn,
    rather than raising -- a channel that ends up with zero (or too few)
    samples after upstream filtering (e.g. an overly aggressive time window
    on a short session) must not crash a pipeline processing many
    channels/sessions."""

    def _assert_all_nan_result(self, result, expected_len):
        """Shared assertions for an EnvelopeRRResult produced from an
        empty/too-short trace. ``tonic``/``residual`` are NaN-filled at the
        *input's* length (not forced to zero-length), so downstream code
        expecting ``len(tonic) == len(trace)`` doesn't also need to
        special-case the failure path."""
        self.assertTrue(np.isnan(result.snr))
        self.assertTrue(np.isnan(result.snr_tonic))
        self.assertTrue(np.isnan(result.snr_phasic))
        self.assertTrue(np.isnan(result.noise))
        self.assertTrue(np.isnan(result.signal))
        self.assertEqual(len(result.peaks), 0)
        self.assertEqual(len(result.tonic), expected_len)
        self.assertEqual(len(result.residual), expected_len)
        self.assertTrue(np.all(np.isnan(result.tonic)))
        self.assertTrue(np.all(np.isnan(result.residual)))
        self.assertEqual(result.n_tonic_swings, 0)
        # Corrections were never computed (no snr_phasic to correct), not
        # "computed and NaN" -- mirrors how the module already represents
        # "no correction configured" elsewhere.
        self.assertIsNone(result.snr_corrected)
        self.assertIsNone(result.snr_phasic_corrected)

    def test_fit_on_empty_trace_returns_nan_and_warns(self):
        """.fit() on an empty array should warn and return an all-NaN result,
        not raise (an empty trace crashes arPLS's sparse baseline fit
        otherwise, since its difference matrix needs at least 2 samples)."""
        empty_trace = np.array([])
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(empty_trace)
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

        self._assert_all_nan_result(result, expected_len=0)

    def test_estimate_on_empty_trace_returns_nan_and_warns(self):
        """.estimate() should propagate the same empty-trace handling as .fit()."""
        empty_trace = np.array([])
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            snr, noise, peaks = estimator.estimate(empty_trace)
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

        self.assertTrue(np.isnan(snr))
        self.assertTrue(np.isnan(noise))
        self.assertEqual(len(peaks), 0)

    def test_estimate_components_on_empty_trace_returns_nan(self):
        """.estimate_components() should also return an all-NaN breakdown."""
        empty_trace = np.array([])
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr_total, snr_tonic, snr_phasic = estimator.estimate_components(empty_trace)

        self.assertTrue(np.isnan(snr_total))
        self.assertTrue(np.isnan(snr_tonic))
        self.assertTrue(np.isnan(snr_phasic))

    def test_decompose_on_empty_trace_returns_empty_arrays(self):
        """.decompose() should return empty (tonic, residual) arrays, not raise."""
        empty_trace = np.array([])
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tonic, residual = estimator.decompose(empty_trace)

        self.assertEqual(len(tonic), 0)
        self.assertEqual(len(residual), 0)

    def test_empty_list_input_is_also_handled(self):
        """A plain empty list (not yet a numpy array) should be handled the same
        way as an empty ndarray -- callers may pass `.values` or similar
        array-likes rather than an explicit np.array."""
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit([])
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

        self._assert_all_nan_result(result, expected_len=0)

    def test_convenience_properties_reflect_empty_trace_result(self):
        """The `*_` properties should mirror the all-NaN result after fitting
        an empty trace, same as they do for a normal fit."""
        estimator = EnvelopeRRSNR(fps=FPS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(np.array([]))

        self.assertTrue(np.isnan(estimator.snr_))
        self.assertTrue(np.isnan(estimator.snr_tonic_))
        self.assertTrue(np.isnan(estimator.snr_phasic_))
        self.assertTrue(np.isnan(estimator.noise_))
        self.assertEqual(len(estimator.peaks_), 0)
        self.assertIsNone(estimator.snr_corrected_)
        np.testing.assert_array_equal(estimator.tonic_, result.tonic)
        np.testing.assert_array_equal(estimator.residual_, result.residual)

    # -- Near-empty / too-short (below min_samples), not just literally empty --

    def test_single_sample_trace_does_not_crash(self):
        """A 1-sample trace used to crash arPLS's sparse difference matrix
        (degenerate for L<2) before empty/near-empty handling existed;
        confirm it's now routed through the same NaN+warn path as empty."""
        estimator = EnvelopeRRSNR(fps=FPS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(np.array([0.5]))
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self._assert_all_nan_result(result, expected_len=1)

    def test_trace_just_below_min_samples_warns_and_returns_nan(self):
        """A trace one sample short of config['min_samples'] should hit the
        too-short guard, not attempt a fit."""
        estimator = EnvelopeRRSNR(fps=FPS)
        min_samples = estimator.config["min_samples"]
        trace = 0.01 * np.random.default_rng(0).standard_normal(min_samples - 1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(trace)
            messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
            self.assertTrue(any("min_samples" in m for m in messages))

        self._assert_all_nan_result(result, expected_len=min_samples - 1)

    def test_trace_at_exactly_min_samples_is_fit_normally(self):
        """A trace of exactly config['min_samples'] length should be fit
        normally (no too-short warning) -- the boundary is inclusive."""
        estimator = EnvelopeRRSNR(fps=FPS)
        min_samples = estimator.config["min_samples"]
        trace = 0.01 * np.random.default_rng(0).standard_normal(min_samples)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(trace)
            too_short_warnings = [
                w
                for w in caught
                if issubclass(w.category, RuntimeWarning) and "min_samples" in str(w.message)
            ]
            self.assertEqual(len(too_short_warnings), 0)

        self.assertEqual(len(result.tonic), min_samples)
        self.assertFalse(np.all(np.isnan(result.tonic)))

    def test_min_samples_scales_with_fps(self):
        """min_samples should scale with fps like the module's other windows,
        preserving real-world duration rather than a fixed sample count."""
        e_20 = EnvelopeRRSNR(fps=20.0)
        e_40 = EnvelopeRRSNR(fps=40.0)
        ratio = e_40.config["min_samples"] / e_20.config["min_samples"]
        self.assertAlmostEqual(ratio, 2.0, delta=0.05)

    def test_min_samples_cannot_be_configured_below_hard_floor_of_two(self):
        """arPLS's sparse difference matrix is degenerate below 2 samples
        regardless of configuration -- an explicit min_samples override
        below 2 must be clamped, not silently reopen the original crash."""
        estimator = EnvelopeRRSNR(fps=FPS, config={"min_samples": 0})
        self.assertEqual(estimator.config["min_samples"], 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Must not raise, even with min_samples forced to its floor and a
            # 1-sample trace (below even that floor).
            estimator.fit(np.array([0.1]))

    def test_custom_min_samples_is_respected_above_the_floor(self):
        """A user-supplied min_samples above the hard floor of 2 should be
        used as given, not overridden by the fps-scaled default."""
        estimator = EnvelopeRRSNR(fps=FPS, config={"min_samples": 10})
        self.assertEqual(estimator.config["min_samples"], 10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.fit(0.01 * np.random.default_rng(0).standard_normal(9))
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertTrue(np.isnan(result.snr))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(0.01 * np.random.default_rng(0).standard_normal(10))
        self.assertEqual(len(result.tonic), 10)


class TestEnvelopeRRSNRPhasic(unittest.TestCase):
    """Test snr_phasic / peak detection on traces with a flat tonic and known phasic
    events."""

    def test_detects_expected_number_of_events(self):
        """Detected event count should be close to the true count (rise-rate gating
        or boundary effects can occasionally add or drop one)."""
        trace, event_indices = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit(trace)
        self.assertAlmostEqual(len(result.peaks), len(event_indices), delta=2)

    def test_detected_peaks_land_near_true_event_indices(self):
        """Every true event should have a detected peak within 0.5 s of it."""
        trace, event_indices = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit(trace)
        self.assertGreater(len(result.peaks), 0)
        for true_idx in event_indices:
            nearest_distance = np.min(np.abs(result.peaks - true_idx))
            self.assertLess(nearest_distance, 0.5 * FPS)

    def test_snr_phasic_scales_linearly_with_true_amplitude(self):
        """Tripling the true event amplitude should roughly triple snr_phasic."""
        trace_small, _ = make_phasic_only_trace(event_amplitude=0.1)
        trace_large, _ = make_phasic_only_trace(event_amplitude=0.3)

        snr_small = EnvelopeRRSNR(fps=FPS).fit(trace_small).snr_phasic
        snr_large = EnvelopeRRSNR(fps=FPS).fit(trace_large).snr_phasic

        expected_ratio = 0.3 / 0.1
        actual_ratio = snr_large / snr_small
        self.assertAlmostEqual(actual_ratio / expected_ratio, 1.0, delta=0.3)


class TestEnvelopeRRSNRArtifact(unittest.TestCase):
    """Test robustness of noise/SNR estimates to one large-amplitude artifact, added
    on top of a known sinusoidal-tonic trace."""

    ARTIFACT_AMPLITUDE = 500 * NOISE_SIGMA  # "hundreds of SD", per real problem recordings
    ARTIFACT_ONSET_S = 40.0
    ARTIFACT_WIDTH_S = 2.0

    def _clean_and_artifact_results(self, seed):
        """Fit the same trace with and without one injected artifact; returns
        (result_clean, result_artifact, onset_idx, end_idx)."""
        trace_clean, _ = make_tonic_only_trace(seed=seed)
        trace_artifact, onset_idx, end_idx = inject_artifact(
            trace_clean.copy(),
            amplitude=self.ARTIFACT_AMPLITUDE,
            onset_s=self.ARTIFACT_ONSET_S,
            width_s=self.ARTIFACT_WIDTH_S,
        )
        result_clean = EnvelopeRRSNR(fps=FPS).fit(trace_clean)
        result_artifact = EnvelopeRRSNR(fps=FPS).fit(trace_artifact)
        return result_clean, result_artifact, onset_idx, end_idx

    def test_noise_floor_robust_to_large_artifact(self):
        """The noise floor should stay close to its own no-artifact value despite an
        artifact ~500x the noise floor being added."""
        result_clean, result_artifact, _, _ = self._clean_and_artifact_results(seed=2)
        ratio = result_artifact.noise / result_clean.noise
        self.assertAlmostEqual(ratio, 1.0, delta=0.5)

    def test_tonic_snr_robust_to_large_artifact(self):
        """snr_tonic should stay close to its own no-artifact value despite the
        same large artifact (tonic-range estimation is the median swing across
        detected extrema, robust to one contaminated region by construction)."""
        result_clean, result_artifact, _, _ = self._clean_and_artifact_results(seed=3)
        ratio = result_artifact.snr_tonic / result_clean.snr_tonic
        self.assertAlmostEqual(ratio, 1.0, delta=0.5)

    def test_artifact_itself_is_detected_as_a_phasic_peak(self):
        """The artifact should surface as an inspectable detected peak in its own
        right, not be silently absorbed or ignored."""
        _, result_artifact, onset_idx, end_idx = self._clean_and_artifact_results(seed=4)
        in_artifact_window = (result_artifact.peaks >= onset_idx) & (
            result_artifact.peaks <= end_idx
        )
        self.assertTrue(np.any(in_artifact_window))


class TestEnvelopeRRSNRConvenienceAPI(unittest.TestCase):
    """Test that the one-shot convenience methods agree with an equivalent .fit()
    call, since they're thin wrappers around it."""

    def test_estimate_matches_fit(self):
        """.estimate() returns (snr, noise, peaks) matching a .fit() result."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)

        result = estimator.fit(trace)
        snr, noise, peaks = estimator.estimate(trace)

        self.assertEqual(snr, result.snr)
        self.assertEqual(noise, result.noise)
        np.testing.assert_array_equal(peaks, result.peaks)

    def test_estimate_components_matches_fit(self):
        """.estimate_components() returns (snr_total, snr_tonic, snr_phasic)
        matching a .fit() result."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)

        result = estimator.fit(trace)
        snr_total, snr_tonic, snr_phasic = estimator.estimate_components(trace)

        self.assertEqual(snr_total, result.snr)
        self.assertEqual(snr_tonic, result.snr_tonic)
        self.assertEqual(snr_phasic, result.snr_phasic)

    def test_decompose_matches_fit(self):
        """.decompose() returns (tonic, residual) matching a .fit() result."""
        trace, tonic_true = make_tonic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)

        result = estimator.fit(trace)
        tonic, residual = estimator.decompose(trace)

        np.testing.assert_array_equal(tonic, result.tonic)
        np.testing.assert_array_equal(residual, result.residual)
        np.testing.assert_allclose(residual, trace - tonic)

    def test_decompose_reuses_cached_fit_when_trace_omitted(self):
        """.decompose() with no argument should reuse the most recent .fit(),
        not silently do nothing or re-fit."""
        trace, _ = make_tonic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)
        result = estimator.fit(trace)

        tonic, residual = estimator.decompose()

        np.testing.assert_array_equal(tonic, result.tonic)
        np.testing.assert_array_equal(residual, result.residual)


class TestEnvelopeRRSNRConstructorValidation(unittest.TestCase):
    """Test the not-yet-fitted guard and convenience-property consistency.

    Note: the cut-down estimator no longer exposes `signal_statistic` /
    `noise_method` / `tonic_method` / `tonic_range_method` knobs (it's a single
    fixed configuration: arPLS tonic + MAD noise + robust tonic range), so the
    corresponding validation tests from the fuller estimator don't apply here.
    """

    def test_accessing_result_before_fit_raises(self):
        """Every convenience property (and .decompose() with no argument) should
        raise a clear RuntimeError before .fit()/.estimate() has ever been called."""
        estimator = EnvelopeRRSNR(fps=FPS)
        for accessor in (
            lambda: estimator.snr_,
            lambda: estimator.snr_corrected_,
            lambda: estimator.snr_tonic_,
            lambda: estimator.snr_phasic_,
            lambda: estimator.noise_,
            lambda: estimator.peaks_,
            lambda: estimator.tonic_,
            lambda: estimator.residual_,
            lambda: estimator.decompose(),
        ):
            with self.assertRaises(RuntimeError):
                accessor()

    def test_convenience_properties_match_fit_result(self):
        """Every `*_` property should mirror the corresponding field on the most
        recent .fit() result."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)
        result = estimator.fit(trace)

        self.assertEqual(estimator.snr_, result.snr)
        self.assertEqual(estimator.snr_corrected_, result.snr_corrected)
        self.assertEqual(estimator.snr_tonic_, result.snr_tonic)
        self.assertEqual(estimator.snr_phasic_, result.snr_phasic)
        self.assertEqual(estimator.noise_, result.noise)
        np.testing.assert_array_equal(estimator.peaks_, result.peaks)
        np.testing.assert_array_equal(estimator.tonic_, result.tonic)
        np.testing.assert_array_equal(estimator.residual_, result.residual)


class TestEnvelopeRRSNRConfigKnobs(unittest.TestCase):
    """Test the remaining tuning knobs (window scaling) -- the tonic/noise/
    tonic-range algorithm choices themselves are no longer configurable (this
    productized estimator keeps only the single tuned configuration: arPLS
    tonic tracking, MAD noise estimation, robust tonic-range)."""

    def test_scale_window_basic(self):
        """scale_window preserves real-world duration across a different fps."""
        self.assertEqual(EnvelopeRRSNR.scale_window(20, fps=40.0), 40)
        self.assertEqual(EnvelopeRRSNR.scale_window(20, fps=20.0), 20)


class TestEnvelopeRRSNRBiasCorrection(unittest.TestCase):
    """Test bias_correction auto-apply/override logic and apply_correction=True
    on the one-shot convenience methods."""

    def test_default_config_auto_applies_tuned_correction(self):
        """Constructing with all-default settings should auto-apply the tuned bias correction."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit(trace)
        self.assertIsNotNone(result.snr_corrected)
        self.assertIsNotNone(result.snr_phasic_corrected)

    def test_non_default_peak_threshold_sd_disables_auto_correction(self):
        """The tuned correction was fit against peak_threshold_sd=2.0 specifically
        -- changing it should silently resolve bias_correction to None rather than
        misapplying a correction fit for a different threshold."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, peak_threshold_sd=3.0).fit(trace)
        self.assertIsNone(result.snr_corrected)
        self.assertIsNone(result.snr_phasic_corrected)

    def test_explicit_none_disables_correction_even_at_tuned_defaults(self):
        """bias_correction=None should disable the correction even when the rest of the
        config matches the tuned defaults exactly."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, bias_correction=None).fit(trace)
        self.assertIsNone(result.snr_corrected)

    def test_custom_bias_correction_tuple_is_used(self):
        """A custom (slope, intercept) tuple should be applied exactly as given."""
        trace, _ = make_phasic_only_trace()
        slope, intercept = 2.0, 0.5
        result = EnvelopeRRSNR(fps=FPS, bias_correction=(slope, intercept)).fit(trace)
        expected_phasic_corrected = (result.snr_phasic - intercept) / slope
        self.assertAlmostEqual(result.snr_phasic_corrected, expected_phasic_corrected)

    def test_estimate_apply_correction_returns_corrected_total(self):
        """.estimate(apply_correction=True) should return the corrected total, matching .fit()."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)
        result = estimator.fit(trace)
        snr_corrected, _, _ = estimator.estimate(trace, apply_correction=True)
        self.assertEqual(snr_corrected, result.snr_corrected)

    def test_estimate_components_apply_correction_returns_corrected_values(self):
        """.estimate_components(apply_correction=True) should return the corrected
        total/tonic/phasic triple, matching .fit()."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)
        result = estimator.fit(trace)
        total, tonic, phasic = estimator.estimate_components(trace, apply_correction=True)
        self.assertEqual(total, result.snr_corrected)
        self.assertEqual(tonic, result.snr_tonic)
        self.assertEqual(phasic, result.snr_phasic_corrected)

    def test_fit_bias_correction_from_benchmark_recovers_known_linear_bias(self):
        """Fitting against a synthetic, exactly-linear bias should recover its
        true slope/intercept."""
        true_snr = np.array([5.0, 10.0, 20.0, 40.0])
        snr_est = 0.9 * true_snr + 1.5  # simulated linear bias
        slope, intercept = EnvelopeRRSNR.fit_bias_correction_from_benchmark(true_snr, snr_est)
        self.assertAlmostEqual(slope, 0.9, places=6)
        self.assertAlmostEqual(intercept, 1.5, places=6)


class TestPrivateHelperEdgeCases(unittest.TestCase):
    """Test a handful of defensive edge cases in private helper functions that
    aren't reachable through EnvelopeRRSNR's public API in normal operation (the
    public API's own validation/preprocessing prevents these inputs from ever
    reaching them), but are still worth pinning down directly since they're real
    safety nets against degenerate data."""

    def test_robust_std_on_empty_array_returns_nan(self):
        """An empty input has no meaningful spread to estimate -- should return NaN."""
        self.assertTrue(np.isnan(_robust_std(np.array([]))))

    def test_mad_noise_std_returns_nan_on_nan_input(self):
        """.fit() always replaces NaNs in the input trace before this is ever
        reached, so this guards a case the public API itself prevents."""
        self.assertTrue(np.isnan(_mad_noise_std(np.array([1.0, 2.0, np.nan, 3.0]))))

    def test_mad_noise_std_does_not_collapse_to_nan_on_heavy_contamination(self):
        """Regression test: a short window heavily dominated by one huge outlier
        used to make the second trim step empty out completely and return NaN,
        which would silently poison any downstream aggregation across windows."""
        rng = np.random.default_rng(1)
        x = NOISE_SIGMA * rng.standard_normal(600)
        x[300:350] += 50.0  # huge, mostly-dominant spike
        result = _mad_noise_std(x)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)

    def test_detect_peaks_rise_rate_handles_a_candidate_at_the_trace_start(self):
        """A candidate at index 0 has no samples before it to compute a rise slope
        from -- should be handled gracefully (treated as a zero slope), not raise."""
        residual = np.concatenate([[0.0], np.linspace(-1, 1, 25)])
        candidates = np.array([0, 5, 10, 15, 20])  # >=5, so gating actually runs
        kept = _detect_peaks_rise_rate(residual, candidates, sigma=0.1, rise_window=3)
        self.assertIsInstance(kept, np.ndarray)

    def test_detect_peaks_rise_rate_handles_few_candidates(self):
        """With exactly 5 (still >=5, so gating runs) candidates and distinct
        slopes, the below-median half naturally has fewer than 5 elements --
        exercises the percentile fallback for the slope threshold."""
        residual = np.linspace(-1, 1, 20)
        candidates = np.array([5, 8, 11, 14, 17])
        kept = _detect_peaks_rise_rate(residual, candidates, sigma=0.1, rise_window=3)
        self.assertIsInstance(kept, np.ndarray)

    def test_mad_noise_std_returns_rstd_when_second_trim_empties_completely(self):
        """If the second trim step's own scale estimate collapses to (near) zero
        rather than genuinely emptying the array, the function should fall back
        to the first-pass scale estimate rather than trusting a degenerate
        second-pass one. Scripted via mock, in call order: the first
        _robust_std call (on the first trim) returns a genuine small positive
        value, the second (on the second trim) returns exactly 0 -- forcing
        `if result > 0` to fail and fall through to the first call's value.
        Naturally constructing this exact internal state from a residual array
        alone is impractical, since real data essentially never gives an
        exactly-zero robust scale on a non-trivial sample."""
        residual = 0.01 * np.random.default_rng(5).standard_normal(300)
        call_values = iter([0.001, 0.0])
        with patch(
            "aind_dynamic_foraging_basic_analysis.metrics.snr_envelope_rr._robust_std",
            side_effect=lambda x: next(call_values),
        ):
            result = _mad_noise_std(residual)
        self.assertEqual(result, 0.001)

    def test_mad_noise_std_falls_back_to_full_residual_when_first_pass_collapses(self):
        """If even the FIRST-pass scale estimate collapses to (non-positive)
        zero, the second trim is skipped entirely (there's no positive scale to
        trim against) and the function falls all the way back to a robust
        scale of the whole (untrimmed) detrended residual. Scripted via mock:
        the first _robust_std call returns exactly 0, so only one more call
        happens (on the untrimmed residual, not a second trim)."""
        residual = 0.01 * np.random.default_rng(5).standard_normal(300)
        call_values = iter([0.0, 0.007])
        with patch(
            "aind_dynamic_foraging_basic_analysis.metrics.snr_envelope_rr._robust_std",
            side_effect=lambda x: next(call_values),
        ):
            result = _mad_noise_std(residual)
        self.assertEqual(result, 0.007)

    def test_mad_noise_std_returns_nan_on_empty_input(self):
        """An empty residual has nothing to estimate a noise floor from -- should
        return NaN rather than raise."""
        self.assertTrue(np.isnan(_mad_noise_std(np.array([]))))

    def test_unset_sentinel_repr(self):
        """_UnsetType's repr should be short and unambiguous for debugging --
        never actually shown to a normal caller (it's a default-argument
        sentinel), so nothing else exercises it."""
        self.assertEqual(repr(_UNSET), "<unset>")
        self.assertIsInstance(_UNSET, _UnsetType)

    def test_robust_tonic_range_falls_back_to_ptp_with_fewer_than_two_extrema(self):
        """A perfectly flat tonic curve has no local extrema at all -- should
        fall back to ptp(tonic) (here 0.0, correctly) rather than raise or
        return garbage from an empty median."""
        tonic = np.full(200, 0.5)
        tonic_range, n_swings = _robust_tonic_range(tonic, min_distance=20)
        self.assertEqual(tonic_range, 0.0)
        self.assertEqual(n_swings, 0)


if __name__ == "__main__":
    unittest.main()
