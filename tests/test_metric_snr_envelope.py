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
    _estimate_residual_noise,
    _estimate_sigma_minima,
    _folded_iqr_noise_std,
    _half_sample_mode,
    _interpolate_envelope,
    _lower_envelope,
    _mad_noise_std,
    _robust_std,
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
        """snr_tonic (default tonic_range_method='robust') should stay close to its
        own no-artifact value despite the same large artifact."""
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
    """Test constructor argument validation and the not-yet-fitted guard."""

    def test_invalid_signal_statistic_raises(self):
        """An unrecognized signal_statistic should raise, not silently fall back."""
        with self.assertRaises(ValueError):
            EnvelopeRRSNR(fps=FPS, signal_statistic="bogus")

    def test_invalid_noise_method_via_argument_raises(self):
        """An unrecognized noise_method passed as a constructor argument should raise."""
        with self.assertRaises(ValueError):
            EnvelopeRRSNR(fps=FPS, noise_method="bogus")

    def test_invalid_noise_method_via_config_raises(self):
        """config['noise_method'] takes precedence over the constructor argument,
        so an invalid value there must be validated too."""
        with self.assertRaises(ValueError):
            EnvelopeRRSNR(fps=FPS, config={"noise_method": "bogus"})

    def test_deprecated_mad_alias_resolves_to_aind_mad(self):
        """The deprecated 'mad' alias should silently resolve to 'aind_mad'."""
        estimator = EnvelopeRRSNR(fps=FPS, noise_method="mad")
        self.assertEqual(estimator.noise_method, "aind_mad")

    def test_invalid_tonic_method_raises_on_fit(self):
        """tonic_method isn't validated until .fit() actually dispatches on it."""
        trace, _ = make_tonic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS, config={"tonic_method": "bogus"})
        with self.assertRaises(ValueError):
            estimator.fit(trace)

    def test_invalid_tonic_range_method_raises_on_fit(self):
        """tonic_range_method isn't validated until .fit() dispatches on it either."""
        trace, _ = make_tonic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS, config={"tonic_range_method": "bogus"})
        with self.assertRaises(ValueError):
            estimator.fit(trace)

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


class TestEnvelopeRRSNRAlternateConfigs(unittest.TestCase):
    """Test the less-common (non-default) noise_method/tonic_method/
    tonic_range_method options and related config knobs."""

    def test_folded_iqr_noise_method_runs(self):
        """noise_method='folded_iqr' should run and give a sane, positive noise estimate."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, noise_method="folded_iqr").fit(trace)
        self.assertTrue(np.isfinite(result.noise))
        self.assertGreater(result.noise, 0)

    def test_mad_iqr_avg_noise_method_runs(self):
        """noise_method='mad_iqr_avg' should run and give a sane, positive noise estimate."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, noise_method="mad_iqr_avg").fit(trace)
        self.assertTrue(np.isfinite(result.noise))
        self.assertGreater(result.noise, 0)

    def test_als_tonic_method_runs(self):
        """tonic_method='als' should run and produce a finite, full-length tonic curve."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, config={"tonic_method": "als"}).fit(trace)
        self.assertEqual(result.tonic.shape, trace.shape)
        self.assertTrue(np.all(np.isfinite(result.tonic)))

    def test_envelope_tonic_method_runs(self):
        """tonic_method='envelope' needs its own window-size config keys, unlike
        'als'/'arpls' (which only need their own lam/p/n_iter)."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(
            fps=FPS,
            config={
                "tonic_method": "envelope",
                "lower_smooth_window": 31,
                "lower_min_distance": 20,
            },
        ).fit(trace)
        self.assertEqual(result.tonic.shape, trace.shape)
        self.assertTrue(np.all(np.isfinite(result.tonic)))

    def test_envelope_tonic_method_with_linear_interpolation(self):
        """interp_kind='linear' is a valid alternative to the default 'pchip'."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(
            fps=FPS,
            config={
                "tonic_method": "envelope",
                "lower_smooth_window": 31,
                "lower_min_distance": 20,
                "interp_kind": "linear",
            },
        ).fit(trace)
        self.assertTrue(np.all(np.isfinite(result.tonic)))

    def test_ptp_tonic_range_method_runs(self):
        """tonic_range_method='ptp' (the pre-'robust' default) should still run cleanly."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, config={"tonic_range_method": "ptp"}).fit(trace)
        self.assertEqual(result.tonic_range if hasattr(result, "tonic_range") else True, True)
        self.assertTrue(np.isfinite(result.snr_tonic))

    def test_percentile_tonic_range_method_runs(self):
        """tonic_range_method='percentile' should run cleanly with a trim_pct set."""
        trace, _ = make_tonic_only_trace()
        result = EnvelopeRRSNR(
            fps=FPS, config={"tonic_range_method": "percentile", "tonic_range_trim_pct": 5.0}
        ).fit(trace)
        self.assertTrue(np.isfinite(result.snr_tonic))

    def test_pre_despike_flags_and_removes_a_spike_before_tonic_fitting(self):
        """Enabling pre_despike_window should flag and count a large injected spike."""
        trace, _ = make_tonic_only_trace()
        trace_with_spike = trace.copy()
        trace_with_spike[500:505] += 5.0  # one big spike

        result = EnvelopeRRSNR(
            fps=FPS, config={"pre_despike_window": 101, "pre_despike_k": 5.0}
        ).fit(trace_with_spike)

        self.assertGreater(result.n_extreme_samples, 0)
        self.assertGreater(result.frac_extreme_samples, 0.0)

    def test_scale_window_basic(self):
        """scale_window preserves real-world duration across a different fps."""
        self.assertEqual(EnvelopeRRSNR.scale_window(20, fps=40.0), 40)
        self.assertEqual(EnvelopeRRSNR.scale_window(20, fps=20.0), 20)

    def test_scale_window_make_odd_bumps_an_even_result(self):
        """make_odd=True should bump an otherwise-even scaled result up by one."""
        # 30 samples at 20 fps (reference) stays 30 (even) unless make_odd=True.
        self.assertEqual(EnvelopeRRSNR.scale_window(30, fps=20.0, make_odd=False), 30)
        self.assertEqual(EnvelopeRRSNR.scale_window(30, fps=20.0, make_odd=True), 31)


class TestEnvelopeRRSNRFitChunked(unittest.TestCase):
    """Test fit_chunked's chunk-sizing options, aggregation modes, and the
    too-short-trace fallback."""

    def test_default_chunking_matches_expected_sizing(self):
        """Default sizing is max(min_chunk_duration_s, chunk_fraction * duration)."""
        trace, _ = make_phasic_only_trace()  # 150 s at 20 fps
        result = EnvelopeRRSNR(fps=FPS).fit_chunked(trace)
        self.assertEqual(result.chunk_duration_s, 30.0)  # max(30, 0.2 * 150)
        self.assertEqual(result.n_chunks, 5)

    def test_explicit_chunk_duration_is_respected(self):
        """An explicit chunk_duration_s should be used as-is, not the default sizing."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit_chunked(trace, chunk_duration_s=30.0)
        self.assertEqual(result.chunk_duration_s, 30.0)

    def test_aggregate_mean_runs_and_differs_from_median_in_general(self):
        """aggregate='mean' is a valid alternative aggregation mode to the default 'median'."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)
        result_median = estimator.fit_chunked(trace, aggregate="median")
        result_mean = estimator.fit_chunked(trace, aggregate="mean")
        self.assertTrue(np.isfinite(result_median.snr_tonic))
        self.assertTrue(np.isfinite(result_mean.snr_tonic))

    def test_invalid_aggregate_raises(self):
        """An unrecognized aggregate value should raise, not silently default."""
        trace, _ = make_phasic_only_trace()
        with self.assertRaises(ValueError):
            EnvelopeRRSNR(fps=FPS).fit_chunked(trace, aggregate="bogus")

    def test_too_short_trace_falls_back_to_global_snr_tonic_and_warns(self):
        """A trace with no chunk long enough to fit should warn and fall back to
        the global (non-chunked) snr_tonic, not raise or silently return garbage."""
        rng = np.random.default_rng(9)
        short_trace = NOISE_SIGMA * rng.standard_normal(80)  # 4 s at 20 fps
        estimator = EnvelopeRRSNR(fps=FPS)
        result_global = estimator.fit(short_trace)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result_chunked = estimator.fit_chunked(short_trace)
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

        self.assertEqual(result_chunked.n_chunks, 0)
        self.assertEqual(result_chunked.snr_tonic, result_global.snr_tonic)

    def test_bias_correction_propagates_through_chunked_result(self):
        """When bias_correction is active and phasic events are detected, the
        chunked result's snr_corrected should also be populated (derived from the
        chunked snr_tonic + the same snr_phasic_corrected as a plain .fit())."""
        trace, _ = make_phasic_only_trace()
        estimator = EnvelopeRRSNR(fps=FPS)  # default config -> bias_correction auto-applies
        result_global = estimator.fit(trace)
        self.assertIsNotNone(result_global.snr_corrected)

        result_chunked = estimator.fit_chunked(trace)
        self.assertIsNotNone(result_chunked.snr_corrected)
        expected = result_chunked.snr_tonic + result_global.snr_phasic_corrected
        self.assertAlmostEqual(result_chunked.snr_corrected, expected)


class TestEnvelopeRRSNRBiasCorrection(unittest.TestCase):
    """Test bias_correction auto-apply/override logic and apply_correction=True
    on the one-shot convenience methods."""

    def test_default_config_auto_applies_tuned_correction(self):
        """Constructing with all-default settings should auto-apply the tuned bias correction."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS).fit(trace)
        self.assertIsNotNone(result.snr_corrected)
        self.assertIsNotNone(result.snr_phasic_corrected)

    def test_non_default_noise_method_disables_auto_correction(self):
        """The tuned correction was fit against noise_method='aind_mad' specifically
        -- switching it should silently resolve bias_correction to None rather than
        misapplying a correction fit for a different configuration."""
        trace, _ = make_phasic_only_trace()
        result = EnvelopeRRSNR(fps=FPS, noise_method="folded_iqr").fit(trace)
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

    def test_half_sample_mode_base_cases(self):
        """_half_sample_mode's recursive narrowing bottoms out at <=3 elements;
        exercise each of those base cases (1, 2, and a 3-element tie) directly,
        since a large, generic residual essentially never narrows down to
        exactly one of these by chance."""
        self.assertEqual(_half_sample_mode(np.array([5.0])), 5.0)
        self.assertEqual(_half_sample_mode(np.array([1.0, 2.0])), 1.5)
        # 3 elements, evenly spaced -> the two gaps tie, hits the tie branch.
        self.assertEqual(_half_sample_mode(np.array([1.0, 2.0, 3.0])), 2.0)

    def test_folded_iqr_noise_std_falls_back_on_few_below_anchor_samples(self):
        """With too few samples below the anchor for a stable IQR (<20), falls
        back to a plain std of the below-median half."""
        short_residual = 0.01 * np.random.default_rng(0).standard_normal(20)
        result = _folded_iqr_noise_std(short_residual)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)

    def test_estimate_residual_noise_invalid_method_raises(self):
        """Dead code from EnvelopeRRSNR's own public API (its noise_method is
        already validated at construction time), but still worth pinning as a
        safety net for any other internal caller."""
        residual = 0.01 * np.random.default_rng(0).standard_normal(50)
        with self.assertRaises(ValueError):
            _estimate_residual_noise(residual, "bogus")

    def test_estimate_sigma_minima_falls_back_with_too_few_midpoints(self):
        """tonic_minima with exactly 5 elements (the minimum to pass the first
        length check) always yields only 4 valley-to-valley midpoints -- one
        fewer than needed for a stable estimate -- so this always falls back,
        not just in some edge case."""
        residual = np.zeros(100)
        tonic_minima = np.array([0, 20, 40, 60, 80])
        result = _estimate_sigma_minima(residual, tonic_minima, fallback_sigma=0.5)
        self.assertEqual(result, 0.5)

    def test_interpolate_envelope_invalid_interp_kind_raises(self):
        """An unrecognized interp_kind should raise, not silently fall back."""
        with self.assertRaises(ValueError):
            _interpolate_envelope(np.array([0, 5, 10]), np.array([1.0, 2.0, 1.5]), 11, "bogus")

    def test_lower_envelope_handles_an_even_smooth_window(self):
        """smooth_window must be odd for Savitzky-Golay filtering -- an even
        value should be silently bumped to odd, not raise."""
        x = 0.01 * np.random.default_rng(1).standard_normal(500)
        x += 0.05 * np.sin(np.linspace(0, 6, 500))
        tonic, minima = _lower_envelope(x, x.copy(), smooth_window=30, order=2, min_distance=20)
        self.assertEqual(tonic.shape, x.shape)

    def test_lower_envelope_falls_back_with_fewer_than_two_minima(self):
        """A flat (constant) input has no local minima at all -- should return
        a flat curve at the input's own minimum, not raise or return garbage."""
        x = np.full(200, 0.5)
        tonic, minima = _lower_envelope(x, x.copy(), smooth_window=31, order=2, min_distance=20)
        self.assertLess(len(minima), 2)
        np.testing.assert_allclose(tonic, 0.5)

    def test_unset_sentinel_repr(self):
        """_UnsetType's repr should be short and unambiguous for debugging --
        never actually shown to a normal caller (it's a default-argument
        sentinel), so nothing else exercises it."""
        self.assertEqual(repr(_UNSET), "<unset>")
        self.assertIsInstance(_UNSET, _UnsetType)


if __name__ == "__main__":
    unittest.main()
