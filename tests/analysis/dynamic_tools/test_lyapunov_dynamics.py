import numpy as np

from btorch.analysis.dynamic_tools.lyapunov_dynamics import (
    compute_expansion_to_contraction_ratio,
    compute_lyapunov_exponent_spectrum,
    compute_max_lyapunov_exponent,
)


def _logistic_map(r: float, x0: float, n_steps: int, burn_in: int = 300):
    """Generate a logistic map time series with a burn-in transient."""
    x = np.empty(n_steps + burn_in, dtype=float)
    x[0] = x0
    for i in range(1, x.size):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x[burn_in:]


def _logistic_map_lyapunov(series: np.ndarray, r: float):
    """Estimate logistic map Lyapunov exponent from the map derivative."""
    deriv = np.abs(r * (1.0 - 2.0 * series))
    return np.mean(np.log(deriv))


def test_max_lyapunov_exponent_logistic_map():
    # Chaotic regime (r=4) should yield a positive Lyapunov exponent.
    r_chaotic = 4.0
    # Short series keeps runtime reasonable while preserving chaos.
    series_chaotic = _logistic_map(r=r_chaotic, x0=0.1234, n_steps=3000)
    # Remove the DC component to avoid low-frequency warnings in nolds.
    series_chaotic_centered = series_chaotic - series_chaotic.mean()
    # nolds estimates Lyapunov from a reconstructed trajectory.
    lyap_chaotic = compute_max_lyapunov_exponent(
        series_chaotic_centered, emb_dim=6, lag=1, tau=1
    )
    # Analytic Lyapunov for the logistic map: mean(log|f'(x)|).
    analytic_lyap = _logistic_map_lyapunov(series_chaotic, r=r_chaotic)
    assert np.isfinite(lyap_chaotic)
    assert lyap_chaotic > 0.0
    assert np.isfinite(analytic_lyap)
    assert 0.4 < analytic_lyap < 0.9
    # Allow some estimator bias but keep it bounded.
    # nolds can underestimate for short series; keep a loose bound.
    assert abs(lyap_chaotic - analytic_lyap) < 0.75

    # Periodic regime (r=2.5) should yield a negative Lyapunov exponent.
    r_periodic = 2.5
    # Periodic dynamics should have contraction on average.
    series_periodic = _logistic_map(r=r_periodic, x0=0.1234, n_steps=3000)
    # Add tiny jitter to avoid -inf from exact periodicity in the estimator.
    rng = np.random.default_rng(0)
    series_periodic = series_periodic + rng.normal(
        0.0, 1e-10, size=series_periodic.size
    )
    # Remove the mean to reduce low-frequency warnings in nolds.
    series_periodic_centered = series_periodic - series_periodic.mean()
    lyap_periodic = compute_max_lyapunov_exponent(
        series_periodic_centered, emb_dim=6, lag=1, tau=1
    )
    analytic_periodic = _logistic_map_lyapunov(series_periodic, r=r_periodic)
    assert np.isfinite(lyap_periodic)
    assert lyap_periodic < 0.0
    assert analytic_periodic < 0.0
    # Periodic signals can still bias toward 0 in short reconstructions.
    assert abs(lyap_periodic - analytic_periodic) < 0.8


def test_lyapunov_spectrum_logistic_map_signs():
    # Spectrum should contain at least one positive exponent in chaos.
    series = _logistic_map(r=4.0, x0=0.1234, n_steps=3000)
    # emb_dim and matrix_dim must satisfy nolds constraints.
    spectrum = np.asarray(
        compute_lyapunov_exponent_spectrum(series, emb_dim=5, matrix_dim=3)
    )
    assert spectrum.size == 3
    assert np.max(spectrum) > 0.0

    # Expansion/contraction ratio should be positive and finite for mixed spectra.
    ratio = compute_expansion_to_contraction_ratio(spectrum)
    # If there is contraction, the ratio should be finite.
    has_contraction = np.any(spectrum < 0.0)
    if has_contraction:
        assert np.isfinite(ratio)
    assert ratio > 0.0
