"""
Peak detection and initial parameter estimation utilities.

Provides automatic peak detection and intelligent initial parameter guesses
for multi-peak fitting, while respecting user-defined fixed parameters.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import List, Dict, Tuple, Optional


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: Optional[int] = None,
    height_threshold: float = 0.1,
    prominence_threshold: Optional[float] = None,
    distance: Optional[int] = None,
) -> List[Dict]:
    """
    Automatically detect peaks in data and estimate their parameters.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., x-axis data).
    y : np.ndarray
        Dependent variable (e.g., y-axis data).
    n_peaks : int, optional
        Expected number of peaks. If provided, returns the n_peaks strongest peaks.
        If None, uses scipy's find_peaks with automatic thresholding.
    height_threshold : float, default=0.1
        Minimum peak height as a fraction of max(y). Used if n_peaks is None.
    prominence_threshold : float, optional
        Minimum prominence for a peak. If None, auto-estimated.
    distance : int, optional
        Minimum distance between peaks (in data points). If None, auto-estimated.

    Returns
    -------
    peaks_info : list of dict
        Each dict contains estimated parameters for a peak:
        {
            "mu": float - peak center (x-coordinate)
            "A": float - peak amplitude (height above baseline)
            "sigma": float - estimated standard deviation (FWHM / 2.355)
            "prominence": float - peak prominence (for ranking quality)
        }
    """
    
    # Baseline estimation (using minimum as baseline)
    baseline = np.min(y)
    y_normalized = y - baseline

    # Auto-estimate distance between peaks if needed
    if distance is None:
        mean_spacing = (x.max() - x.min()) / 20  # Assume 20 peak widths max
        distance = max(1, int(len(x) * mean_spacing / (x.max() - x.min())))

    # Auto-estimate prominence if needed
    if prominence_threshold is None:
        prominence_threshold = (y_normalized.max() - y_normalized.min()) * 0.05

    # Find peaks
    peak_indices, properties = find_peaks(
        y_normalized,
        height=height_threshold * y_normalized.max(),
        prominence=prominence_threshold,
        distance=distance,
    )

    if len(peak_indices) == 0:
        # Fallback: find at least the global maximum
        peak_indices = np.array([np.argmax(y_normalized)])
        properties = {"prominences": np.array([y_normalized[peak_indices[0]]]),
                     "left_bases": np.array([0]),
                     "right_bases": np.array([len(y) - 1])}

    # If n_peaks is specified, keep only the strongest peaks
    if n_peaks is not None and len(peak_indices) > n_peaks:
        # Sort by prominence (quality metric) and keep top n_peaks
        sorted_idx = np.argsort(properties["prominences"])[::-1][:n_peaks]
        peak_indices = peak_indices[sorted_idx]
        
        # Re-sort by x position
        position_order = np.argsort(x[peak_indices])
        peak_indices = peak_indices[position_order]

    # Calculate widths using half-maximum
    widths = peak_widths(y_normalized, peak_indices, rel_height=0.5)
    width_points = widths[0]  # Width in data points

    # Compile peak information
    peaks_info = []
    for idx, width_pts in zip(peak_indices, width_points):
        mu = x[idx]
        amplitude = y_normalized[idx]
        
        # Convert width from data points to x-axis units
        if width_pts > 0:
            width_x = width_pts * (x[1] - x[0]) if len(x) > 1 else width_pts
        else:
            # Fallback: estimate from neighboring points
            width_x = (x.max() - x.min()) / len(peak_indices) / 3
        
        # Standard deviation from FWHM: σ = FWHM / 2.355
        sigma = width_x / 2.355
        sigma = max(sigma, (x.max() - x.min()) / 100)  # Minimum reasonable width

        peaks_info.append({
            "mu": mu,
            "A": amplitude,
            "sigma": sigma,
            "prominence": properties["prominences"][
                list(peak_indices).index(idx)
            ] if "prominences" in properties else amplitude,
        })

    # Sort by position (left to right)
    peaks_info.sort(key=lambda p: p["mu"])

    return peaks_info


def estimate_initial_parameters(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict],
) -> List[Dict]:
    """
    Estimate intelligent initial parameters for fitting.

    For each peak definition, estimates initial parameters using peak detection.
    If a parameter is fixed (e.g., "mu" specified), it is preserved and not overwritten.

    Parameters
    ----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    peaks : list of dict
        Peak definitions. Each must have "model" key.
        May optionally have fixed parameters (e.g., "mu": 50.0).

    Returns
    -------
    estimated_peaks : list of dict
        Each dict contains estimated parameters for that peak,
        with fixed parameters preserved from input.

    Examples
    --------
    >>> peaks = [
    ...     {"model": "gauss"},  # Free parameters
    ...     {"model": "gauss", "mu": 50},  # Fixed center at x=50
    ... ]
    >>> estimated = estimate_initial_parameters(x, y, peaks)
    >>> # estimated[0] has auto-detected mu, sigma, A
    >>> # estimated[1] has mu=50 (preserved), sigma and A estimated
    """
    
    # Auto-detect peaks
    detected = detect_peaks(x, y, n_peaks=len(peaks))

    estimated_peaks = []
    for i, (peak_def, detected_info) in enumerate(zip(peaks, detected)):
        peak_params = peak_def.copy()

        # Model type (required)
        model = peak_params["model"]

        # Estimate amplitude (A)
        if "A" not in peak_params:
            peak_params["A"] = max(detected_info["A"], 0.01)

        # Estimate center (mu) - unless fixed
        if "mu" not in peak_params:
            peak_params["mu"] = detected_info["mu"]
        # else: mu is fixed, preserve user's value

        # Estimate width (sigma for Gaussian-like, or specific for model)
        sigma_est = detected_info["sigma"]

        if model == "gauss":
            if "sigma" not in peak_params:
                peak_params["sigma"] = sigma_est

        elif model == "voigt":
            if "sigma" not in peak_params:
                peak_params["sigma"] = sigma_est
            if "gamma" not in peak_params:
                peak_params["gamma"] = sigma_est / 2

        elif model == "asym":
            if "sigma_L" not in peak_params:
                peak_params["sigma_L"] = sigma_est
            if "sigma_R" not in peak_params:
                peak_params["sigma_R"] = sigma_est
            if "gamma" not in peak_params:
                peak_params["gamma"] = sigma_est / 2

        elif model == "skew":
            if "sigma" not in peak_params:
                peak_params["sigma"] = sigma_est
            if "gamma" not in peak_params:
                peak_params["gamma"] = sigma_est / 2
            if "alpha" not in peak_params:
                peak_params["alpha"] = 0.0  # No skew by default

        estimated_peaks.append(peak_params)

    return estimated_peaks
