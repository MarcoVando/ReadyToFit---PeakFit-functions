"""Parameter management, flattening, bounds handling, and initial guess generation.

This module handles:
- Flattening nested peak parameter dicts into flat optimization arrays
- Unflattening fitted parameters back to structured format
- Fixed parameter handling (removing from optimization, reinserting after fit)
- Default bounds generation for each peak model
- Intelligent initial guess generation from data statistics
- Bounds validation
"""

import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---- Parameter definitions per model ----
PARAM_ORDER = {
    "gauss": ["A", "mu", "sigma"],
    "voigt": ["A", "mu", "sigma", "gamma"],
    "asym":  ["A", "mu", "sigma_L", "sigma_R", "gamma"],
    "skew":  ["A", "mu", "sigma", "gamma", "alpha"],
}


# ---------------------------------------------------------------------
# PARAMETER TRANSFORMATIONS
# ---------------------------------------------------------------------

def flatten_params(peaks: List[Dict], p0_list: List[Dict]) -> Optional[List[float]]:
    """
    Convert structured parameter dictionaries into a flat list.

    This is required by scipy.optimize.curve_fit.

    Parameters
    ----------
    peaks : list of dict
        Peak definitions.

    p0_list : list of dict
        Initial parameters (one dict per peak).

    Returns
    -------
    flat : list of float or None
        Flattened parameter list, or None if input is invalid.
    """

    # ---- Validate input type ----
    if not isinstance(p0_list, list) or not all(isinstance(p, dict) for p in p0_list):
        warnings.warn(
            "p0 must be a list of dictionaries (one per peak). Falling back to defaults.",
            UserWarning
        )
        return None

    # ---- Validate length ----
    if len(p0_list) != len(peaks):
        warnings.warn(
            f"p0 length ({len(p0_list)}) != number of peaks ({len(peaks)}). "
            "Missing peaks will use defaults.",
            UserWarning
        )

    flat_params = []

    for peak_idx, peak in enumerate(peaks):
        model = peak["model"]

        if model not in PARAM_ORDER:
            raise ValueError(f"Unknown model: {model}")

        # Remove fixed parameters
        param_names = PARAM_ORDER[model]
        if "mu" in peak:
            param_names = [n for n in param_names if n != "mu"]

        # Safe access to user parameters
        params = p0_list[peak_idx] if peak_idx < len(p0_list) else {}

        for name in param_names:
            if name in params:
                flat_params.append(params[name])
            else:
                warnings.warn(
                    f"Missing parameter '{name}' for peak {peak_idx}. Using default value.",
                    UserWarning
                )
                flat_params.append(1.0)  # fallback default

    return flat_params


def unflatten_params(peaks: List[Dict], popt: List[float]) -> List[Dict]:
    """
    Convert flat parameter array back into structured dictionaries.

    Parameters
    ----------
    peaks : list of dict
        Peak definitions.

    popt : list of float
        Optimized parameters (flat).

    Returns
    -------
    params : list of dict
        Structured parameters per peak.
    """

    structured_params = []
    idx = 0

    for peak in peaks:
        model = peak["model"]

        if model not in PARAM_ORDER:
            raise ValueError(f"Unknown model: {model}")

        param_names = PARAM_ORDER[model]

        # Remove fixed parameters
        if "mu" in peak:
            param_names = [n for n in param_names if n != "mu"]

        peak_params = {}

        for name in param_names:
            peak_params[name] = popt[idx]
            idx += 1

        # Reinsert fixed μ
        if "mu" in peak:
            peak_params["mu"] = peak["mu"]

        structured_params.append(peak_params)

    return structured_params


# ---------------------------------------------------------------------
# DEFAULT GENERATORS
# ---------------------------------------------------------------------

def generate_default_p0(peaks: List[Dict], x: np.ndarray, y: np.ndarray) -> List[Dict]:
    """
    Generate default initial parameters (structured form).

    Parameters
    ----------
    peaks : list of dict
        Peak definitions.

    x, y : array-like
        Data used to estimate initial values.

    Returns
    -------
    p0 : list of dict
        Default parameter guesses.
    """

    p0 = []

    # Global estimates from data
    A0 = np.max(y)
    x0 = x[np.argmax(y)]
    width = (x.max() - x.min()) / 20

    for peak_idx, peak in enumerate(peaks):
        model = peak["model"]

        if model not in PARAM_ORDER:
            raise ValueError(f"Unknown model: {model}")

        params = {}

        # Amplitude scaled per peak
        params["A"] = A0 / (peak_idx + 1)

        # Only include μ if not fixed
        if "mu" not in peak:
            params["mu"] = x0

        # Model-specific defaults
        if model == "gauss":
            params["sigma"] = width

        elif model == "voigt":
            params["sigma"] = width
            params["gamma"] = width / 2

        elif model == "asym":
            params["sigma_L"] = width
            params["sigma_R"] = width
            params["gamma"] = width / 2

        elif model == "skew":
            params["sigma"] = width
            params["gamma"] = width / 2
            params["alpha"] = 0

        p0.append(params)

    return p0


def generate_default_bounds(peaks: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Generate default parameter bounds (no constraints).

    Parameters
    ----------
    peaks : list of dict
        Peak definitions.

    Returns
    -------
    bounds : tuple (lower, upper)
        Flat bounds lists for curve_fit.
    """

    lower, upper = [], []

    for peak in peaks:
        model = peak["model"]

        if model not in PARAM_ORDER:
            raise ValueError(f"Unknown model: {model}")

        param_names = PARAM_ORDER[model]

        # Remove fixed μ
        if "mu" in peak:
            param_names = [n for n in param_names if n != "mu"]

        for _ in param_names:
            lower.append(-np.inf)
            upper.append(np.inf)

    return lower, upper


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_bounds(
    bounds: Optional[Tuple[List[float], List[float]]],
    expected_len: int
) -> Optional[Tuple[List[float], List[float]]]:
    """
    Validate bounds structure and size.

    Parameters
    ----------
    bounds : tuple or None
        Candidate bounds.

    expected_len : int
        Expected number of parameters.

    Returns
    -------
    bounds : tuple or None
        Validated bounds or None if invalid.
    """

    if bounds is None:
        return None

    # Structure check
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        warnings.warn(
            "Bounds must be a tuple (lower, upper). Using defaults.",
            UserWarning
        )
        return None

    lower, upper = bounds

    # Length check
    if len(lower) != expected_len or len(upper) != expected_len:
        warnings.warn(
            f"Bounds size mismatch (expected {expected_len}). Using defaults.",
            UserWarning
        )
        return None

    return bounds
