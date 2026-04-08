from scipy.optimize import curve_fit
import numpy as np
from models import build_model
from typing import List, Dict, Tuple, Callable, Optional

def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Dict],
    p0: Optional[List[float]] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    debug: bool = False
) -> Dict:
    """
    Fit a multi-peak model to data using scipy.optimize.curve_fit.

    This function builds a composite model from multiple peak definitions
    and fits it to the provided (x, y) data.

    Parameters
    ----------
    x : array-like
        Independent variable data.

    y : array-like
        Dependent variable data.

    peaks : list of dict
        List of peak definitions. Each dict must include:
            - "model": str
                Model type ("gauss", "voigt", "asym", "skew")

        Optional:
            - "mu": float
                If provided, fixes the peak center (μ is not fitted).

    p0 : list of float, optional
        Initial parameter guess (flattened across all peaks).
        Can be partial or contain None values → auto-filled.

    bounds : tuple (lower, upper), optional
        Bounds for parameters. Each must match parameter length.
        Use None entries for defaults.
        To fix a parameter: set lower[i] == upper[i].

    debug : bool, default=False
        If True, prints fitted parameter values.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "popt": optimized parameters (flat list)
        - "param_names": parameter names
        - "total_fit": full fitted curve
        - "peak_fits": list of individual peak curves
        - "residual": y - total_fit
        - "p0": final initial guess used
        - "bounds": final bounds used
        - "model_function": callable model
        - "param_slices": parameter index ranges per peak
    """

    # ---- Build composite model ----
    model_fun, param_slices, param_names = build_model(peaks)

    # ---- Generate default initial guess (data-driven) ----
    A0 = np.max(y)
    x0 = x[np.argmax(y)]
    width = (x.max() - x.min()) / 20

    default_p0 = []

    for peak_idx, peak in enumerate(peaks):
        model = peak["model"]
        A_guess = A0 / (peak_idx + 1)

        # Define default parameters per model
        if model == "gauss":
            params = [A_guess, x0, width]

        elif model == "voigt":
            params = [A_guess, x0, width, width / 2]

        elif model == "asym":
            params = [A_guess, x0, width, width, width / 2]

        elif model == "skew":
            params = [A_guess, x0, width, width / 2, 0]

        else:
            raise ValueError(f"Unknown model type: {model}")

        # Remove fixed μ if specified
        if "mu" in peak:
            params.pop(1)

        default_p0.extend(params)

    # ---- Initial guess handling ----
    if p0 is None:
        p0 = generate_default_p0(peaks, x, y)

    final_p0 = flatten_params(peaks, p0)

    # Fallback if invalid p0
    if final_p0 is None:
        if debug:
            print("Invalid p0 → falling back to default guess")
        p0 = generate_default_p0(peaks, x, y)
        final_p0 = flatten_params(peaks, p0)

    # ---- Bounds handling ----
    validated_bounds = validate_bounds(bounds, len(final_p0))

    if validated_bounds is None:
        final_bounds = generate_default_bounds(peaks)
    else:
        final_bounds = validated_bounds

    # ---- Perform fit ----
    popt, _ = curve_fit(
        model_fun,
        x,
        y,
        p0=final_p0,
        bounds=final_bounds
    )

    # ---- Compute fitted curves ----
    total_fit = model_fun(x, *popt)

    # Individual peak contributions
    peak_fits = []
    for (start, end), peak in zip(param_slices, peaks):
        sub_params = popt[start:end]

        # Build single-peak model
        single_model, _, _ = build_model([peak])
        peak_fits.append(single_model(x, *sub_params))

    # Residuals
    residual = y - total_fit

    # ---- Package results ----
    result = {
        "popt": popt,
        "param_names": param_names,
        "total_fit": total_fit,
        "peak_fits": peak_fits,
        "residual": residual,
        "p0": final_p0,
        "bounds": final_bounds,
        "model_function": model_fun,
        "param_slices": param_slices,
    }

    # ---- Debug output ----
    if debug:
        print("\nFitted parameters:")
        for name, value in zip(param_names, popt):
            print(f"{name}: {value}")

    return result
