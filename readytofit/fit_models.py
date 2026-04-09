from scipy.optimize import curve_fit
import numpy as np
from .models import build_model
from .parameters import flatten_params, unflatten_params, generate_default_p0, validate_bounds, generate_default_bounds
from .peak_detection import estimate_initial_parameters
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
        To fix a parameter: set lower[i] = value-1e-12, upper[i] = value-1e-12.  
        Please note: lower[i] == upper[i] will raise an exception from scipy

    debug : bool, default=False
        If True, prints fitted parameter values.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "popt": optimized parameters (flat list, free parameters only)
        - "params": structured parameters per peak (list of dicts with all parameters including fixed ones)
        - "param_names": names of free parameters
        - "total_fit": full fitted curve
        - "peak_fits": list of individual peak curves
        - "residual": y - total_fit
        - "rmse": root mean square error
        - "p0": final initial guess used
        - "bounds": final bounds used
        - "model_function": callable model
        - "param_slices": parameter index ranges per peak
    """

    # ---- Build composite model ----
    model_fun, param_slices, param_names = build_model(peaks)

    # ---- Initial guess handling ----
    if p0 is None:
        # Use intelligent peak detection to estimate initial parameters
        if debug:
            print("Using peak detection for initial parameter estimation...")
        peaks_estimated = estimate_initial_parameters(x, y, peaks)
        p0 = peaks_estimated
    
    final_p0 = flatten_params(peaks, p0)

    # Fallback if invalid p0
    if final_p0 is None:
        if debug:
            print("Invalid p0 → falling back to manual default guess")
        peaks_estimated = estimate_initial_parameters(x, y, peaks)
        p0 = peaks_estimated
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

    # ---- Unflatten parameters back to structured format ----
    params_structured = unflatten_params(peaks, popt)

    # ---- Package results ----
    result = {
        "popt": popt,
        "params": params_structured,  # Structured parameters with fixed values included
        "param_names": param_names,
        "total_fit": total_fit,
        "peak_fits": peak_fits,
        "residual": residual,
        "rmse": np.sqrt(np.mean(residual**2)),
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
