"""Area integration utilities for computing peak and total fit areas.

Provides functions for:
- Numerical integration of curves using the trapezoidal rule
- Computing individual peak areas from decomposed fit results
- Computing total signal area under the fitted curve
"""

from typing import Optional, Dict, List
import numpy as np

def area_integration(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """Compute area under curve using trapezoidal rule.
    
    Parameters
    ----------
    y : np.ndarray
        Values of the dependent variable.
    x : np.ndarray, optional
        Values of the independent variable (x-axis spacing).
        If None, uniform spacing (dx=1) is assumed.
        
    Returns
    -------
    float
        Area under the curve.
    """
    return np.trapezoid(y, x)


def evaluate_peak_areas(x: np.ndarray, result: Dict) -> Dict:
    """Compute areas for total fit and individual peaks.
    
    Integrates the total fitted curve and each individual peak contribution
    across the full x-range using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (x-axis).
    result : dict
        Fitting result dictionary from fit_model().
        Must contain "total_fit" and "peak_fits" keys.

    Returns
    -------
    dict
        Dictionary with:
        - "total" (float): area under the total fitted curve
        - "peaks" (list of float): areas of individual peaks
    """
    total_area = area_integration(result["total_fit"], x)

    peak_areas = [
        area_integration(peak, x)
        for peak in result["peak_fits"]
    ]

    return {
        "total": total_area,
        "peaks": peak_areas
    }
