from typing import Optional, Dict, List
import numpy as np

def area_integration(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """
    Compute area under curve using trapezoidal rule.
    """
    return np.trapezoid(y, x)


def evaluate_peak_areas(x: np.ndarray, result: Dict) -> Dict:
    """
    Compute areas for total fit and individual peaks.

    Returns
    -------
    dict with:
        - total
        - peaks (list)
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
