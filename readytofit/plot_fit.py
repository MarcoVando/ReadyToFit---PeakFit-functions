\"\"\"Visualization utilities for fitting results.

Provides comprehensive plotting of multi-peak fitting results with:
- Raw data and total fit overlay
- Individual peak contributions (dashed lines)
- Shaded peak areas
- Residual curve
- RMSE annotation

Designed for easy comparison of data vs. fitted model.
\"\"\"

from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_fit_result( x: np.ndarray, y: np.ndarray, result: Dict, show_residual: bool = True,
                     residual_offset: float = -0.1, show_rmse: bool = True, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None) -> (plt.Figure, plt.Axes):
    """
    Plot multi-peak fitting results.

    Displays:
    - Raw data
    - Total fit
    - Individual peak contributions
    - Filled peak areas
    - Residual (optional)
    - RMSE (optional)

    Parameters
    ----------
    x, y : array-like
        Input data.

    result : dict
        Output from `fit_model`, must contain:
        - "total_fit"
        - "peak_fits"

    show_residual : bool
        Plot residual if True.

    residual_offset : float
        Vertical offset for residual visualization.

    show_rmse : bool
        Display RMSE on plot if True
    
    fig, ax : matplotlib Figure and Axes, optional
        If provided, plots on these axes instead of creating new ones.
    
    Returns
    ----------
    fig, ax : matplotlib Figure and Axes
        The figure and axes containing the plot.
    
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1, tight_layout=True)

    # =========================================================
    # DATA + TOTAL FIT
    # =========================================================
    ax.plot(x, y, label="data", linewidth=2)

    total_fit = result["total_fit"]
    ax.plot(x, total_fit, color="red", linewidth=2, label="total fit")

    # =========================================================
    # INDIVIDUAL PEAKS
    # =========================================================
    for i, peak in enumerate(result.get("peak_fits", [])):
        # dashed peak contribution
        ax.plot( x, peak, linestyle="--", linewidth=1, alpha=0.9, label=f"peak {i}")

        # filled area under peak
        ax.fill_between(x, peak, alpha=0.25)

    # =========================================================
    # RESIDUAL
    # =========================================================
    if show_residual:
        residual = y - total_fit

        ax.plot(x, residual + residual_offset, color="black",
            linewidth=1, label="residual")

        ax.plot(x, np.full_like(x, residual_offset), color="gray", alpha=0.5)

    # =========================================================
    # RMSE METRIC
    # =========================================================
    if show_rmse:
        rmse = np.sqrt(np.mean((y - total_fit) ** 2))
        ax.text(0.02, 0.95, f"RMSE = {rmse:.4g}", transform=ax.transAxes, verticalalignment="top")

    ax.legend()
    plt.tight_layout()
    
    return fig, ax
