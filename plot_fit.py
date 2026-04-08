from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_fit_result( x: np.ndarray,y: np.ndarray,result: Dict,show_residual: bool = True,residual_offset: float = -0.1, show_rmse: bool = True,) -> None:
    """
    General-purpose plotting for multi-peak fitting.

    Features
    --------
    - Raw data
    - Total fit
    - Individual peaks (dashed, thin)
    - Filled peak areas
    - Residual plot
    - RMSE annotation

    Parameters
    ----------
    x, y : array-like
        Input data.

    result : dict
        Output from `fit_model`.

    show_residual : bool
        Whether to plot residual.

    residual_offset : float
        Vertical offset for residual.

    show_rmse : bool
        Whether to display RMSE on plot.
    """

    fig, ax = plt.subplots()

    # ---- raw data ----
    ax.plot(x, y, label="data", linewidth=2)

    # ---- total fit ----
    total_fit = result["total_fit"]
    ax.plot(x, total_fit, color="red", label="total fit", linewidth=2)

    # ---- individual peaks ----
    peak_fits = result.get("peak_fits", [])

    for i, peak in enumerate(peak_fits):
        # dashed thin line
        ax.plot(
            x,
            peak,
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label=f"peak {i}"
        )

        # filled area
        ax.fill_between(
            x,
            peak,
            alpha=0.25
        )

    # ---- residual ----
    if show_residual:
        residual = y - total_fit

        ax.plot(
            x,
            residual + residual_offset,
            color="black",
            linewidth=1,
            label="residual"
        )

        ax.plot(
            x,
            np.zeros_like(x) + residual_offset,
            color="gray",
            alpha=0.5
        )

    # ---- RMSE ----
    if show_rmse:
        residual = y - total_fit
        rmse = np.sqrt(np.mean(residual**2))

        ax.text(
            0.02,
            0.95,
            f"RMSE = {rmse:.4g}",
            transform=ax.transAxes,
            verticalalignment="top"
        )

    ax.legend()
    plt.tight_layout()
    plt.show()
