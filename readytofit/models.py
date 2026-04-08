"""Model building and composite function construction.

This module constructs composite multi-peak models from individual peak
definitions. It handles:
- Building composite functions that sum multiple peaks
- Managing parameter indexing across all peaks
- Handling fixed parameters (e.g., fixed peak centers)
- Parameter naming and organization
"""

from .functions import *
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile, erf

from typing import Callable, List, Dict, Tuple


def build_model(peaks: List[Dict]) -> Tuple[Callable, List[Tuple[int, int]], List[str]]:
    """
    Build a composite model function from a list of peak definitions.

    Each peak specifies a model type and optionally fixed parameters
    (currently only "mu" is supported as fixed).

    Parameters
    ----------
    peaks : list of dict
        Each dict describes a peak, e.g.:
        [
            {"model": "gauss", "mu": 50},   # mu fixed
            {"model": "lorentz"}            # all parameters free
        ]

    Returns
    -------
    model_fun : callable
        Function f(x, *params) that evaluates the sum of all peak models.

    param_slices : list of (int, int)
        Index ranges for each peak’s parameters in the flattened parameter list.

    param_names : list of str
        Names of all free parameters, labeled with peak index
        (e.g., ["amp_0", "sigma_0", "amp_1", "mu_1", ...]).
    """

    funcs = []            # List of individual peak functions
    param_slices = []     # Parameter index ranges per peak
    param_names = []      # Flattened parameter names

    param_idx = 0  # Tracks current index in the flattened parameter list

    for peak_idx, peak in enumerate(peaks):
        model_name = peak["model"]
        mu_fixed = peak.get("mu")

        # Retrieve base model function and its parameter names
        func, names = get_model(model_name)

        # ---- Handle fixed parameters (currently only "mu") ----
        if mu_fixed is not None:
            if "mu" not in names:
                raise ValueError(f"Model '{model_name}' has no 'mu' parameter.")

            mu_index = names.index("mu")

            def make_fixed_mu_func(f, mu_idx, mu_val):
                """
                Wrap the original function, inserting a fixed mu value
                into the correct position in the parameter list.
                """
                def wrapped(x, *params):
                    return f(
                        x,
                        *params[:mu_idx],
                        mu_val,
                        *params[mu_idx:]
                    )
                return wrapped

            # Replace function with wrapped version
            func = make_fixed_mu_func(func, mu_index, mu_fixed)

            # Remove "mu" from free parameter names
            names = [n for n in names if n != "mu"]

        # ---- Track parameter layout ----
        n_params = len(names)

        funcs.append(func)
        param_slices.append((param_idx, param_idx + n_params))

        # Label parameters with peak index (e.g., amp_0, sigma_0)
        param_names.extend(f"{name}_{peak_idx}" for name in names)

        param_idx += n_params

    # ---- Combined model function ----
    def model_fun(x, *params):
        """
        Evaluate the sum of all peak models.

        Parameters
        ----------
        x : array-like
            Input values.
        *params : float
            Flattened parameter list for all peaks.

        Returns
        -------
        y : array-like
            Model evaluation.
        """
        y = 0
        for func, (start, end) in zip(funcs, param_slices):
            y += func(x, *params[start:end])
        return y

    return model_fun, param_slices, param_names
