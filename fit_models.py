from scipy.optimize import curve_fit
import numpy as np
from models import build_model

def fit_model(x, y, model1, model2, mask1, mask2, p0_1=None, bounds_2=None, p0_2=None, bounds_2=None):
    """
    Generic 2-peak fitter.

    Parameters
    ----------
    model1, model2 : str
        "gauss", "voigt", "asym", "skew"
    mask1, mask2 : str
        x ranges where peak 1 and peak 2 are present
    p0_1, p0_2 : list
        Initial parameters (user-defined or auto)

    bounds_1, bounds_2 : tuple (lower, upper)
        Parameter bounds

    Returns
    -------
    dict with popt, fits, residuals
    """

    model_fun, n1, n2 = build_model(model1, model2)

    # ---- default initial guess ----
    if p0_1 is None:
        a0 = np.max(y[mask_1])
        width = (x.max() - x.min()) / 2000

        def default_params(model, a_guess):
            if model == "gauss":
                return [a_guess, x[np.argmax(y[mask_1]), width]

            elif model == "voigt":
                return [a_guess, x[np.argmax(y[mask_1])], width, width / 2]

            elif model == "asym":
                return [a_guess, x[np.argmax(y[mask_1])], width, width, width / 2]

            elif model == "skew":
                return [a_guess, x[np.argmax(y[mask_1])], width, width / 2, 0]

        p0 = (
            default_params(model1, a0) +
            default_params(model2, a0 / 2)
        )

    # ---- default bounds (fully free) ----
    if bounds is None:
        lower = [-np.inf] * len(p0)
        upper = [np.inf] * len(p0)
        bounds = (lower, upper)

    # ---- fit ----
    popt, _ = curve_fit(model_fun, x, y, p0=p0, bounds=bounds)

    # ---- evaluate ----
    y_fit = model_fun(x, *popt)

    f1, _ = build_model(model1, model2)[0], None  # reuse split logic
    peak1 = lambda x, *p: build_model(model1, model2)[0](x, *p[:n1])
    peak2 = lambda x, *p: build_model(model1, model2)[0](x, *p[n1:])

    peak1_fit = peak1(x, *popt)
    peak2_fit = peak2(x, *popt)

    residual = y - y_fit

    return {
        "popt": popt,
        "total_fit": y_fit,
        "peak1_fit": peak1_fit,
        "peak2_fit": peak2_fit,
        "residual": residual,
        "p0": p0
    }
