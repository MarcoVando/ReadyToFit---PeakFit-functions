from scipy.optimize import curve_fit
import numpy as np
from models import build_model

def fit_model(x, y, model1, model2, p0=None, bounds=None):
    """
    Generic 2-peak fitter.

    Parameters
    ----------
    model1, model2 : str
        "gauss", "voigt", "asym", "skew"

    p0 : list
        Initial parameters (user-defined or auto)

    bounds : tuple (lower, upper)
        Parameter bounds

    Returns
    -------
    dict with popt, fits, residuals
    """

    model_fun, n1, n2 = build_model(model1, model2)

    # ---- default initial guess ----
    if p0 is None:
        A0 = np.max(y)
        width = (x.max() - x.min()) / 2000

        def default_params(model, A_guess):
            if model == "gauss":
                return [A_guess, x[np.argmax(y)], width]

            elif model == "voigt":
                return [A_guess, x[np.argmax(y)], width, width / 2]

            elif model == "asym":
                return [A_guess, x[np.argmax(y)], width, width, width / 2]

            elif model == "skew":
                return [A_guess, x[np.argmax(y)], width, width / 2, 0]

        p0 = (
            default_params(model1, A0) +
            default_params(model2, A0 / 2)
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
