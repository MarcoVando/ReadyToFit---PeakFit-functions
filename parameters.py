import warnings

PARAM_ORDER = {
    "gauss": ["A", "mu", "sigma"],
    "voigt": ["A", "mu", "sigma", "gamma"],
    "asym":  ["A", "mu", "sigma_L", "sigma_R", "gamma"],
    "skew":  ["A", "mu", "sigma", "gamma", "alpha"]
}

def flatten_params(peaks, p0_list):
    """
    Convert list-of-dicts into flat list for curve_fit
    Adds warnings for malformed input
    """

    # ---- TYPE CHECK ----
    if not isinstance(p0_list, list) or not all(isinstance(p, dict) for p in p0_list):
        warnings.warn(
            "p0 should be a list of dictionaries (one per peak). "
            "Falling back to automatic defaults.",
            UserWarning
        )
        return None  # signal fallback

    # ---- LENGTH CHECK ----
    if len(p0_list) != len(peaks):
        warnings.warn(
            f"p0 length ({len(p0_list)}) does not match number of peaks ({len(peaks)}). "
            "Missing peaks will use default values.",
            UserWarning
        )

    flat = []

    for i, peak in enumerate(peaks):
        model = peak["model"]
        names = PARAM_ORDER[model]

        if "mu" in peak:
            names = [n for n in names if n != "mu"]

        # safe access
        params = p0_list[i] if i < len(p0_list) else {}

        for name in names:
            if name in params:
                flat.append(params[name])
            else:
                warnings.warn(
                    f"Missing parameter '{name}' for peak {i}. Using default value.",
                    UserWarning
                )
                flat.append(1.0)

    return flat
    
def unflatten_params(peaks, popt):
    """
    Convert fitted array back to structured dict
    """

    out = []
    idx = 0

    for i, peak in enumerate(peaks):
        model = peak["model"]

        names = PARAM_ORDER[model]
        if "mu" in peak:
            names = [n for n in names if n != "mu"]

        params = {}

        for name in names:
            params[name] = popt[idx]
            idx += 1

        # re-add fixed mu
        if "mu" in peak:
            params["mu"] = peak["mu"]

        out.append(params)

    return out

def generate_default_p0(peaks, x, y):
    """
    Generate default initial parameters (list-of-dicts).
    """

    p0 = []

    A0 = np.max(y)
    x0 = x[np.argmax(y)]
    width = (x.max() - x.min()) / 20

    for i, peak in enumerate(peaks):
        model = peak["model"]

        params = {}

        # amplitude scaled per peak
        params["A"] = A0 / (i + 1)

        # mu only if not fixed
        if "mu" not in peak:
            params["mu"] = x0

        # model-specific params
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

def generate_default_bounds(peaks):
    """
    Generate default bounds (flat lists).
    """

    lower = []
    upper = []

    for peak in peaks:
        model = peak["model"]

        names = PARAM_ORDER[model]

        # remove fixed mu
        if "mu" in peak:
            names = [n for n in names if n != "mu"]

        for name in names:
            # default: no constraints
            lower.append(-np.inf)
            upper.append(np.inf)

    return (lower, upper)


def validate_bounds(bounds, expected_len):
    """
    Validate bounds length and structure.
    """

    if bounds is None:
        return None

    if not isinstance(bounds, tuple) or len(bounds) != 2:
        warnings.warn("Bounds must be a tuple (lower, upper). Using defaults.")
        return None

    lower, upper = bounds

    if len(lower) != expected_len or len(upper) != expected_len:
        warnings.warn(
            f"Bounds size mismatch (expected {expected_len}). Using defaults."
        )
        return None

    return bounds
