import functions
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile, erf


def build_model(model1, model2):
    def get_model(model, prefix=""):
        if model == "gauss":
            return lambda x, A, mu, sigma: gaussian(x, A, mu, sigma)

        elif model == "lorentz":
            return lambda x, A, mu, sigma: lorentzian(x, A, mu, sigma)
           
        elif model == "voigt":
            return lambda x, A, mu, sigma, gamma: voigt(x, A, mu, sigma, gamma)

        elif model == "asym-voigt":
            return lambda x, A, mu, sigma_L, sigma_R, gamma: \
                asym_voigt(x, A, sigma_L, sigma_R, gamma, mu)

        elif model == "skew":
            return lambda x, A, sigma, gamma, alpha, mu: \
                skew_voigt_fixed_mu(x, A, sigma, gamma, alpha, mu)

        else:
            raise ValueError(f"Unknown model: {model}")

    f1 = get_model(model1)
    f2 = get_model(model2)

    # Combine both models
    return lambda x, *params: (
        f1(x, *params[:len(params)//2]) +
        f2(x, *params[len(params)//2:])
    )
