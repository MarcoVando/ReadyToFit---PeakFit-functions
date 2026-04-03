import functions
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile, erf


def build_model(model1, model2):

    def get_model_and_nparams(model):
        if model == "gauss":
            return (lambda x, A, mu, sigma: gaussian(x, A, mu, sigma), 3)

        elif model == "voigt":
            return (lambda x, A, mu, sigma, gamma: voigt(x, A, mu, sigma, gamma), 4)

        elif model == "asym-voigt":
            return (lambda x, A, mu, sigma_L, sigma_R, gamma:
                    asym_voigt(x, A, sigma_L, sigma_R, gamma, mu), 5)

        elif model == "skew":
            return (lambda x, A, sigma, gamma, alpha, mu:
                    skew_voigt_fixed_mu(x, A, sigma, gamma, alpha, mu), 5)
        elif model == None:
            return (0,0)
        else:
            raise ValueError(f"Unknown model: {model}")

    f1, n1 = get_model_and_nparams(model1)
    f2, n2 = get_model_and_nparams(model2)

    return lambda x, *params: (
        f1(x, *params[:n1]) +
        f2(x, *params[n1:n1+n2])
    )

# --- usage example
#model = build_model("gauss", "voigt")
#y = model(x,
#          A1, mu1, sigma1,          # gauss
#          A2, mu2, sigma2, gamma2) # voigt
