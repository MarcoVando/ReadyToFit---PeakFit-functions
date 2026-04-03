import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile, erf

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def lorentzian(x, A, mu, gamma):
    return A * (gamma**2 / ((x - mu)**2 + gamma**2))
    
def voigt(x, A, mu, sigma, gamma):
    return A * voigt_profile(x - mu, sigma, gamma)

def asym_voigt(x, A, mu, sigma_L, sigma_R, gamma):
    sigma = np.where(x < mu, sigma_L, sigma_R)
    return A * voigt_profile(x - mu, sigma, gamma)

def skew_voigt(x, A, mu, sigma, gamma, alpha):
    base = voigt_profile(x - mu, sigma, gamma)
    skew = 1 + erf(alpha * (x - mu))
    return A * base * skew
