"""
Function definitions for:
- Effective hydration shell pressure calculation
- Curve fitting
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


# Functions for secondary axis for denaturation studies
def phi_to_P(phi, P0=1):
    """Converts phi (kJ/mol) to effective hydration shell P (kbar) given P0 (bar)

    Args:
        phi (np.float): Value of phi in kJ/mol
        P0 (np.float): Value of system (simulation) pressure in bar (optional, default = 1 bar)

    Returns:
        P (np.float): Effective hydration shell pressure in kbar.
    """
    rho_w = 997 / (18.015e-3)  # mol/m^3
    # phi: kJ/mol -> J/mol; rho_w phi: Pa -> bar; bar -> kbar
    P = (P0 - rho_w * (1000 * phi) * 1e-5) * 1e-3
    return P


def P_to_phi(P, P0=1):
    """Converts effective hydration shell P (kbar) to phi (kJ/mol) given P0 (bar)

    Args:
        P (np.float): Value of effective hydration shell pressure in kbar
        P0 (np.float): Value of system (simulation) pressure in bar (optional, default = 1 bar)

    Returns:
        phi (np.float): phi potential in kJ/mol
    """
    rho_w = 997 / (18.015e-3)  # mol/m^3
    phi = 1e5 * (P0 - 1000 * P) / (1e3 * rho_w)
    return phi


# Functions for curve fitting for critical phi analysis
def linear_model(x, A, B):
    """
    Function defining linear model
    """
    y = A * x + B
    return y


def fit_linear_model(xdata, ydata, yerr, p_guess):
    """Fits linear model to data with error bars using
    weighted least squares regression.

    Args:
        xdata (ndarray): 1-dimensional array containing x values
        ydata (ndarray): 1-dimensional array containing y values
        yerr (ndarray): 1-dimensional array containing y standard errors
        p_guess (ndarray): 1-dimensional array of length 2 containing initial guesses
                           for curve fitting

    Returns:
        2-d tuple (popt, perr, chi_sq), where
        popt = array containing the fitted parameters for the linear model.
        perr = array containing errors corresponding to fitted parameters
        chi_sq = chi squared statistic
    """
    popt, cov = curve_fit(linear_model, xdata, ydata, sigma=yerr,
                          p0=p_guess, absolute_sigma=True)

    # Estimate errors
    perr = np.zeros(len(popt))
    for i in range(len(popt)):
        perr[i] = np.absolute(cov[i][i]) ** 0.5

    # Chi-squared of fit
    y = ydata
    y_pred = linear_model(xdata, *popt)
    chi_sq = np.sum((y - y_pred) ** 2 / yerr ** 2)

    return popt, perr, chi_sq


def integrated_step_gaussian(x, A, B, C, D, E, F):
    """
    Function defining the integrated step gaussian model.
    """
    SH = 30  # Sharpness factor
    y = A * x + 1 / SH * (B - A) * np.log(np.exp(SH * x) + np.exp(SH * C)) - 0.5 * np.sqrt(np.pi) * D * E * erf((C - x) / E)
    return y + F


def derivative_integrated_step_gaussian(x, A, B, C, D, E, F):
    """
    Function defining the derivative of the integrated step gaussian model (aka step + gaussian)
    """
    SH = 30  # Sharpness factor
    dy_dx = A + (B - A) / (1 + np.exp(-SH * (x - C))) + D * np.exp(-(x - C) ** 2 / E ** 2)
    return dy_dx


def penalized_integrated_step_gaussian(x, *p):
    """
    Integrated step gaussian model with a penalty for positive slopes (which
    gets added to the residual during curve fitting).

    Warning:
        This function has been deprecated, as it can be unstable in certain situtations.
        Use curve fitting with upper and lower bounds instead (implemented in `fit_integrated_step_gaussian`).
    """
    A = p[0]
    B = p[1]
    D = p[3]
    # A should be negative or zero, linear slope after transition
    # B should be negative or zero, linear slope before transition
    # D should be negative
    slope_penalty = (np.heaviside(A, 0) * A) ** 2 \
        + (np.heaviside(B, 0) * B) ** 2 \
        + (np.heaviside(D, 0) * D) ** 2
    return integrated_step_gaussian(x, *p) + slope_penalty


def fit_integrated_step_gaussian(xdata, ydata, yerr, p_guess):
    r"""Fits the integrated step gaussian model to data with error bars using
    weighted least squares regression.

    The integrated step gaussian model is defined as:

    .. math:: y = A \cdot x + \frac{B - A}{S} \log(S \cdot x + \exp(S \cdot C)) - \frac{\sqrt{\pi}}{2} \cdot D \cdot E \cdot erf(\frac{C - x}{E})

    Args:
        xdata (ndarray): 1-dimensional array containing x values
        ydata (ndarray): 1-dimensional array containing y values
        yerr (ndarray): 1-dimensional array containing y standard errors
        p_guess (ndarray): 1-dimensional array of length 6 containing initial guesses
                           for curve fitting

    Returns:
        2-d tuple (popt, perr, chi_sq), where
        popt = array containing the fitted parameters for the integrated step gaussian model.
        perr = array containing errors corresponding to fitted parameters
        chi_sq = chi squared statistic
    """
    #                        A    B    C    D    E    F
    lower_bounds = np.array([-np.inf, -np.inf, min(xdata), -np.inf, 0, -np.inf])
    upper_bounds = np.array([0, 0, max(xdata), 0, np.inf, np.inf])

    popt, cov = curve_fit(integrated_step_gaussian, xdata, ydata, sigma=yerr,
                          p0=p_guess, absolute_sigma=True, bounds=(lower_bounds, upper_bounds))

    # Estimate errors
    perr = np.zeros(len(popt))
    for i in range(len(popt)):
        perr[i] = np.absolute(cov[i][i]) ** 0.5

    # Chi-squared of fit
    y = ydata
    y_pred = integrated_step_gaussian(xdata, *popt)
    chi_sq = np.sum((y - y_pred) ** 2 / yerr ** 2)

    return popt, perr, chi_sq
