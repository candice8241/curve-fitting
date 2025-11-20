# -*- coding: utf-8 -*-
"""
Peak Fitting Functions Module

Provides peak profile functions (Voigt, Pseudo-Voigt) and related utilities
for XRD peak fitting.

@author: candicewang928@gmail.com
"""

import numpy as np
from scipy.special import wofz
from scipy.signal import savgol_filter


def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """
    Pseudo-Voigt profile: weighted sum of Gaussian and Lorentzian.

    Parameters
    ----------
    x : array_like
        Independent variable
    amplitude : float
        Peak amplitude
    center : float
        Peak center position
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter
    eta : float
        Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)

    Returns
    -------
    ndarray
        Pseudo-Voigt profile values

    Notes
    -----
    The profile is: eta * Lorentzian + (1-eta) * Gaussian
    """
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian


def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile using the Faddeeva function.

    The Voigt profile is the convolution of a Gaussian and Lorentzian,
    computed efficiently using the complex error function (Faddeeva function).

    Parameters
    ----------
    x : array_like
        Independent variable
    amplitude : float
        Peak amplitude
    center : float
        Peak center position
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter

    Returns
    -------
    ndarray
        Voigt profile values

    Notes
    -----
    The Voigt profile is more physically accurate than Pseudo-Voigt but
    slightly more computationally expensive.
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def calculate_fwhm(sigma, gamma, eta):
    """
    Calculate Full Width at Half Maximum (FWHM) for Pseudo-Voigt profile.

    Parameters
    ----------
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter
    eta : float
        Mixing parameter

    Returns
    -------
    float
        FWHM value

    Notes
    -----
    FWHM is calculated as weighted average:
    FWHM = eta * FWHM_Lorentzian + (1-eta) * FWHM_Gaussian
    """
    fwhm_g = 2.355 * sigma  # 2*sqrt(2*ln(2)) * sigma
    fwhm_l = 2 * gamma
    return eta * fwhm_l + (1 - eta) * fwhm_g


def calculate_area(amplitude, sigma, gamma, eta):
    """
    Calculate integrated area under Pseudo-Voigt peak.

    Parameters
    ----------
    amplitude : float
        Peak amplitude
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter
    eta : float
        Mixing parameter

    Returns
    -------
    float
        Integrated area

    Notes
    -----
    Area is calculated as weighted average:
    Area = eta * Area_Lorentzian + (1-eta) * Area_Gaussian
    """
    area_g = amplitude * sigma * np.sqrt(2 * np.pi)
    area_l = amplitude * np.pi * gamma
    return eta * area_l + (1 - eta) * area_g


def estimate_fwhm_robust(x, y, peak_idx, smooth=True):
    """
    Robust FWHM estimation using interpolation at half-maximum.

    Parameters
    ----------
    x : array_like
        X data
    y : array_like
        Y data
    peak_idx : int
        Index of the peak maximum
    smooth : bool, optional
        Whether to smooth the data before estimation. Default is True.

    Returns
    -------
    fwhm : float
        Estimated FWHM
    baseline : float
        Estimated baseline level

    Notes
    -----
    This function uses linear interpolation to find the exact positions
    where the peak reaches half of its maximum height, providing a more
    accurate FWHM estimate than simple data point counting.
    """
    if smooth and len(y) > 11:
        try:
            y_smooth = savgol_filter(y, min(11, len(y)//2*2+1), 3)
        except:
            y_smooth = y
    else:
        y_smooth = y

    peak_height = y_smooth[peak_idx]

    # Estimate local baseline from edges
    n_edge = max(3, len(y) // 10)
    baseline = (np.mean(y_smooth[:n_edge]) + np.mean(y_smooth[-n_edge:])) / 2

    half_max = (peak_height + baseline) / 2

    # Find left half-max point with interpolation
    left_idx = peak_idx
    for j in range(peak_idx, 0, -1):
        if y_smooth[j] <= half_max:
            # Linear interpolation
            if y_smooth[j+1] != y_smooth[j]:
                frac = (half_max - y_smooth[j]) / (y_smooth[j+1] - y_smooth[j])
                left_x = x[j] + frac * (x[j+1] - x[j])
            else:
                left_x = x[j]
            break
    else:
        left_x = x[0]

    # Find right half-max point with interpolation
    for j in range(peak_idx, len(y_smooth)-1):
        if y_smooth[j] <= half_max:
            if y_smooth[j-1] != y_smooth[j]:
                frac = (half_max - y_smooth[j]) / (y_smooth[j-1] - y_smooth[j])
                right_x = x[j] - frac * (x[j] - x[j-1])
            else:
                right_x = x[j]
            break
    else:
        right_x = x[-1]

    fwhm = abs(right_x - left_x)

    # Sanity check
    dx = np.mean(np.diff(x))
    if fwhm < dx * 2:
        fwhm = dx * 8

    return fwhm, baseline
