# -*- coding: utf-8 -*-
"""
Smoothing Functions Module

Provides various smoothing algorithms for XRD data preprocessing.

@author: candicewang928@gmail.com
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def apply_gaussian_smoothing(y, sigma=2):
    """
    Apply Gaussian smoothing to data.

    Parameters
    ----------
    y : array_like
        Input data
    sigma : float, optional
        Standard deviation for Gaussian kernel (higher = more smoothing)
        Default is 2.

    Returns
    -------
    ndarray
        Smoothed data
    """
    return gaussian_filter1d(y, sigma=sigma)


def apply_savgol_smoothing(y, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to data.

    Parameters
    ----------
    y : array_like
        Input data
    window_length : int, optional
        Length of the filter window (must be odd)
        Default is 11.
    polyorder : int, optional
        Order of the polynomial used to fit the samples
        Default is 3.

    Returns
    -------
    ndarray
        Smoothed data

    Notes
    -----
    The function automatically adjusts window_length if it's even or
    too large relative to the data size.
    """
    # Ensure window_length is odd and not larger than data
    window_length = min(window_length, len(y))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    return savgol_filter(y, window_length, polyorder)


def apply_smoothing(y, method='gaussian', **kwargs):
    """
    Apply smoothing to data using specified method.

    Parameters
    ----------
    y : array_like
        Input data
    method : {'gaussian', 'savgol'}, optional
        Smoothing method to use. Default is 'gaussian'.
    **kwargs : dict
        Additional parameters for the smoothing method:
        - For 'gaussian': sigma (float)
        - For 'savgol': window_length (int), polyorder (int)

    Returns
    -------
    ndarray
        Smoothed data

    Examples
    --------
    >>> y_smooth = apply_smoothing(y, method='gaussian', sigma=3)
    >>> y_smooth = apply_smoothing(y, method='savgol', window_length=15, polyorder=3)
    """
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 2)
        return apply_gaussian_smoothing(y, sigma=sigma)
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)
        return apply_savgol_smoothing(y, window_length=window_length, polyorder=polyorder)
    else:
        return y
