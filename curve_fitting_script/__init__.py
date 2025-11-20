# -*- coding: utf-8 -*-
"""
XRD Peak Fitting Toolkit
A comprehensive toolkit for X-ray diffraction peak analysis and fitting.

@author: candicewang928@gmail.com
"""

__version__ = "2.0.0"
__author__ = "candicewang928@gmail.com"

from .smoothing import apply_smoothing, apply_gaussian_smoothing, apply_savgol_smoothing
from .clustering import cluster_peaks_dbscan
from .background import (
    fit_global_background,
    find_background_points_auto,
    find_group_minima,
    create_piecewise_background
)
from .peak_fitting import (
    pseudo_voigt,
    voigt,
    calculate_fwhm,
    calculate_area,
    estimate_fwhm_robust
)

__all__ = [
    'apply_smoothing',
    'apply_gaussian_smoothing',
    'apply_savgol_smoothing',
    'cluster_peaks_dbscan',
    'fit_global_background',
    'find_background_points_auto',
    'find_group_minima',
    'create_piecewise_background',
    'pseudo_voigt',
    'voigt',
    'calculate_fwhm',
    'calculate_area',
    'estimate_fwhm_robust',
]
