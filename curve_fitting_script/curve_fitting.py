# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - Improved Version
@author: candicewang928@gmail.com
Enhanced with better peak fitting algorithm and group-based background fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# ---------- Smoothing functions ----------
def apply_gaussian_smoothing(y, sigma=2):
    """
    Apply Gaussian smoothing to data.

    Parameters:
    -----------
    y : array
        Input data
    sigma : float
        Standard deviation for Gaussian kernel (higher = more smoothing)

    Returns:
    --------
    y_smooth : array
        Smoothed data
    """
    return gaussian_filter1d(y, sigma=sigma)


def apply_savgol_smoothing(y, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to data.

    Parameters:
    -----------
    y : array
        Input data
    window_length : int
        Length of the filter window (must be odd)
    polyorder : int
        Order of the polynomial used to fit the samples

    Returns:
    --------
    y_smooth : array
        Smoothed data
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

    Parameters:
    -----------
    y : array
        Input data
    method : str
        'gaussian' or 'savgol'
    **kwargs : dict
        Additional parameters for the smoothing method

    Returns:
    --------
    y_smooth : array
        Smoothed data
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


# ---------- DBSCAN clustering for peak grouping ----------
def cluster_peaks_dbscan(peak_positions, eps=None, min_samples=1):
    """
    Use DBSCAN density clustering to group nearby peaks.

    Parameters:
    -----------
    peak_positions : array
        1D array of peak positions (e.g., 2theta values)
    eps : float, optional
        Maximum distance between two peaks to be in the same group.
        If None, automatically estimated from data.
    min_samples : int
        Minimum number of peaks to form a cluster.

    Returns:
    --------
    labels : array
        Cluster labels for each peak (-1 means noise/outlier)
    n_clusters : int
        Number of clusters found
    """
    if len(peak_positions) == 0:
        return np.array([]), 0

    if len(peak_positions) == 1:
        return np.array([0]), 1

    # Reshape for sklearn
    X = np.array(peak_positions).reshape(-1, 1)

    # Auto-estimate eps if not provided
    if eps is None:
        # Use median distance between adjacent peaks as eps
        sorted_pos = np.sort(peak_positions)
        if len(sorted_pos) > 1:
            distances = np.diff(sorted_pos)
            eps = np.median(distances) * 1.5  # 1.5x median distance
        else:
            eps = 1.0

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    # Handle noise points (label -1) by assigning them to nearest cluster
    noise_mask = labels == -1
    if np.any(noise_mask) and np.any(~noise_mask):
        for i in np.where(noise_mask)[0]:
            # Find nearest non-noise point
            non_noise_idx = np.where(~noise_mask)[0]
            distances = np.abs(peak_positions[non_noise_idx] - peak_positions[i])
            nearest = non_noise_idx[np.argmin(distances)]
            labels[i] = labels[nearest]
    elif np.all(noise_mask):
        # All points are noise, treat as single cluster
        labels = np.zeros(len(labels), dtype=int)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


# ---------- Global background fitting ----------
def fit_global_background(x, y, peak_indices, method='spline', smoothing_factor=None, poly_order=3):
    """
    Fit a global smooth background to the data, excluding peak regions.

    Parameters:
    -----------
    x : array
        X data (2theta values)
    y : array
        Y data (intensity values)
    peak_indices : list
        Indices of detected peaks
    method : str
        'spline' for smooth spline, 'piecewise' for piecewise linear,
        'polynomial' for polynomial fit (smoothest, bounded curvature)
    smoothing_factor : float, optional
        Smoothing factor for spline (larger = smoother)
    poly_order : int, optional
        Order of polynomial for 'polynomial' method (default: 3)
        Order 2 = parabola (constant 2nd derivative)
        Order 3 = cubic (linear 2nd derivative)

    Returns:
    --------
    background : array
        Background values at each x point
    bg_points : list of tuples
        (x, y) coordinates of background anchor points
    """
    if len(peak_indices) == 0:
        # No peaks, return a simple baseline
        return np.full_like(y, np.median(y)), []

    # Sort peak indices
    sorted_peaks = sorted(peak_indices)

    # Find background anchor points (minima between peaks and at edges)
    bg_x = []
    bg_y = []

    # Left edge - find minimum in first region
    first_peak = sorted_peaks[0]
    left_region_end = max(0, first_peak - 5)
    if left_region_end > 0:
        left_min_idx = np.argmin(y[:left_region_end+1])
        bg_x.append(x[left_min_idx])
        bg_y.append(y[left_min_idx])
    else:
        bg_x.append(x[0])
        bg_y.append(y[0])

    # Find minima between adjacent peaks
    for i in range(len(sorted_peaks) - 1):
        idx1 = sorted_peaks[i]
        idx2 = sorted_peaks[i + 1]

        if idx2 > idx1 + 1:
            # Find minimum between peaks
            between_region = y[idx1:idx2+1]
            min_local = np.argmin(between_region)
            min_idx = idx1 + min_local
            bg_x.append(x[min_idx])
            bg_y.append(y[min_idx])

    # Right edge - find minimum in last region
    last_peak = sorted_peaks[-1]
    right_region_start = min(len(x) - 1, last_peak + 5)
    if right_region_start < len(x) - 1:
        right_min_idx = right_region_start + np.argmin(y[right_region_start:])
        bg_x.append(x[right_min_idx])
        bg_y.append(y[right_min_idx])
    else:
        bg_x.append(x[-1])
        bg_y.append(y[-1])

    bg_x = np.array(bg_x)
    bg_y = np.array(bg_y)

    # Sort by x
    sort_idx = np.argsort(bg_x)
    bg_x = bg_x[sort_idx]
    bg_y = bg_y[sort_idx]

    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(bg_x) > 0])
    bg_x = bg_x[unique_mask]
    bg_y = bg_y[unique_mask]

    bg_points = list(zip(bg_x, bg_y))

    if len(bg_x) < 2:
        return np.full_like(y, np.mean(bg_y) if len(bg_y) > 0 else np.median(y)), bg_points

    if method == 'polynomial':
        # Polynomial fit with bounded curvature
        # This ensures smooth background with controlled 2nd derivative
        try:
            # Fit polynomial to anchor points
            # poly_order: 2 = parabola (constant 2nd derivative = const curvature)
            #            3 = cubic (linear 2nd derivative)
            #            4+ = higher order (flexible but still smooth)

            # Ensure poly_order is not too high for number of points
            max_order = min(poly_order, len(bg_x) - 1, 5)  # Cap at 5th order

            # Fit polynomial
            coeffs = np.polyfit(bg_x, bg_y, max_order)
            poly = np.poly1d(coeffs)
            background = poly(x)

            # Ensure background doesn't go above maximum data value
            background = np.clip(background, None, np.max(y))

            # For 2nd order polynomial, the 2nd derivative is constant (2*a2)
            # For 3rd order, 2nd derivative is linear: 2*a2 + 6*a3*x
            # This ensures bounded curvature

        except Exception:
            # Fallback to linear interpolation
            background = np.interp(x, bg_x, bg_y)

    elif method == 'spline' and len(bg_x) >= 4:
        # Smooth spline interpolation
        if smoothing_factor is None:
            # Auto-determine smoothing factor for smooth curvature
            smoothing_factor = len(bg_x) * 1.0  # Increased for smoother background

        try:
            spline = UnivariateSpline(bg_x, bg_y, s=smoothing_factor, k=3)
            background = spline(x)
            # Ensure background doesn't go above data at anchor points
            background = np.clip(background, None, np.max(y))
        except Exception:
            # Fallback to linear interpolation
            background = np.interp(x, bg_x, bg_y)
    else:
        # Piecewise linear interpolation (least smooth)
        background = np.interp(x, bg_x, bg_y)

    return background, bg_points


# ---------- Peak profile functions ----------
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian"""
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile using Faddeeva function"""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def calculate_fwhm(sigma, gamma, eta):
    """Calculate FWHM from Pseudo-Voigt parameters"""
    fwhm_g = 2.355 * sigma
    fwhm_l = 2 * gamma
    return eta * fwhm_l + (1 - eta) * fwhm_g

def calculate_area(amplitude, sigma, gamma, eta):
    """Calculate integrated area"""
    area_g = amplitude * sigma * np.sqrt(2 * np.pi)
    area_l = amplitude * np.pi * gamma
    return eta * area_l + (1 - eta) * area_g

def estimate_fwhm_robust(x, y, peak_idx, smooth=True):
    """
    Robust FWHM estimation using interpolation
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


def find_group_minima(x, y, peak_indices):
    """
    Find local minima between adjacent peaks in a group.
    Returns a list of (x, y) points representing the minima including edges.

    Parameters:
    -----------
    x : array
        X data (2theta values)
    y : array
        Y data (intensity values)
    peak_indices : list
        List of peak indices in sorted order

    Returns:
    --------
    minima_points : list of tuples
        List of (x, y) coordinates for background interpolation points
    """
    if len(peak_indices) == 0:
        return []

    if len(peak_indices) == 1:
        # Single peak: use edges as background points
        idx = peak_indices[0]
        window = 30
        left_idx = max(0, idx - window)
        right_idx = min(len(x), idx + window)

        # Find minimum in left and right regions
        left_region = y[left_idx:idx]
        right_region = y[idx:right_idx]

        if len(left_region) > 0:
            left_min_local = np.argmin(left_region)
            left_min_idx = left_idx + left_min_local
        else:
            left_min_idx = left_idx

        if len(right_region) > 0:
            right_min_local = np.argmin(right_region)
            right_min_idx = idx + right_min_local
        else:
            right_min_idx = right_idx - 1

        return [(x[left_min_idx], y[left_min_idx]), (x[right_min_idx], y[right_min_idx])]

    # Multiple peaks: find minima between each adjacent pair
    minima_points = []

    # Left edge: find minimum before first peak
    first_peak = peak_indices[0]
    window = 30
    left_edge_idx = max(0, first_peak - window)
    left_region = y[left_edge_idx:first_peak]
    if len(left_region) > 0:
        left_min_local = np.argmin(left_region)
        left_min_idx = left_edge_idx + left_min_local
        minima_points.append((x[left_min_idx], y[left_min_idx]))
    else:
        minima_points.append((x[left_edge_idx], y[left_edge_idx]))

    # Find minima between adjacent peaks
    for i in range(len(peak_indices) - 1):
        idx1 = peak_indices[i]
        idx2 = peak_indices[i + 1]

        # Search for minimum between the two peaks
        if idx2 > idx1:
            between_region = y[idx1:idx2+1]
            min_local = np.argmin(between_region)
            min_idx = idx1 + min_local
            minima_points.append((x[min_idx], y[min_idx]))

    # Right edge: find minimum after last peak
    last_peak = peak_indices[-1]
    right_edge_idx = min(len(x) - 1, last_peak + window)
    right_region = y[last_peak:right_edge_idx+1]
    if len(right_region) > 0:
        right_min_local = np.argmin(right_region)
        right_min_idx = last_peak + right_min_local
        minima_points.append((x[right_min_idx], y[right_min_idx]))
    else:
        minima_points.append((x[right_edge_idx], y[right_edge_idx]))

    return minima_points


def create_piecewise_background(x_data, minima_points):
    """
    Create a piecewise linear background by interpolating between minima points.

    Parameters:
    -----------
    x_data : array
        X coordinates where background values are needed
    minima_points : list of tuples
        List of (x, y) coordinates for background interpolation

    Returns:
    --------
    background : array
        Background values at each x_data point
    """
    if len(minima_points) < 2:
        # Not enough points, return zeros
        return np.zeros_like(x_data)

    # Sort points by x coordinate
    sorted_points = sorted(minima_points, key=lambda p: p[0])
    bg_x = np.array([p[0] for p in sorted_points])
    bg_y = np.array([p[1] for p in sorted_points])

    # Linear interpolation
    background = np.interp(x_data, bg_x, bg_y)

    return background


# ---------- Main GUI Application ----------
class PeakFittingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive XRD Peak Fitting Tool - Improved")
        self.master.geometry("1400x850")
        self.master.configure(bg='#F0E6FA')

        # Data storage
        self.x = None
        self.y = None
        self.filename = None
        self.filepath = None
        self.selected_peaks = []
        self.peak_markers = []
        self.peak_texts = []
        self.fitted = False
        self.fit_results = None
        self.fit_lines = []

        # Background fitting storage
        self.bg_points = []
        self.bg_markers = []
        self.bg_line = None
        self.bg_connect_line = None
        self.selecting_bg = False

        # Undo stack
        self.undo_stack = []

        # Fitting method
        self.fit_method = tk.StringVar(value="pseudo_voigt")

        # Overlap mode for better handling of overlapping peaks
        self.overlap_mode = False

        # Distance threshold for grouping (in FWHM units)
        self.group_distance_threshold = 2.5

        # Smoothing settings
        self.smoothing_enabled = tk.BooleanVar(value=False)
        self.smoothing_method = tk.StringVar(value="gaussian")
        self.smoothing_sigma = tk.DoubleVar(value=2.0)
        self.smoothing_window = tk.IntVar(value=11)
        self.y_smoothed = None  # Store smoothed data

        self.create_widgets()

    def create_widgets(self):
        """Create all GUI components"""
        # Top control panel
        control_frame = tk.Frame(self.master, bg='#BA55D3', height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)

        btn_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 12,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 3
        }

        self.btn_load = tk.Button(control_frame, text="Load File",
                                   bg='#9370DB', fg='white',
                                   command=self.load_file, **btn_style)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_fit = tk.Button(control_frame, text="Fit Peaks",
                                 bg='#BA55D3', fg='white',
                                 command=self.fit_peaks, state=tk.DISABLED, **btn_style)
        self.btn_fit.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_reset = tk.Button(control_frame, text="Reset",
                                    bg='#FF69B4', fg='white',
                                    command=self.reset_peaks, state=tk.DISABLED, **btn_style)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_save = tk.Button(control_frame, text="Save Results",
                                  bg='#90EE90', fg='#006400',
                                  command=self.save_results, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_clear_fit = tk.Button(control_frame, text="Clear Fit",
                                       bg='#FF8C00', fg='white',
                                       command=self.clear_fit, state=tk.DISABLED, **btn_style)
        self.btn_clear_fit.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_undo = tk.Button(control_frame, text="Undo",
                                  bg='#DDA0DD', fg='#4B0082',
                                  command=self.undo_action, state=tk.DISABLED, **btn_style)
        self.btn_undo.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_auto_find = tk.Button(control_frame, text="Auto Find",
                                       bg='#4169E1', fg='white',
                                       command=self.auto_find_peaks, state=tk.DISABLED, **btn_style)
        self.btn_auto_find.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_overlap_mode = tk.Button(control_frame, text="Overlap",
                                          bg='#FF6B9D', fg='white',
                                          command=self.toggle_overlap_mode,
                                          state=tk.DISABLED, **btn_style)
        self.btn_overlap_mode.pack(side=tk.LEFT, padx=5, pady=8)

        self.status_label = tk.Label(control_frame, text="Please load a file to start",
                                     bg='#BA55D3', fg='white',
                                     font=('Arial', 11, 'bold'))
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Background fitting control panel
        bg_frame = tk.Frame(self.master, bg='#E6D5F5', height=50)
        bg_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        bg_frame.pack_propagate(False)

        bg_label = tk.Label(bg_frame, text="Background:",
                           bg='#E6D5F5', fg='#4B0082',
                           font=('Arial', 10, 'bold'))
        bg_label.pack(side=tk.LEFT, padx=10, pady=10)

        btn_bg_style = {
            'font': ('Arial', 9, 'bold'),
            'width': 14,
            'height': 1,
            'relief': tk.RAISED,
            'bd': 2
        }

        self.btn_select_bg = tk.Button(bg_frame, text="Select BG Points",
                                        bg='#B0A0D0', fg='#2F0060',
                                        command=self.toggle_bg_selection,
                                        state=tk.DISABLED, **btn_bg_style)
        self.btn_select_bg.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_subtract_bg = tk.Button(bg_frame, text="Subtract BG",
                                         bg='#90EE90', fg='#006400',
                                         command=self.subtract_background,
                                         state=tk.DISABLED, **btn_bg_style)
        self.btn_subtract_bg.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_clear_bg = tk.Button(bg_frame, text="Clear BG",
                                      bg='#FFB6C1', fg='#8B0000',
                                      command=self.clear_background,
                                      state=tk.DISABLED, **btn_bg_style)
        self.btn_clear_bg.pack(side=tk.LEFT, padx=5, pady=8)

        tk.Label(bg_frame, text="Fit Method:",
                bg='#E6D5F5', fg='#4B0082',
                font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(20, 5), pady=10)

        fit_method_combo = ttk.Combobox(bg_frame, textvariable=self.fit_method,
                                        values=["pseudo_voigt", "voigt"],
                                        state="readonly", width=12)
        fit_method_combo.pack(side=tk.LEFT, padx=5, pady=8)

        self.coord_label = tk.Label(bg_frame, text="",
                                    bg='#E6D5F5', fg='#4B0082',
                                    font=('Courier', 9))
        self.coord_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Smoothing control panel
        smooth_frame = tk.Frame(self.master, bg='#D5E6F5', height=50)
        smooth_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        smooth_frame.pack_propagate(False)

        smooth_label = tk.Label(smooth_frame, text="Smoothing:",
                               bg='#D5E6F5', fg='#0047AB',
                               font=('Arial', 10, 'bold'))
        smooth_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Smoothing enable checkbox
        self.chk_smooth = tk.Checkbutton(smooth_frame, text="Enable",
                                         variable=self.smoothing_enabled,
                                         bg='#D5E6F5', fg='#0047AB',
                                         font=('Arial', 9, 'bold'),
                                         command=self.on_smoothing_changed)
        self.chk_smooth.pack(side=tk.LEFT, padx=5, pady=8)

        # Smoothing method
        tk.Label(smooth_frame, text="Method:",
                bg='#D5E6F5', fg='#0047AB',
                font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2), pady=10)

        smooth_method_combo = ttk.Combobox(smooth_frame, textvariable=self.smoothing_method,
                                           values=["gaussian", "savgol"],
                                           state="readonly", width=8)
        smooth_method_combo.pack(side=tk.LEFT, padx=2, pady=8)
        smooth_method_combo.bind('<<ComboboxSelected>>', lambda e: self.on_smoothing_changed())

        # Sigma/Window parameter
        tk.Label(smooth_frame, text="Sigma:",
                bg='#D5E6F5', fg='#0047AB',
                font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2), pady=10)

        self.smooth_sigma_entry = tk.Entry(smooth_frame, textvariable=self.smoothing_sigma,
                                           width=5, font=('Arial', 9))
        self.smooth_sigma_entry.pack(side=tk.LEFT, padx=2, pady=8)

        tk.Label(smooth_frame, text="Window:",
                bg='#D5E6F5', fg='#0047AB',
                font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2), pady=10)

        self.smooth_window_entry = tk.Entry(smooth_frame, textvariable=self.smoothing_window,
                                            width=5, font=('Arial', 9))
        self.smooth_window_entry.pack(side=tk.LEFT, padx=2, pady=8)

        # Apply smoothing button
        self.btn_apply_smooth = tk.Button(smooth_frame, text="Apply",
                                          bg='#4682B4', fg='white',
                                          font=('Arial', 9, 'bold'),
                                          width=8, height=1,
                                          command=self.apply_smoothing_to_data,
                                          state=tk.DISABLED)
        self.btn_apply_smooth.pack(side=tk.LEFT, padx=10, pady=8)

        # Reset to original button
        self.btn_reset_smooth = tk.Button(smooth_frame, text="Reset Data",
                                          bg='#CD5C5C', fg='white',
                                          font=('Arial', 9, 'bold'),
                                          width=10, height=1,
                                          command=self.reset_to_original_data,
                                          state=tk.DISABLED)
        self.btn_reset_smooth.pack(side=tk.LEFT, padx=5, pady=8)

        # Results display panel
        results_frame = tk.Frame(self.master, bg='#F5E6FF', height=120)
        results_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        results_frame.pack_propagate(False)

        results_label = tk.Label(results_frame, text="Fitting Results:",
                                bg='#F5E6FF', fg='#4B0082',
                                font=('Arial', 10, 'bold'))
        results_label.pack(side=tk.TOP, anchor='w', padx=10, pady=5)

        columns = ('Peak', '2theta', 'FWHM', 'Area', 'Amplitude', 'Sigma', 'Gamma', 'Eta')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=4)

        col_widths = {'Peak': 50, '2theta': 100, 'FWHM': 100, 'Area': 100,
                      'Amplitude': 100, 'Sigma': 80, 'Gamma': 80, 'Eta': 60}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=col_widths.get(col, 80), anchor='center')

        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        style = ttk.Style()
        style.configure('Treeview', background='#FAF0FF', foreground='#4B0082',
                       font=('Courier', 9))
        style.configure('Treeview.Heading', font=('Arial', 9, 'bold'),
                       foreground='#4B0082')

        # Main plot area
        plot_frame = tk.Frame(self.master, bg='white')
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='white')
        self.ax.set_facecolor('#FAF0FF')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
        self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_title('Click on peaks to select | Use toolbar or scroll to zoom/pan',
                         fontsize=14, fontweight='bold', color='#9370DB')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg='#E6D5F5')
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.config(bg='#E6D5F5')

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Info panel
        info_frame = tk.Frame(self.master, bg='#F0E6FA', height=80)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        info_frame.pack_propagate(False)

        self.info_text = tk.Text(info_frame, height=4, bg='#FAF0FF',
                                 fg='#4B0082', font=('Courier', 10),
                                 relief=tk.SUNKEN, bd=2)
        self.info_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.info_text.insert('1.0', 'Welcome! Load your XRD data file to begin peak fitting.\n')
        self.info_text.insert('2.0', 'Use the toolbar buttons or mouse scroll wheel to zoom and pan.\n')
        self.info_text.insert('3.0', 'Click on peaks to select them for fitting.\n')
        self.info_text.config(state=tk.DISABLED)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax or self.x is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if event.button == 'up':
            scale_factor = 0.8
        elif event.button == 'down':
            scale_factor = 1.25
        else:
            return

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata - new_height * rely, ydata + new_height * (1 - rely)]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Display mouse coordinates"""
        if event.inaxes == self.ax and event.xdata is not None:
            self.coord_label.config(text=f"2theta: {event.xdata:.4f}  Intensity: {event.ydata:.2f}")
        else:
            self.coord_label.config(text="")

    def load_file(self):
        """Load XRD data file"""
        filepath = filedialog.askopenfilename(
            title="Select XRD Data File",
            filetypes=[("XY files", "*.xy"), ("DAT files", "*.dat"),
                       ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, encoding='latin1') as f:
                data = np.genfromtxt(f, comments="#")

            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns")

            self.x = data[:, 0]
            self.y = data[:, 1]
            self.y_original = self.y.copy()
            self.filepath = filepath
            self.filename = os.path.splitext(os.path.basename(filepath))[0]

            self.reset_peaks()
            self.clear_background()
            self.fitted = False
            self.undo_stack = []
            self.btn_undo.config(state=tk.DISABLED)

            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8, label='Data')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename}\nClick on peaks to select',
                            fontsize=14, fontweight='bold', color='#9370DB')
            self.canvas.draw()

            self.btn_fit.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_select_bg.config(state=tk.NORMAL)
            self.btn_clear_bg.config(state=tk.NORMAL)
            self.btn_auto_find.config(state=tk.NORMAL)
            self.btn_overlap_mode.config(state=tk.NORMAL)
            self.btn_apply_smooth.config(state=tk.NORMAL)
            self.btn_reset_smooth.config(state=tk.NORMAL)

            self.status_label.config(text=f"Loaded: {self.filename}")
            self.update_info(f"File loaded: {self.filename}\nData points: {len(self.x)}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax or self.x is None:
            return

        if self.toolbar.mode != '':
            return

        x_click = event.xdata
        idx = np.argmin(np.abs(self.x - x_click))
        point_x = self.x[idx]
        point_y = self.y[idx]

        if self.selecting_bg:
            marker, = self.ax.plot(point_x, point_y, 's', color='#4169E1',
                                  markersize=6, markeredgecolor='#FFD700',
                                  markeredgewidth=1, zorder=10)
            self.bg_points.append((point_x, point_y))
            self.bg_markers.append(marker)
            self.update_bg_connect_line()
            self.canvas.draw()

            self.undo_stack.append(('bg_point', len(self.bg_points) - 1))
            self.btn_undo.config(state=tk.NORMAL)
            self.update_info(f"BG point {len(self.bg_points)} at 2theta = {point_x:.4f}\n")

            if len(self.bg_points) >= 2:
                self.btn_subtract_bg.config(state=tk.NORMAL)
        elif not self.fitted:
            marker, = self.ax.plot(point_x, point_y, '*', color='#FF1493',
                                  markersize=15, markeredgecolor='#FFD700',
                                  markeredgewidth=1.5, zorder=10)
            text = self.ax.text(point_x, point_y * 1.03, f'P{len(self.selected_peaks)+1}',
                               ha='center', fontsize=8, color='#FF1493',
                               fontweight='bold', zorder=11)

            self.selected_peaks.append(idx)
            self.peak_markers.append(marker)
            self.peak_texts.append(text)
            self.canvas.draw()

            self.undo_stack.append(('peak', len(self.selected_peaks) - 1))
            self.btn_undo.config(state=tk.NORMAL)
            self.update_info(f"Peak {len(self.selected_peaks)} at 2theta = {point_x:.4f}\n")
            self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def toggle_bg_selection(self):
        """Toggle background selection mode"""
        self.selecting_bg = not self.selecting_bg
        if self.selecting_bg:
            self.btn_select_bg.config(bg='#FFD700', fg='#000000', text="Stop Selection")
            self.status_label.config(text="Selecting background points...")
        else:
            self.btn_select_bg.config(bg='#B0A0D0', fg='#2F0060', text="Select BG Points")
            self.status_label.config(text=f"{len(self.bg_points)} BG points selected")

    def update_bg_connect_line(self):
        """Update background connecting line"""
        if self.bg_connect_line is not None:
            try:
                self.bg_connect_line.remove()
            except:
                pass
            self.bg_connect_line = None

        if len(self.bg_points) >= 2:
            sorted_points = sorted(self.bg_points, key=lambda p: p[0])
            bg_x = [p[0] for p in sorted_points]
            bg_y = [p[1] for p in sorted_points]
            self.bg_connect_line, = self.ax.plot(bg_x, bg_y, '-', color='#4169E1',
                                                 linewidth=1.5, alpha=0.7, zorder=8)

    def undo_action(self):
        """Undo last action"""
        if not self.undo_stack:
            return

        action_type, index = self.undo_stack.pop()

        if action_type == 'peak':
            if self.selected_peaks and index == len(self.selected_peaks) - 1:
                self.selected_peaks.pop()
                marker = self.peak_markers.pop()
                text = self.peak_texts.pop()
                try:
                    marker.remove()
                    text.remove()
                except:
                    pass
                self.canvas.draw()
                self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

        elif action_type == 'bg_point':
            if self.bg_points and index == len(self.bg_points) - 1:
                self.bg_points.pop()
                marker = self.bg_markers.pop()
                try:
                    marker.remove()
                except:
                    pass
                self.update_bg_connect_line()
                self.canvas.draw()

                if len(self.bg_points) < 2:
                    self.btn_subtract_bg.config(state=tk.DISABLED)

        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

    def subtract_background(self):
        """Subtract background"""
        if len(self.bg_points) < 2:
            messagebox.showwarning("Insufficient Points", "Please select at least 2 background points!")
            return

        try:
            sorted_points = sorted(self.bg_points, key=lambda p: p[0])
            bg_x = np.array([p[0] for p in sorted_points])
            bg_y = np.array([p[1] for p in sorted_points])

            bg_interp = np.interp(self.x, bg_x, bg_y)
            self.y = self.y_original - bg_interp

            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8,
                        label='Data (BG subtracted)')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename} (BG Subtracted)',
                            fontsize=14, fontweight='bold', color='#9370DB')

            # Re-add peak markers
            for i, idx in enumerate(self.selected_peaks):
                marker, = self.ax.plot(self.x[idx], self.y[idx], '*', color='#FF1493',
                                      markersize=15, markeredgecolor='#FFD700',
                                      markeredgewidth=1.5, zorder=10)
                text = self.ax.text(self.x[idx], self.y[idx] * 1.03, f'P{i+1}',
                                   ha='center', fontsize=8, color='#FF1493',
                                   fontweight='bold', zorder=11)
                self.peak_markers[i] = marker
                self.peak_texts[i] = text

            self.canvas.draw()

            self.bg_points = []
            self.bg_markers = []
            self.bg_line = None
            self.bg_connect_line = None
            self.btn_subtract_bg.config(state=tk.DISABLED)

            self.update_info("Background subtracted\n")
            self.status_label.config(text="Background subtracted")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract background:\n{str(e)}")

    def clear_background(self):
        """Clear background selection"""
        for marker in self.bg_markers:
            try:
                marker.remove()
            except:
                pass

        if self.bg_line is not None:
            try:
                self.bg_line.remove()
            except:
                pass

        if self.bg_connect_line is not None:
            try:
                self.bg_connect_line.remove()
            except:
                pass

        self.bg_points = []
        self.bg_markers = []
        self.bg_line = None
        self.bg_connect_line = None
        self.selecting_bg = False

        self.undo_stack = [item for item in self.undo_stack if item[0] != 'bg_point']
        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

        self.btn_select_bg.config(bg='#B0A0D0', fg='#2F0060', text="Select BG Points")
        self.btn_subtract_bg.config(state=tk.DISABLED)

        if self.x is not None:
            self.canvas.draw()

    def auto_find_peaks(self):
        """
        Automatically find all peaks in the data using scipy.signal.find_peaks
        with prominence and height filtering
        """
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        # Clear existing peaks first
        self.reset_peaks()

        try:
            # Smooth data for better peak detection
            if len(self.y) > 15:
                window_length = min(15, len(self.y) // 2 * 2 + 1)
                y_smooth = savgol_filter(self.y, window_length, 3)
            else:
                y_smooth = self.y

            # Calculate data statistics for adaptive thresholds
            y_range = np.max(self.y) - np.min(self.y)
            y_std = np.std(self.y)
            dx = np.mean(np.diff(self.x))

            # Adaptive parameters based on data characteristics
            # Minimum height: above noise level
            height_threshold = np.min(self.y) + y_range * 0.05

            # Prominence: peak must stand out from surroundings
            prominence_threshold = y_range * 0.02

            # Minimum distance between peaks (in data points)
            # Estimate based on typical XRD peak width
            min_distance = max(5, int(0.1 / dx)) if dx > 0 else 5

            # Find peaks with multiple criteria
            peaks, properties = find_peaks(
                y_smooth,
                height=height_threshold,
                prominence=prominence_threshold,
                distance=min_distance,
                width=2  # Minimum width in data points
            )

            if len(peaks) == 0:
                # Try with less strict parameters
                peaks, properties = find_peaks(
                    y_smooth,
                    height=np.min(self.y) + y_range * 0.02,
                    prominence=y_range * 0.01,
                    distance=3
                )

            if len(peaks) == 0:
                messagebox.showinfo("No Peaks Found",
                    "No peaks detected automatically.\n"
                    "Try manual selection or adjust your data.")
                return

            # Additional filtering: check if peak is significantly above local baseline
            filtered_peaks = []
            for idx in peaks:
                # Local window
                window = 40
                left = max(0, idx - window)
                right = min(len(self.y), idx + window)

                # Local baseline from edges
                edge_n = max(3, (right - left) // 10)
                local_baseline = (np.mean(self.y[left:left+edge_n]) +
                                 np.mean(self.y[right-edge_n:right])) / 2

                # Check if peak is at least 10% above local baseline
                if self.y[idx] > local_baseline * 1.1:
                    filtered_peaks.append(idx)

            peaks = filtered_peaks

            if len(peaks) == 0:
                messagebox.showinfo("No Peaks Found",
                    "No significant peaks detected.\n"
                    "Try manual selection.")
                return

            # Add peaks to selection
            for idx in peaks:
                point_x = self.x[idx]
                point_y = self.y[idx]

                marker, = self.ax.plot(point_x, point_y, '*', color='#FF1493',
                                      markersize=15, markeredgecolor='#FFD700',
                                      markeredgewidth=1.5, zorder=10)
                text = self.ax.text(point_x, point_y * 1.03, f'P{len(self.selected_peaks)+1}',
                                   ha='center', fontsize=8, color='#FF1493',
                                   fontweight='bold', zorder=11)

                self.selected_peaks.append(idx)
                self.peak_markers.append(marker)
                self.peak_texts.append(text)

            self.canvas.draw()

            # Group peaks for display
            groups = self._group_peaks_for_display()

            self.update_info(f"Auto-detected {len(peaks)} peaks\n")
            if groups:
                self.update_info(f"Peak groups: {groups}\n")
            self.status_label.config(text=f"{len(peaks)} peaks auto-detected")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Auto peak detection failed:\n{str(e)}")
            self.update_info(f"Auto detection error: {traceback.format_exc()}\n")

    def _group_peaks_for_display(self):
        """
        Group peaks and return grouping info for display
        """
        if len(self.selected_peaks) < 2:
            return None

        # Sort peaks by position
        sorted_indices = sorted(range(len(self.selected_peaks)),
                               key=lambda i: self.x[self.selected_peaks[i]])
        sorted_peaks = [self.selected_peaks[i] for i in sorted_indices]

        # Estimate FWHM for each peak
        fwhm_estimates = []
        for idx in sorted_peaks:
            window_size = 50
            left = max(0, idx - window_size)
            right = min(len(self.x), idx + window_size)
            x_local = self.x[left:right]
            y_local = self.y[left:right]
            local_peak_idx = idx - left
            fwhm, _ = estimate_fwhm_robust(x_local, y_local, local_peak_idx)
            fwhm_estimates.append(fwhm)

        # Group overlapping peaks
        peak_groups = []
        current_group = [sorted_indices[0] + 1]  # 1-indexed for display

        for i in range(1, len(sorted_peaks)):
            prev_idx = sorted_peaks[i-1]
            curr_idx = sorted_peaks[i]
            distance = abs(self.x[curr_idx] - self.x[prev_idx])
            avg_fwhm = (fwhm_estimates[i-1] + fwhm_estimates[i]) / 2

            if distance < avg_fwhm * self.group_distance_threshold:
                current_group.append(sorted_indices[i] + 1)
            else:
                if len(current_group) > 1:
                    peak_groups.append(current_group)
                current_group = [sorted_indices[i] + 1]

        if len(current_group) > 1:
            peak_groups.append(current_group)

        return peak_groups if peak_groups else None

    def on_smoothing_changed(self):
        """Called when smoothing settings change"""
        pass  # Placeholder for future use

    def apply_smoothing_to_data(self):
        """Apply smoothing to the current data"""
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        try:
            method = self.smoothing_method.get()
            sigma = self.smoothing_sigma.get()
            window = self.smoothing_window.get()

            if method == 'gaussian':
                self.y_smoothed = apply_smoothing(self.y_original, method='gaussian', sigma=sigma)
                self.update_info(f"Applied Gaussian smoothing (sigma={sigma})\n")
            else:
                self.y_smoothed = apply_smoothing(self.y_original, method='savgol', window_length=window)
                self.update_info(f"Applied Savitzky-Golay smoothing (window={window})\n")

            # Update the working data
            self.y = self.y_smoothed.copy()

            # Redraw plot
            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8, label='Smoothed Data')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename} (Smoothed)\nClick on peaks to select',
                            fontsize=14, fontweight='bold', color='#9370DB')

            # Re-add peak markers if any
            for i, idx in enumerate(self.selected_peaks):
                marker, = self.ax.plot(self.x[idx], self.y[idx], '*', color='#FF1493',
                                      markersize=15, markeredgecolor='#FFD700',
                                      markeredgewidth=1.5, zorder=10)
                text = self.ax.text(self.x[idx], self.y[idx] * 1.03, f'P{i+1}',
                                   ha='center', fontsize=8, color='#FF1493',
                                   fontweight='bold', zorder=11)
                self.peak_markers[i] = marker
                self.peak_texts[i] = text

            self.canvas.draw()
            self.status_label.config(text="Smoothing applied")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply smoothing:\n{str(e)}")

    def reset_to_original_data(self):
        """Reset data to original (unsmoothed)"""
        if self.x is None or self.y_original is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        self.y = self.y_original.copy()
        self.y_smoothed = None

        # Redraw plot
        self.ax.clear()
        self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8, label='Data')
        self.ax.set_facecolor('#FAF0FF')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
        self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_title(f'{self.filename}\nClick on peaks to select',
                        fontsize=14, fontweight='bold', color='#9370DB')

        # Re-add peak markers if any
        for i, idx in enumerate(self.selected_peaks):
            marker, = self.ax.plot(self.x[idx], self.y[idx], '*', color='#FF1493',
                                  markersize=15, markeredgecolor='#FFD700',
                                  markeredgewidth=1.5, zorder=10)
            text = self.ax.text(self.x[idx], self.y[idx] * 1.03, f'P{i+1}',
                               ha='center', fontsize=8, color='#FF1493',
                               fontweight='bold', zorder=11)
            self.peak_markers[i] = marker
            self.peak_texts[i] = text

        self.canvas.draw()
        self.update_info("Data reset to original\n")
        self.status_label.config(text="Data reset")

    def toggle_overlap_mode(self):
        """Toggle overlap mode for better handling of overlapping peaks"""
        self.overlap_mode = not self.overlap_mode
        if self.overlap_mode:
            self.btn_overlap_mode.config(bg='#32CD32', text="Overlap ON")
            # Use smaller threshold for grouping when overlap mode is on
            self.group_distance_threshold = 3.5
            self.update_info("Overlap mode ON: Peaks within 3.5*FWHM will be grouped together\n")
        else:
            self.btn_overlap_mode.config(bg='#FF6B9D', text="Overlap")
            self.group_distance_threshold = 2.5
            self.update_info("Overlap mode OFF: Standard grouping (2.5*FWHM)\n")

    def fit_peaks(self):
        """
        Optimized peak fitting - fits each group separately with piecewise linear background
        based on minima between adjacent peaks
        """
        if len(self.selected_peaks) == 0:
            messagebox.showwarning("No Peaks", "Please select at least one peak first!")
            return

        fit_method = self.fit_method.get()
        self.update_info(f"Fitting {len(self.selected_peaks)} peaks using {fit_method}...\n")
        self.status_label.config(text="Fitting in progress...")
        self.master.update()

        try:
            dx = np.mean(np.diff(self.x))

            # Sort peaks by position
            sorted_indices = sorted(range(len(self.selected_peaks)),
                                   key=lambda i: self.x[self.selected_peaks[i]])
            sorted_peaks = [self.selected_peaks[i] for i in sorted_indices]

            # Step 1: Fit global background first
            self.update_info("Fitting global background...\n")
            # Use piecewise linear background (adjacent anchor points connected)
            # This provides maximum flexibility to follow local baseline variations
            bg_method = 'piecewise'

            global_bg, global_bg_points = fit_global_background(
                self.x, self.y, sorted_peaks,
                method=bg_method
            )

            # Subtract global background
            y_nobg = self.y - global_bg
            self.update_info(f"Piecewise linear background fitted "
                           f"with {len(global_bg_points)} anchor points (adjacent points connected)\n")

            # Step 2: Estimate FWHM for each peak (using background-subtracted data)
            fwhm_estimates = []
            baseline_estimates = []

            for idx in sorted_peaks:
                window_size = 50
                left = max(0, idx - window_size)
                right = min(len(self.x), idx + window_size)
                x_local = self.x[left:right]
                y_local = y_nobg[left:right]  # Use background-subtracted data
                local_peak_idx = idx - left
                fwhm, baseline = estimate_fwhm_robust(x_local, y_local, local_peak_idx)
                fwhm_estimates.append(fwhm)
                baseline_estimates.append(baseline)

            # Step 3: Group peaks using DBSCAN clustering
            peak_positions = np.array([self.x[idx] for idx in sorted_peaks])

            # Calculate eps based on average FWHM
            avg_fwhm = np.mean(fwhm_estimates)
            eps = avg_fwhm * self.group_distance_threshold

            cluster_labels, n_clusters = cluster_peaks_dbscan(peak_positions, eps=eps)

            # Convert cluster labels to peak groups
            peak_groups = []
            for cluster_id in range(max(cluster_labels) + 1):
                group = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if group:
                    peak_groups.append(group)

            # Sort groups by position
            peak_groups.sort(key=lambda g: self.x[sorted_peaks[g[0]]])

            self.update_info(f"DBSCAN clustering: {n_clusters} groups (eps={eps:.4f})\n")

            # Report grouped peaks
            for group in peak_groups:
                if len(group) > 1:
                    original_nums = [sorted_indices[g] + 1 for g in group]
                    self.update_info(f"Peaks {original_nums} grouped by DBSCAN clustering\n")

            use_voigt = (fit_method == "voigt")
            n_params_per_peak = 4 if use_voigt else 5

            # Store all results
            all_popt = {}  # peak_index -> parameters
            group_windows = []

            # Store global background info for plotting
            global_bg_info = {
                'background': global_bg,
                'bg_points': global_bg_points
            }

            # Step 4: Fit each group separately (using background-subtracted data)
            for g_idx, group in enumerate(peak_groups):
                self.status_label.config(text=f"Fitting group {g_idx+1}/{len(peak_groups)}...")
                self.master.update()

                group_peak_indices = [sorted_peaks[i] for i in group]
                group_fwhms = [fwhm_estimates[i] for i in group]
                is_overlapping = len(group) > 1

                # Create fitting window for this group - limited to peak region
                if self.overlap_mode:
                    window_multiplier = 4 if is_overlapping else 2.5
                else:
                    window_multiplier = 3 if is_overlapping else 2

                left_center = self.x[min(group_peak_indices)]
                right_center = self.x[max(group_peak_indices)]
                left_fwhm = group_fwhms[0]
                right_fwhm = group_fwhms[-1]

                window_left = left_center - left_fwhm * window_multiplier
                window_right = right_center + right_fwhm * window_multiplier

                left_idx = max(0, np.searchsorted(self.x, window_left))
                right_idx = min(len(self.x), np.searchsorted(self.x, window_right))
                group_windows.append((left_idx, right_idx))

                x_fit = self.x[left_idx:right_idx]
                # Use global background-subtracted data
                y_fit_nobg = y_nobg[left_idx:right_idx]

                if len(x_fit) < 5:
                    continue

                # Display info about fitting window
                if len(group) == 1:
                    self.update_info(f"Group {g_idx+1}: Peak {sorted_indices[group[0]]+1}, "
                                   f"window [{window_left:.2f}, {window_right:.2f}]\n")
                else:
                    peak_nums = [sorted_indices[g]+1 for g in group]
                    self.update_info(f"Group {g_idx+1}: Peaks {peak_nums}, "
                                   f"window [{window_left:.2f}, {window_right:.2f}]\n")

                # Build parameters for this group
                p0 = []
                bounds_lower = []
                bounds_upper = []

                for i in group:
                    idx = sorted_peaks[i]
                    local_idx = idx - left_idx
                    cen_guess = self.x[idx]
                    fwhm_est = fwhm_estimates[i]

                    sig_guess = fwhm_est / 2.355
                    gam_guess = fwhm_est / 2

                    # Amplitude estimation using background-subtracted data
                    peak_height = y_fit_nobg[local_idx] if local_idx < len(y_fit_nobg) else np.max(y_fit_nobg)
                    if peak_height <= 0:
                        peak_height = np.max(y_fit_nobg) * 0.5

                    amp_guess = peak_height * sig_guess * np.sqrt(2 * np.pi)

                    y_range = np.max(y_fit_nobg) - np.min(y_fit_nobg)
                    amp_lower = 0
                    amp_multiplier = 10 if (is_overlapping or self.overlap_mode) else 5
                    amp_upper = y_range * sig_guess * np.sqrt(2 * np.pi) * amp_multiplier

                    # Center constraints
                    if is_overlapping or self.overlap_mode:
                        center_tolerance = fwhm_est * 0.8
                    else:
                        center_tolerance = fwhm_est * 0.5

                    sig_lower = dx * 0.5
                    sig_upper = fwhm_est * 3
                    gam_lower = dx * 0.5
                    gam_upper = fwhm_est * 3

                    if use_voigt:
                        p0.extend([amp_guess, cen_guess, sig_guess, gam_guess])
                        bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower])
                        bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper])
                    else:
                        p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, 0.5])
                        bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower, 0])
                        bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper, 1.0])

                # Define fitting function (no background parameters since we pre-subtract)
                n_group_peaks = len(group)
                if use_voigt:
                    def make_func(n_peaks):
                        def func(x, *params):
                            y = np.zeros_like(x)
                            for i in range(n_peaks):
                                offset = i * 4
                                amp, cen, sig, gam = params[offset:offset+4]
                                y += voigt(x, amp, cen, sig, gam)
                            return y
                        return func
                else:
                    def make_func(n_peaks):
                        def func(x, *params):
                            y = np.zeros_like(x)
                            for i in range(n_peaks):
                                offset = i * 5
                                amp, cen, sig, gam, eta = params[offset:offset+5]
                                y += pseudo_voigt(x, amp, cen, sig, gam, eta)
                            return y
                        return func

                multi_peak_func = make_func(n_group_peaks)

                # Perform fitting
                if is_overlapping or self.overlap_mode:
                    max_iter = 30000
                    ftol = 1e-9
                    xtol = 1e-9
                else:
                    max_iter = 10000
                    ftol = 1e-8
                    xtol = 1e-8

                try:
                    popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                                       p0=p0, bounds=(bounds_lower, bounds_upper),
                                       method='trf', maxfev=max_iter,
                                       ftol=ftol, xtol=xtol)
                except Exception:
                    try:
                        popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                                           p0=p0, bounds=(bounds_lower, bounds_upper),
                                           method='dogbox', maxfev=50000)
                    except Exception as e:
                        self.update_info(f"Group {g_idx+1} fit failed: {str(e)}\n")
                        continue

                # Store results for each peak in this group
                for j, i in enumerate(group):
                    offset = j * n_params_per_peak
                    all_popt[i] = {
                        'params': popt[offset:offset+n_params_per_peak],
                        'group_idx': g_idx,
                        'window': (left_idx, right_idx)
                    }

            # Step 5: Plot results
            colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_peaks)))

            # Plot global background
            if len(global_bg_points) >= 2:
                bg_x = [p[0] for p in global_bg_points]
                bg_y = [p[1] for p in global_bg_points]
                # Plot background anchor points
                bg_markers, = self.ax.plot(bg_x, bg_y, 'o', color='#4169E1',
                                          markersize=6, alpha=0.8, zorder=3)
                self.fit_lines.append(bg_markers)
                # Plot smooth background line
                bg_line, = self.ax.plot(self.x, global_bg, '-', color='#4169E1',
                                       linewidth=1.5, alpha=0.6,
                                       label='Global Background', zorder=3)
                self.fit_lines.append(bg_line)

            # Plot total fit for each group
            for g_idx, (left, right) in enumerate(group_windows):
                x_region = self.x[left:right]
                x_smooth = np.linspace(x_region.min(), x_region.max(), 400)

                # Get global background for this region
                bg_smooth = np.interp(x_smooth, self.x, global_bg)

                # Sum all peaks in this group
                y_total = bg_smooth.copy()
                group = peak_groups[g_idx]

                for i in group:
                    if i not in all_popt:
                        continue
                    params = all_popt[i]['params']
                    if use_voigt:
                        y_total += voigt(x_smooth, *params)
                    else:
                        y_total += pseudo_voigt(x_smooth, *params)

                if g_idx == 0:
                    line1, = self.ax.plot(x_smooth, y_total, color='#FF0000', linewidth=1.5,
                                        label='Total Fit', zorder=5, alpha=0.9)
                else:
                    line1, = self.ax.plot(x_smooth, y_total, color='#FF0000', linewidth=1.5,
                                        zorder=5, alpha=0.9)
                self.fit_lines.append(line1)

            # Plot individual peak components
            for i in range(len(sorted_peaks)):
                if i not in all_popt:
                    continue

                params = all_popt[i]['params']
                left, right = all_popt[i]['window']

                x_smooth = np.linspace(self.x[left], self.x[right], 400)

                if use_voigt:
                    y_component = voigt(x_smooth, *params)
                else:
                    y_component = pseudo_voigt(x_smooth, *params)

                # Add global background
                bg_smooth = np.interp(x_smooth, self.x, global_bg)
                y_with_bg = y_component + bg_smooth

                original_idx = sorted_indices[i]
                line_comp, = self.ax.plot(x_smooth, y_with_bg, '--',
                                         color=colors[i], linewidth=1.2, alpha=0.7, zorder=4,
                                         label=f'Peak {original_idx+1}')
                self.fit_lines.append(line_comp)

            # Step 5: Extract results
            results = []
            info_msg = f"Fitting Results ({fit_method}):\n" + "="*50 + "\n"

            for i in range(len(sorted_peaks)):
                original_idx = sorted_indices[i]

                if i not in all_popt:
                    continue

                params = all_popt[i]['params']

                if use_voigt:
                    amp, cen, sig, gam = params
                    fwhm = 2.355 * sig
                    area = amp
                    eta = "N/A"
                else:
                    amp, cen, sig, gam, eta = params
                    fwhm = calculate_fwhm(sig, gam, eta)
                    area = calculate_area(amp, sig, gam, eta)

                results.append({
                    'Peak': original_idx + 1,
                    'Center_2theta': cen,
                    'FWHM': fwhm,
                    'Area': area,
                    'Amplitude': amp,
                    'Sigma': sig,
                    'Gamma': gam,
                    'Eta': eta
                })

                info_msg += f"Peak {original_idx+1}: 2theta={cen:.4f}, FWHM={fwhm:.5f}, Area={area:.1f}\n"

            results.sort(key=lambda r: r['Peak'])

            self.fit_results = pd.DataFrame(results)
            self.fitted = True

            # Update results table
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            for r in results:
                eta_str = f"{r['Eta']:.3f}" if isinstance(r['Eta'], float) else r['Eta']
                self.results_tree.insert('', 'end', values=(
                    f"{r['Peak']}",
                    f"{r['Center_2theta']:.4f}",
                    f"{r['FWHM']:.5f}",
                    f"{r['Area']:.2f}",
                    f"{r['Amplitude']:.2f}",
                    f"{r['Sigma']:.5f}",
                    f"{r['Gamma']:.5f}",
                    eta_str
                ))

            self.ax.set_title(f'{self.filename} - Fit Complete ({fit_method})',
                            fontsize=14, fontweight='bold', color='#32CD32')
            self.canvas.draw()

            self.update_info(info_msg)
            self.status_label.config(text="Fitting successful!")

            self.btn_save.config(state=tk.NORMAL)
            self.btn_clear_fit.config(state=tk.NORMAL)

        except Exception as e:
            import traceback
            messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
            self.update_info(f"Fitting failed: {traceback.format_exc()}\n")
            self.status_label.config(text="Fitting failed")

    def clear_fit(self):
        """Clear fitting results"""
        for line in self.fit_lines:
            try:
                line.remove()
            except:
                pass
        self.fit_lines = []

        self.fitted = False
        self.fit_results = None

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.ax.set_title(f'{self.filename}\nClick on peaks to select',
                         fontsize=14, fontweight='bold', color='#9370DB')
        self.canvas.draw()

        self.btn_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)
        self.update_info("Fit cleared. Peak selections preserved.\n")
        self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def reset_peaks(self):
        """Clear all peaks and fits"""
        for marker in self.peak_markers:
            try:
                marker.remove()
            except:
                pass
        for text in self.peak_texts:
            try:
                text.remove()
            except:
                pass
        for line in self.fit_lines:
            try:
                line.remove()
            except:
                pass

        self.selected_peaks = []
        self.peak_markers = []
        self.peak_texts = []
        self.fit_lines = []
        self.fitted = False
        self.fit_results = None

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.undo_stack = [item for item in self.undo_stack if item[0] != 'peak']
        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

        if self.x is not None:
            self.ax.set_title(f'{self.filename}\nClick on peaks to select',
                            fontsize=14, fontweight='bold', color='#9370DB')
            self.canvas.draw()
            self.update_info("All peaks and fits cleared\n")
            self.status_label.config(text="Ready to select peaks")

        self.btn_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)

    def save_results(self):
        """Save fitting results"""
        if self.fit_results is None:
            messagebox.showwarning("No Results", "Please fit peaks before saving!")
            return

        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            self.fit_results['File'] = self.filename
            csv_path = os.path.join(save_dir, f"{self.filename}_fit_results.csv")
            self.fit_results.to_csv(csv_path, index=False)

            fig_path = os.path.join(save_dir, f"{self.filename}_fit_plot.png")
            self.fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')

            messagebox.showinfo("Success",
                              f"Results saved!\n\nCSV: {csv_path}\nPlot: {fig_path}")
            self.update_info(f"Results saved to: {save_dir}\n")
            self.status_label.config(text="Results saved!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")

    def update_info(self, message):
        """Update info text"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = PeakFittingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
