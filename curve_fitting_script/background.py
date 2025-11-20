# -*- coding: utf-8 -*-
"""
Background Fitting Module

Provides various methods for background estimation and subtraction in XRD data.

@author: candicewang928@gmail.com
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def fit_global_background(x, y, peak_indices, method='spline', smoothing_factor=None, poly_order=3):
    """
    Fit a global smooth background to the data, excluding peak regions.

    Parameters
    ----------
    x : array_like
        X data (2theta values)
    y : array_like
        Y data (intensity values)
    peak_indices : array_like
        Indices of detected peaks
    method : {'spline', 'piecewise', 'polynomial'}, optional
        Background fitting method. Default is 'spline'.
        - 'spline': Smooth spline interpolation
        - 'piecewise': Piecewise linear (adjacent points connected)
        - 'polynomial': Polynomial fit with bounded curvature (smoothest)
    smoothing_factor : float, optional
        Smoothing factor for spline (larger = smoother). If None, auto-determined.
    poly_order : int, optional
        Order of polynomial for 'polynomial' method (default: 3)
        - 2: parabola (constant 2nd derivative)
        - 3: cubic (linear 2nd derivative)

    Returns
    -------
    background : ndarray
        Background values at each x point
    bg_points : list of tuples
        (x, y) coordinates of background anchor points

    Notes
    -----
    The function automatically identifies local minima between peaks and at edges
    to use as background anchor points.
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
        try:
            # Ensure poly_order is not too high for number of points
            max_order = min(poly_order, len(bg_x) - 1, 5)  # Cap at 5th order

            # Fit polynomial
            coeffs = np.polyfit(bg_x, bg_y, max_order)
            poly = np.poly1d(coeffs)
            background = poly(x)

            # Ensure background doesn't go above maximum data value
            background = np.clip(background, None, np.max(y))

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


def find_background_points_auto(x, y, n_points=10, window_size=50):
    """
    Automatically find background anchor points across the data range.

    Parameters
    ----------
    x : array_like
        X data
    y : array_like
        Y data
    n_points : int, optional
        Target number of background points to find. Default is 10.
    window_size : int, optional
        Size of local window for finding minima. Default is 50.

    Returns
    -------
    bg_points : list of tuples
        List of (x, y) coordinates for background points

    Notes
    -----
    The function divides the x-range into segments and finds the local
    minimum in each segment.
    """
    if len(x) < 2:
        return []

    # Divide the x range into segments
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min

    # Calculate segment boundaries
    segment_boundaries = np.linspace(x_min, x_max, n_points + 1)

    bg_points = []

    for i in range(n_points):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]

        # Find indices in this segment
        mask = (x >= seg_start) & (x <= seg_end)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # Find local minimum in this segment
        seg_y = y[indices]

        # Use a smaller local window to find the true minimum
        local_window = min(window_size, len(seg_y) // 2)

        if local_window >= 3:
            # Smooth to avoid noise
            try:
                seg_y_smooth = savgol_filter(seg_y, min(local_window, len(seg_y)//2*2+1), 2)
            except:
                seg_y_smooth = seg_y
        else:
            seg_y_smooth = seg_y

        # Find minimum
        min_local_idx = np.argmin(seg_y_smooth)
        global_idx = indices[min_local_idx]

        bg_points.append((x[global_idx], y[global_idx]))

    return bg_points


def find_group_minima(x, y, peak_indices):
    """
    Find local minima between adjacent peaks in a group.

    This function is used for group-based background fitting where each peak
    group has its own piecewise linear background.

    Parameters
    ----------
    x : array_like
        X data (2theta values)
    y : array_like
        Y data (intensity values)
    peak_indices : array_like
        List of peak indices in sorted order

    Returns
    -------
    minima_points : list of tuples
        List of (x, y) coordinates for background interpolation points,
        including edges and minima between peaks

    Notes
    -----
    For a single peak, the function finds minima on both sides.
    For multiple peaks, it finds minima between each adjacent pair.
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

    Parameters
    ----------
    x_data : array_like
        X coordinates where background values are needed
    minima_points : list of tuples
        List of (x, y) coordinates for background interpolation

    Returns
    -------
    background : ndarray
        Background values at each x_data point

    Notes
    -----
    If fewer than 2 points are provided, returns zeros.
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
