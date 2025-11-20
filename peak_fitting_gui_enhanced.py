# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - Enhanced Encapsulated Version
@author: candicewang928@gmail.com
Enhanced with better code organization and encapsulation
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

# Import icon utility
try:
    from icon_utils import set_window_icon
except ImportError:
    # Fallback if icon_utils not available
    def set_window_icon(window, icon_path=None, use_default=True):
        try:
            if icon_path and os.path.exists(icon_path):
                if icon_path.endswith('.ico'):
                    window.iconbitmap(icon_path)
                else:
                    icon_image = tk.PhotoImage(file=icon_path)
                    window.iconphoto(True, icon_image)
                    window._icon_image = icon_image
        except:
            pass

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# ==================== Data Processing Module ====================
class DataProcessor:
    """Handles data smoothing and preprocessing operations"""

    @staticmethod
    def gaussian_smoothing(y, sigma=2):
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

    @staticmethod
    def savgol_smoothing(y, window_length=11, polyorder=3):
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

    @classmethod
    def apply_smoothing(cls, y, method='gaussian', **kwargs):
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
            return cls.gaussian_smoothing(y, sigma=sigma)
        elif method == 'savgol':
            window_length = kwargs.get('window_length', 11)
            polyorder = kwargs.get('polyorder', 3)
            return cls.savgol_smoothing(y, window_length=window_length, polyorder=polyorder)
        else:
            return y


# ==================== Peak Clustering Module ====================
class PeakClusterer:
    """Handles peak grouping using DBSCAN clustering"""

    @staticmethod
    def cluster_peaks(peak_positions, eps=None, min_samples=1):
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
            sorted_pos = np.sort(peak_positions)
            if len(sorted_pos) > 1:
                distances = np.diff(sorted_pos)
                eps = np.median(distances) * 1.5
            else:
                eps = 1.0

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_

        # Handle noise points
        noise_mask = labels == -1
        if np.any(noise_mask) and np.any(~noise_mask):
            for i in np.where(noise_mask)[0]:
                non_noise_idx = np.where(~noise_mask)[0]
                distances = np.abs(peak_positions[non_noise_idx] - peak_positions[i])
                nearest = non_noise_idx[np.argmin(distances)]
                labels[i] = labels[nearest]
        elif np.all(noise_mask):
            labels = np.zeros(len(labels), dtype=int)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return labels, n_clusters


# ==================== Background Fitting Module ====================
class BackgroundFitter:
    """Handles background fitting operations"""

    @staticmethod
    def fit_global_background(x, y, peak_indices, method='spline',
                             smoothing_factor=None, poly_order=3):
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
            'spline', 'piecewise', or 'polynomial'
        smoothing_factor : float, optional
            Smoothing factor for spline
        poly_order : int, optional
            Order of polynomial for 'polynomial' method

        Returns:
        --------
        background : array
            Background values at each x point
        bg_points : list of tuples
            (x, y) coordinates of background anchor points
        """
        if len(peak_indices) == 0:
            return np.full_like(y, np.median(y)), []

        sorted_peaks = sorted(peak_indices)
        bg_x, bg_y = [], []

        # Left edge
        first_peak = sorted_peaks[0]
        left_region_end = max(0, first_peak - 5)
        if left_region_end > 0:
            left_min_idx = np.argmin(y[:left_region_end+1])
            bg_x.append(x[left_min_idx])
            bg_y.append(y[left_min_idx])
        else:
            bg_x.append(x[0])
            bg_y.append(y[0])

        # Between peaks
        for i in range(len(sorted_peaks) - 1):
            idx1 = sorted_peaks[i]
            idx2 = sorted_peaks[i + 1]
            if idx2 > idx1 + 1:
                between_region = y[idx1:idx2+1]
                min_local = np.argmin(between_region)
                min_idx = idx1 + min_local
                bg_x.append(x[min_idx])
                bg_y.append(y[min_idx])

        # Right edge
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

        # Sort and remove duplicates
        sort_idx = np.argsort(bg_x)
        bg_x = bg_x[sort_idx]
        bg_y = bg_y[sort_idx]

        unique_mask = np.concatenate([[True], np.diff(bg_x) > 0])
        bg_x = bg_x[unique_mask]
        bg_y = bg_y[unique_mask]

        bg_points = list(zip(bg_x, bg_y))

        if len(bg_x) < 2:
            return np.full_like(y, np.mean(bg_y) if len(bg_y) > 0 else np.median(y)), bg_points

        # Apply fitting method
        if method == 'polynomial':
            try:
                max_order = min(poly_order, len(bg_x) - 1, 5)
                coeffs = np.polyfit(bg_x, bg_y, max_order)
                poly = np.poly1d(coeffs)
                background = poly(x)
                background = np.clip(background, None, np.max(y))
            except Exception:
                background = np.interp(x, bg_x, bg_y)
        elif method == 'spline' and len(bg_x) >= 4:
            if smoothing_factor is None:
                smoothing_factor = len(bg_x) * 1.0
            try:
                spline = UnivariateSpline(bg_x, bg_y, s=smoothing_factor, k=3)
                background = spline(x)
                background = np.clip(background, None, np.max(y))
            except Exception:
                background = np.interp(x, bg_x, bg_y)
        else:
            background = np.interp(x, bg_x, bg_y)

        return background, bg_points

    @staticmethod
    def find_auto_background_points(x, y, n_points=10, window_size=50):
        """
        Automatically find background anchor points across the data range.

        Parameters:
        -----------
        x : array
            X data
        y : array
            Y data
        n_points : int
            Target number of background points to find
        window_size : int
            Size of local window for finding minima

        Returns:
        --------
        bg_points : list of tuples
            List of (x, y) coordinates for background points
        """
        if len(x) < 2:
            return []

        x_min, x_max = x.min(), x.max()
        segment_boundaries = np.linspace(x_min, x_max, n_points + 1)
        bg_points = []

        for i in range(n_points):
            seg_start = segment_boundaries[i]
            seg_end = segment_boundaries[i + 1]

            mask = (x >= seg_start) & (x <= seg_end)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            seg_y = y[indices]
            local_window = min(window_size, len(seg_y) // 2)

            if local_window >= 3:
                try:
                    seg_y_smooth = savgol_filter(seg_y, min(local_window, len(seg_y)//2*2+1), 2)
                except:
                    seg_y_smooth = seg_y
            else:
                seg_y_smooth = seg_y

            min_local_idx = np.argmin(seg_y_smooth)
            global_idx = indices[min_local_idx]
            bg_points.append((x[global_idx], y[global_idx]))

        return bg_points


# ==================== Peak Profile Functions ====================
class PeakProfile:
    """Peak profile mathematical functions"""

    @staticmethod
    def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
        """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian"""
        gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
        return eta * lorentzian + (1 - eta) * gaussian

    @staticmethod
    def voigt(x, amplitude, center, sigma, gamma):
        """Voigt profile using Faddeeva function"""
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def calculate_fwhm(sigma, gamma, eta):
        """Calculate FWHM from Pseudo-Voigt parameters"""
        fwhm_g = 2.355 * sigma
        fwhm_l = 2 * gamma
        return eta * fwhm_l + (1 - eta) * fwhm_g

    @staticmethod
    def calculate_area(amplitude, sigma, gamma, eta):
        """Calculate integrated area"""
        area_g = amplitude * sigma * np.sqrt(2 * np.pi)
        area_l = amplitude * np.pi * gamma
        return eta * area_l + (1 - eta) * area_g

    @staticmethod
    def estimate_fwhm(x, y, peak_idx, smooth=True):
        """
        Robust FWHM estimation using interpolation

        Parameters:
        -----------
        x : array
            X data
        y : array
            Y data
        peak_idx : int
            Index of peak position
        smooth : bool
            Whether to smooth data before estimation

        Returns:
        --------
        fwhm : float
            Full width at half maximum
        baseline : float
            Estimated baseline value
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
        left_x = x[0]
        for j in range(peak_idx, 0, -1):
            if y_smooth[j] <= half_max:
                if y_smooth[j+1] != y_smooth[j]:
                    frac = (half_max - y_smooth[j]) / (y_smooth[j+1] - y_smooth[j])
                    left_x = x[j] + frac * (x[j+1] - x[j])
                else:
                    left_x = x[j]
                break

        # Find right half-max point with interpolation
        right_x = x[-1]
        for j in range(peak_idx, len(y_smooth)-1):
            if y_smooth[j] <= half_max:
                if y_smooth[j-1] != y_smooth[j]:
                    frac = (half_max - y_smooth[j]) / (y_smooth[j-1] - y_smooth[j])
                    right_x = x[j] - frac * (x[j] - x[j-1])
                else:
                    right_x = x[j]
                break

        fwhm = abs(right_x - left_x)

        # Sanity check
        dx = np.mean(np.diff(x))
        if fwhm < dx * 2:
            fwhm = dx * 8

        return fwhm, baseline


# ==================== Peak Detector Module ====================
class PeakDetector:
    """Automatic peak detection"""

    @staticmethod
    def auto_find_peaks(x, y):
        """
        Automatically find all peaks in the data using scipy.signal.find_peaks

        Parameters:
        -----------
        x : array
            X data
        y : array
            Y data

        Returns:
        --------
        peaks : array
            Indices of detected peaks
        """
        # Smooth data for better peak detection
        if len(y) > 15:
            window_length = min(15, len(y) // 2 * 2 + 1)
            y_smooth = savgol_filter(y, window_length, 3)
        else:
            y_smooth = y

        # Calculate data statistics
        y_range = np.max(y) - np.min(y)
        dx = np.mean(np.diff(x))

        # Adaptive parameters
        height_threshold = np.min(y) + y_range * 0.05
        prominence_threshold = y_range * 0.02
        min_distance = max(5, int(0.1 / dx)) if dx > 0 else 5

        # Find peaks
        peaks, properties = find_peaks(
            y_smooth,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=min_distance,
            width=2
        )

        if len(peaks) == 0:
            # Try with less strict parameters
            peaks, properties = find_peaks(
                y_smooth,
                height=np.min(y) + y_range * 0.02,
                prominence=y_range * 0.01,
                distance=3
            )

        # Additional filtering
        filtered_peaks = []
        for idx in peaks:
            window = 40
            left = max(0, idx - window)
            right = min(len(y), idx + window)

            edge_n = max(3, (right - left) // 10)
            local_baseline = (np.mean(y[left:left+edge_n]) +
                             np.mean(y[right-edge_n:right])) / 2

            if y[idx] > local_baseline * 1.1:
                filtered_peaks.append(idx)

        return np.array(filtered_peaks)


# ==================== Main GUI Application ====================
class PeakFittingGUI:
    """Main GUI Application for Peak Fitting"""

    def __init__(self, master, icon_path=None):
        self.master = master
        self.master.title("Interactive XRD Peak Fitting Tool - Enhanced")
        self.master.geometry("1400x850")
        self.master.configure(bg='#F0E6FA')

        # Set window icon
        set_window_icon(self.master, icon_path=icon_path, use_default=True)

        # Data storage
        self.x = None
        self.y = None
        self.y_original = None
        self.filename = None
        self.filepath = None
        self.selected_peaks = []
        self.peak_markers = []
        self.peak_texts = []
        self.fitted = False
        self.fit_results = None
        self.fit_lines = []

        # File navigation
        self.file_list = []
        self.current_file_index = -1

        # Background fitting storage
        self.bg_points = []
        self.bg_markers = []
        self.bg_line = None
        self.bg_connect_line = None
        self.selecting_bg = False

        # Undo stack
        self.undo_stack = []

        # Fitting settings
        self.fit_method = tk.StringVar(value="pseudo_voigt")
        self.overlap_mode = False
        self.group_distance_threshold = 2.5

        # Smoothing settings
        self.smoothing_enabled = tk.BooleanVar(value=False)
        self.smoothing_method = tk.StringVar(value="gaussian")
        self.smoothing_sigma = tk.DoubleVar(value=2.0)
        self.smoothing_window = tk.IntVar(value=11)
        self.y_smoothed = None

        # Initialize GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI components"""
        # Top control panel
        self._create_control_panel()

        # Background fitting panel
        self._create_background_panel()

        # Smoothing panel
        self._create_smoothing_panel()

        # Results display panel
        self._create_results_panel()

        # Main plot area
        self._create_plot_area()

        # Info panel
        self._create_info_panel()

    def _create_control_panel(self):
        """Create top control panel"""
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

        # File navigation buttons
        nav_btn_style = {**btn_style, 'width': 3}

        self.btn_prev_file = tk.Button(control_frame, text="◀",
                                        bg='#9370DB', fg='white',
                                        command=self.prev_file,
                                        state=tk.DISABLED, **nav_btn_style)
        self.btn_prev_file.pack(side=tk.LEFT, padx=2, pady=8)

        self.btn_next_file = tk.Button(control_frame, text="▶",
                                        bg='#9370DB', fg='white',
                                        command=self.next_file,
                                        state=tk.DISABLED, **nav_btn_style)
        self.btn_next_file.pack(side=tk.LEFT, padx=2, pady=8)

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

        self.btn_quick_save = tk.Button(control_frame, text="Quick Save",
                                         bg='#98FB98', fg='#006400',
                                         command=self.quick_save_results, state=tk.DISABLED, **btn_style)
        self.btn_quick_save.pack(side=tk.LEFT, padx=5, pady=8)

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

    def _create_background_panel(self):
        """Create background fitting control panel"""
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

        self.btn_auto_bg = tk.Button(bg_frame, text="Auto Select BG",
                                     bg='#87CEEB', fg='#00008B',
                                     command=self.auto_select_background,
                                     state=tk.DISABLED, **btn_bg_style)
        self.btn_auto_bg.pack(side=tk.LEFT, padx=5, pady=8)

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

    def _create_smoothing_panel(self):
        """Create smoothing control panel"""
        smooth_frame = tk.Frame(self.master, bg='#D5E6F5', height=50)
        smooth_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        smooth_frame.pack_propagate(False)

        smooth_label = tk.Label(smooth_frame, text="Smoothing:",
                               bg='#D5E6F5', fg='#0047AB',
                               font=('Arial', 10, 'bold'))
        smooth_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.chk_smooth = tk.Checkbutton(smooth_frame, text="Enable",
                                         variable=self.smoothing_enabled,
                                         bg='#D5E6F5', fg='#0047AB',
                                         font=('Arial', 9, 'bold'))
        self.chk_smooth.pack(side=tk.LEFT, padx=5, pady=8)

        tk.Label(smooth_frame, text="Method:",
                bg='#D5E6F5', fg='#0047AB',
                font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2), pady=10)

        smooth_method_combo = ttk.Combobox(smooth_frame, textvariable=self.smoothing_method,
                                           values=["gaussian", "savgol"],
                                           state="readonly", width=8)
        smooth_method_combo.pack(side=tk.LEFT, padx=2, pady=8)

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

        self.btn_apply_smooth = tk.Button(smooth_frame, text="Apply",
                                          bg='#4682B4', fg='white',
                                          font=('Arial', 9, 'bold'),
                                          width=8, height=1,
                                          command=self.apply_smoothing_to_data,
                                          state=tk.DISABLED)
        self.btn_apply_smooth.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_reset_smooth = tk.Button(smooth_frame, text="Reset Data",
                                          bg='#CD5C5C', fg='white',
                                          font=('Arial', 9, 'bold'),
                                          width=10, height=1,
                                          command=self.reset_to_original_data,
                                          state=tk.DISABLED)
        self.btn_reset_smooth.pack(side=tk.LEFT, padx=5, pady=8)

    def _create_results_panel(self):
        """Create results display panel"""
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

        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL,
                                         command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        style = ttk.Style()
        style.configure('Treeview', background='#FAF0FF', foreground='#4B0082',
                       font=('Courier', 9))
        style.configure('Treeview.Heading', font=('Arial', 9, 'bold'),
                       foreground='#4B0082')

    def _create_plot_area(self):
        """Create main plot area"""
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

        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def _create_info_panel(self):
        """Create info panel"""
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

    # ==================== Event Handlers ====================

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax or self.x is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        scale_factor = 0.8 if event.button == 'up' else 1.25

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

    def on_click(self, event):
        """Handle mouse clicks - left click adds, right click removes"""
        if event.inaxes != self.ax or self.x is None:
            return

        if self.toolbar.mode != '':
            return

        x_click = event.xdata
        idx = np.argmin(np.abs(self.x - x_click))
        point_x = self.x[idx]
        point_y = self.y[idx]

        if self.selecting_bg:
            self._handle_bg_click(event, idx, point_x, point_y, x_click)
        elif not self.fitted:
            self._handle_peak_click(event, idx, point_x, point_y, x_click)

    def _handle_bg_click(self, event, idx, point_x, point_y, x_click):
        """Handle clicks in background selection mode"""
        if event.button == 1:  # Left click - add point
            marker, = self.ax.plot(point_x, point_y, 's', color='#4169E1',
                                  markersize=5, markeredgecolor='#FFD700',
                                  markeredgewidth=1.5, zorder=10)
            text = self.ax.text(point_x, point_y * 0.97, f'BG{len(self.bg_points)+1}',
                               ha='center', fontsize=7, color='#4169E1',
                               fontweight='bold', zorder=11)
            self.bg_points.append((point_x, point_y))
            self.bg_markers.append((marker, text))
            self.update_bg_connect_line()
            self.canvas.draw()

            self.undo_stack.append(('bg_point', len(self.bg_points) - 1))
            self.btn_undo.config(state=tk.NORMAL)
            self.update_info(f"BG point {len(self.bg_points)} added at 2theta = {point_x:.4f}\n")

            if len(self.bg_points) >= 2:
                self.btn_subtract_bg.config(state=tk.NORMAL)

        elif event.button == 3:  # Right click - remove point
            if len(self.bg_points) > 0:
                # Find nearest background point
                distances = [abs(x_click - p[0]) for p in self.bg_points]
                delete_idx = np.argmin(distances)

                removed_point = self.bg_points.pop(delete_idx)
                marker_tuple = self.bg_markers.pop(delete_idx)

                try:
                    marker_tuple[0].remove()
                    marker_tuple[1].remove()
                except:
                    pass

                # Update labels
                for i, (marker, text) in enumerate(self.bg_markers):
                    text.set_text(f'BG{i+1}')

                self.update_bg_connect_line()
                self.canvas.draw()

                self.update_info(f"BG point removed at 2theta = {removed_point[0]:.4f}\n")

                if len(self.bg_points) < 2:
                    self.btn_subtract_bg.config(state=tk.DISABLED)

    def _handle_peak_click(self, event, idx, point_x, point_y, x_click):
        """Handle clicks in peak selection mode"""
        if event.button == 1:  # Left click - add peak
            search_window = max(5, min(10, len(self.y) // 20))
            left_idx = max(0, idx - search_window)
            right_idx = min(len(self.y), idx + search_window + 1)

            local_y = self.y[left_idx:right_idx]
            local_max_idx = np.argmax(local_y)
            peak_idx = left_idx + local_max_idx

            dx = np.mean(np.diff(self.x))
            distance_to_click = abs(self.x[peak_idx] - x_click)
            max_allowed_distance = dx * search_window * 0.7

            if distance_to_click > max_allowed_distance:
                peak_idx = idx
                peak_x = point_x
                peak_y = point_y
                adjustment_note = "(using click position)"
            else:
                peak_x = self.x[peak_idx]
                peak_y = self.y[peak_idx]
                adjustment_note = "(auto-adjusted to local max)" if peak_idx != idx else ""

            marker, = self.ax.plot(peak_x, peak_y, '*', color='#FF1493',
                                  markersize=15, markeredgecolor='#FFD700',
                                  markeredgewidth=1.5, zorder=10)
            text = self.ax.text(peak_x, peak_y * 1.03, f'P{len(self.selected_peaks)+1}',
                               ha='center', fontsize=8, color='#FF1493',
                               fontweight='bold', zorder=11)

            self.selected_peaks.append(peak_idx)
            self.peak_markers.append(marker)
            self.peak_texts.append(text)
            self.canvas.draw()

            self.undo_stack.append(('peak', len(self.selected_peaks) - 1))
            self.btn_undo.config(state=tk.NORMAL)
            self.update_info(f"Peak {len(self.selected_peaks)} at 2theta = {peak_x:.4f} {adjustment_note}\n")
            self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

        elif event.button == 3:  # Right click - remove peak
            if len(self.selected_peaks) > 0:
                peak_positions = [self.x[peak_idx] for peak_idx in self.selected_peaks]
                distances = [abs(x_click - pos) for pos in peak_positions]
                nearest_idx = np.argmin(distances)

                removed_peak_idx = self.selected_peaks.pop(nearest_idx)
                removed_peak_x = self.x[removed_peak_idx]

                marker = self.peak_markers.pop(nearest_idx)
                text = self.peak_texts.pop(nearest_idx)
                try:
                    marker.remove()
                    text.remove()
                except:
                    pass

                for i, text_obj in enumerate(self.peak_texts):
                    text_obj.set_text(f'P{i+1}')

                self.canvas.draw()

                self.update_info(f"Peak removed at 2theta = {removed_peak_x:.4f}\n")
                self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    # ==================== File Operations ====================

    def load_file(self):
        """Load XRD data file via file dialog"""
        filepath = filedialog.askopenfilename(
            title="Select XRD Data File",
            filetypes=[("XY files", "*.xy"), ("DAT files", "*.dat"),
                       ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return

        self._scan_folder(filepath)
        self.load_file_by_path(filepath)

    def _scan_folder(self, filepath):
        """Scan folder and build file list for navigation"""
        folder = os.path.dirname(filepath)
        all_files = []

        extensions = ['.xy', '.dat', '.txt']
        for file in sorted(os.listdir(folder)):
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(folder, file)
                if os.path.isfile(full_path):
                    all_files.append(full_path)

        self.file_list = all_files
        try:
            self.current_file_index = self.file_list.index(filepath)
        except ValueError:
            self.current_file_index = 0

    def load_file_by_path(self, filepath):
        """Load XRD data file from specific path"""
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

            # Enable buttons
            self.btn_fit.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_select_bg.config(state=tk.NORMAL)
            self.btn_clear_bg.config(state=tk.NORMAL)
            self.btn_auto_bg.config(state=tk.NORMAL)
            self.btn_auto_find.config(state=tk.NORMAL)
            self.btn_overlap_mode.config(state=tk.NORMAL)
            self.btn_apply_smooth.config(state=tk.NORMAL)
            self.btn_reset_smooth.config(state=tk.NORMAL)

            if len(self.file_list) > 1:
                self.btn_prev_file.config(state=tk.NORMAL)
                self.btn_next_file.config(state=tk.NORMAL)

            file_info = f"File {self.current_file_index + 1}/{len(self.file_list)}: {self.filename}"
            self.status_label.config(text=file_info)
            self.update_info(f"File loaded: {self.filename}\nData points: {len(self.x)}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def prev_file(self):
        """Load previous file in the folder"""
        if len(self.file_list) == 0:
            return

        self.current_file_index = (self.current_file_index - 1) % len(self.file_list)
        filepath = self.file_list[self.current_file_index]
        self.load_file_by_path(filepath)

    def next_file(self):
        """Load next file in the folder"""
        if len(self.file_list) == 0:
            return

        self.current_file_index = (self.current_file_index + 1) % len(self.file_list)
        filepath = self.file_list[self.current_file_index]
        self.load_file_by_path(filepath)

    # ==================== Background Operations ====================

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

    def auto_select_background(self):
        """Automatically select background points"""
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        try:
            self.clear_background()

            bg_points = BackgroundFitter.find_auto_background_points(self.x, self.y, n_points=10)

            if len(bg_points) == 0:
                messagebox.showwarning("Auto Selection Failed",
                                      "Could not automatically find background points.")
                return

            for point_x, point_y in bg_points:
                marker, = self.ax.plot(point_x, point_y, 's', color='#4169E1',
                                      markersize=5, markeredgecolor='#FFD700',
                                      markeredgewidth=1.5, zorder=10)
                text = self.ax.text(point_x, point_y * 0.97, f'BG{len(self.bg_points)+1}',
                                   ha='center', fontsize=7, color='#4169E1',
                                   fontweight='bold', zorder=11)
                self.bg_points.append((point_x, point_y))
                self.bg_markers.append((marker, text))

            self.update_bg_connect_line()
            self.canvas.draw()

            if len(self.bg_points) >= 2:
                self.btn_subtract_bg.config(state=tk.NORMAL)

            self.update_info(f"Auto-selected {len(bg_points)} background points\n")
            self.status_label.config(text=f"{len(bg_points)} BG points auto-selected")

        except Exception as e:
            messagebox.showerror("Error", f"Auto background selection failed:\n{str(e)}")

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
        for marker_tuple in self.bg_markers:
            try:
                marker_tuple[0].remove()
                marker_tuple[1].remove()
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

    # ==================== Smoothing Operations ====================

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
                self.y_smoothed = DataProcessor.apply_smoothing(
                    self.y_original, method='gaussian', sigma=sigma)
                self.update_info(f"Applied Gaussian smoothing (sigma={sigma})\n")
            else:
                self.y_smoothed = DataProcessor.apply_smoothing(
                    self.y_original, method='savgol', window_length=window)
                self.update_info(f"Applied Savitzky-Golay smoothing (window={window})\n")

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
        self.update_info("Data reset to original\n")
        self.status_label.config(text="Data reset")

    # ==================== Peak Operations ====================

    def auto_find_peaks(self):
        """Automatically find all peaks in the data"""
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        self.reset_peaks()

        try:
            peaks = PeakDetector.auto_find_peaks(self.x, self.y)

            if len(peaks) == 0:
                messagebox.showinfo("No Peaks Found",
                    "No peaks detected automatically.\n"
                    "Try manual selection or adjust your data.")
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

            self.update_info(f"Auto-detected {len(peaks)} peaks\n")
            self.status_label.config(text=f"{len(peaks)} peaks auto-detected")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Auto peak detection failed:\n{str(e)}")
            self.update_info(f"Auto detection error: {traceback.format_exc()}\n")

    def toggle_overlap_mode(self):
        """Toggle overlap mode for better handling of overlapping peaks"""
        self.overlap_mode = not self.overlap_mode
        if self.overlap_mode:
            self.btn_overlap_mode.config(bg='#32CD32', text="Overlap ON")
            self.group_distance_threshold = 3.5
            self.update_info("Overlap mode ON: Peaks within 3.5*FWHM will be grouped together\n")
        else:
            self.btn_overlap_mode.config(bg='#FF6B9D', text="Overlap")
            self.group_distance_threshold = 2.5
            self.update_info("Overlap mode OFF: Standard grouping (2.5*FWHM)\n")

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
                marker_tuple = self.bg_markers.pop()
                try:
                    marker_tuple[0].remove()
                    marker_tuple[1].remove()
                except:
                    pass
                self.update_bg_connect_line()
                self.canvas.draw()

                if len(self.bg_points) < 2:
                    self.btn_subtract_bg.config(state=tk.DISABLED)

        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

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
        self.btn_quick_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)

    # ==================== Peak Fitting ====================

    def fit_peaks(self):
        """Fit selected peaks using appropriate profile function"""
        if len(self.selected_peaks) == 0:
            messagebox.showwarning("No Peaks", "Please select at least one peak first!")
            return

        fit_method = self.fit_method.get()
        self.update_info(f"Fitting {len(self.selected_peaks)} peaks using {fit_method}...\n")
        self.status_label.config(text="Fitting in progress...")
        self.master.update()

        try:
            # Sort peaks by position
            sorted_indices = sorted(range(len(self.selected_peaks)),
                                   key=lambda i: self.x[self.selected_peaks[i]])
            sorted_peaks = [self.selected_peaks[i] for i in sorted_indices]

            # Fit global background
            self.update_info("Fitting global background...\n")

            if len(self.bg_points) >= 2:
                sorted_bg_points = sorted(self.bg_points, key=lambda p: p[0])
                bg_x = np.array([p[0] for p in sorted_bg_points])
                bg_y = np.array([p[1] for p in sorted_bg_points])
                global_bg = np.interp(self.x, bg_x, bg_y)
                global_bg_points = sorted_bg_points
                self.update_info(f"Using {len(global_bg_points)} manually selected background points\n")
            else:
                global_bg, global_bg_points = BackgroundFitter.fit_global_background(
                    self.x, self.y, sorted_peaks, method='piecewise')
                self.update_info(f"Piecewise linear background fitted "
                               f"with {len(global_bg_points)} anchor points\n")

            # Subtract global background
            y_nobg = self.y - global_bg

            # Estimate FWHM for each peak
            fwhm_estimates = []
            for idx in sorted_peaks:
                window_size = 50
                left = max(0, idx - window_size)
                right = min(len(self.x), idx + window_size)
                x_local = self.x[left:right]
                y_local = y_nobg[left:right]
                local_peak_idx = idx - left
                fwhm, _ = PeakProfile.estimate_fwhm(x_local, y_local, local_peak_idx)
                fwhm_estimates.append(fwhm)

            # Group peaks using DBSCAN clustering
            peak_positions = np.array([self.x[idx] for idx in sorted_peaks])
            avg_fwhm = np.mean(fwhm_estimates)
            eps = avg_fwhm * self.group_distance_threshold

            cluster_labels, n_clusters = PeakClusterer.cluster_peaks(peak_positions, eps=eps)

            peak_groups = []
            for cluster_id in range(max(cluster_labels) + 1):
                group = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if group:
                    peak_groups.append(group)

            peak_groups.sort(key=lambda g: self.x[sorted_peaks[g[0]]])

            self.update_info(f"DBSCAN clustering: {n_clusters} groups (eps={eps:.4f})\n")

            # Fit each group
            use_voigt = (fit_method == "voigt")
            all_popt = {}
            group_windows = []

            for g_idx, group in enumerate(peak_groups):
                self.status_label.config(text=f"Fitting group {g_idx+1}/{len(peak_groups)}...")
                self.master.update()

                group_peak_indices = [sorted_peaks[i] for i in group]
                group_fwhms = [fwhm_estimates[i] for i in group]
                is_overlapping = len(group) > 1

                # Create fitting window
                window_multiplier = 4 if (is_overlapping and self.overlap_mode) else 3
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
                y_fit_nobg = y_nobg[left_idx:right_idx]

                if len(x_fit) < 5:
                    continue

                # Build fit parameters
                p0, bounds_lower, bounds_upper = self._build_fit_parameters(
                    group, sorted_peaks, left_idx, fwhm_estimates, y_fit_nobg, use_voigt)

                # Perform fitting
                popt = self._perform_group_fit(x_fit, y_fit_nobg, p0, bounds_lower, bounds_upper,
                                              len(group), use_voigt, is_overlapping)

                if popt is not None:
                    # Store results
                    n_params = 4 if use_voigt else 5
                    for j, i in enumerate(group):
                        offset = j * n_params
                        all_popt[i] = {
                            'params': popt[offset:offset+n_params],
                            'group_idx': g_idx,
                            'window': (left_idx, right_idx)
                        }

            # Plot results
            self._plot_fit_results(all_popt, sorted_indices, sorted_peaks, peak_groups,
                                  group_windows, global_bg, global_bg_points, use_voigt)

            # Extract and display results
            self._extract_and_display_results(all_popt, sorted_indices, use_voigt, fit_method)

        except Exception as e:
            import traceback
            messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
            self.update_info(f"Fitting failed: {traceback.format_exc()}\n")
            self.status_label.config(text="Fitting failed")

    def _build_fit_parameters(self, group, sorted_peaks, left_idx, fwhm_estimates,
                             y_fit_nobg, use_voigt):
        """Build initial parameters and bounds for curve fitting"""
        p0 = []
        bounds_lower = []
        bounds_upper = []

        dx = np.mean(np.diff(self.x))
        y_range = np.max(y_fit_nobg) - np.min(y_fit_nobg)

        for i in group:
            idx = sorted_peaks[i]
            local_idx = idx - left_idx
            cen_guess = self.x[idx]
            fwhm_est = fwhm_estimates[i]

            sig_guess = fwhm_est / 2.355
            gam_guess = fwhm_est / 2

            peak_height = y_fit_nobg[local_idx] if local_idx < len(y_fit_nobg) else np.max(y_fit_nobg)
            if peak_height <= 0:
                peak_height = np.max(y_fit_nobg) * 0.5

            amp_guess = peak_height * sig_guess * np.sqrt(2 * np.pi)

            amp_lower = 0
            amp_upper = y_range * sig_guess * np.sqrt(2 * np.pi) * 10

            center_tolerance = fwhm_est * 0.8 if self.overlap_mode else fwhm_est * 0.5

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

        return p0, bounds_lower, bounds_upper

    def _perform_group_fit(self, x_fit, y_fit_nobg, p0, bounds_lower, bounds_upper,
                          n_peaks, use_voigt, is_overlapping):
        """Perform curve fitting for a group of peaks"""
        # Define fitting function
        if use_voigt:
            def multi_peak_func(x, *params):
                y = np.zeros_like(x)
                for i in range(n_peaks):
                    offset = i * 4
                    amp, cen, sig, gam = params[offset:offset+4]
                    y += PeakProfile.voigt(x, amp, cen, sig, gam)
                return y
        else:
            def multi_peak_func(x, *params):
                y = np.zeros_like(x)
                for i in range(n_peaks):
                    offset = i * 5
                    amp, cen, sig, gam, eta = params[offset:offset+5]
                    y += PeakProfile.pseudo_voigt(x, amp, cen, sig, gam, eta)
                return y

        max_iter = 30000 if (is_overlapping or self.overlap_mode) else 10000
        ftol = 1e-9 if (is_overlapping or self.overlap_mode) else 1e-8
        xtol = 1e-9 if (is_overlapping or self.overlap_mode) else 1e-8

        try:
            popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               method='trf', maxfev=max_iter,
                               ftol=ftol, xtol=xtol)
            return popt
        except Exception:
            try:
                popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                                   p0=p0, bounds=(bounds_lower, bounds_upper),
                                   method='dogbox', maxfev=50000)
                return popt
            except Exception as e:
                self.update_info(f"Group fit failed: {str(e)}\n")
                return None

    def _plot_fit_results(self, all_popt, sorted_indices, sorted_peaks, peak_groups,
                         group_windows, global_bg, global_bg_points, use_voigt):
        """Plot fitting results"""
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_peaks)))

        # Clear manual background connecting line
        if self.bg_connect_line is not None:
            try:
                self.bg_connect_line.remove()
            except:
                pass
            self.bg_connect_line = None

        # Plot global background
        if len(global_bg_points) >= 2:
            bg_x = [p[0] for p in global_bg_points]
            bg_y = [p[1] for p in global_bg_points]
            bg_markers, = self.ax.plot(bg_x, bg_y, 'o', color='#4169E1',
                                      markersize=6, alpha=0.8, zorder=3)
            self.fit_lines.append(bg_markers)

            bg_label = 'Manual Background' if len(self.bg_points) >= 2 else 'Auto Background'
            bg_line, = self.ax.plot(self.x, global_bg, '-', color='#4169E1',
                                   linewidth=1.5, alpha=0.6,
                                   label=bg_label, zorder=3)
            self.fit_lines.append(bg_line)

        # Plot total fit for each group
        for g_idx, (left, right) in enumerate(group_windows):
            x_region = self.x[left:right]
            x_smooth = np.linspace(x_region.min(), x_region.max(), 400)
            bg_smooth = np.interp(x_smooth, self.x, global_bg)

            y_total = bg_smooth.copy()
            group = peak_groups[g_idx]

            for i in group:
                if i not in all_popt:
                    continue
                params = all_popt[i]['params']
                if use_voigt:
                    y_total += PeakProfile.voigt(x_smooth, *params)
                else:
                    y_total += PeakProfile.pseudo_voigt(x_smooth, *params)

            label = 'Total Fit' if g_idx == 0 else None
            line1, = self.ax.plot(x_smooth, y_total, color='#FF0000', linewidth=1.5,
                                label=label, zorder=5, alpha=0.9)
            self.fit_lines.append(line1)

        # Plot individual peak components
        for i in range(len(sorted_peaks)):
            if i not in all_popt:
                continue

            params = all_popt[i]['params']
            left, right = all_popt[i]['window']

            x_smooth = np.linspace(self.x[left], self.x[right], 400)

            if use_voigt:
                y_component = PeakProfile.voigt(x_smooth, *params)
            else:
                y_component = PeakProfile.pseudo_voigt(x_smooth, *params)

            bg_smooth = np.interp(x_smooth, self.x, global_bg)
            y_with_bg = y_component + bg_smooth

            original_idx = sorted_indices[i]
            line_comp, = self.ax.plot(x_smooth, y_with_bg, '--',
                                     color=colors[i], linewidth=1.2, alpha=0.7, zorder=4,
                                     label=f'Peak {original_idx+1}')
            self.fit_lines.append(line_comp)

    def _extract_and_display_results(self, all_popt, sorted_indices, use_voigt, fit_method):
        """Extract fitting results and display in table"""
        results = []
        info_msg = f"Fitting Results ({fit_method}):\n" + "="*50 + "\n"

        for i in range(len(sorted_indices)):
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
                fwhm = PeakProfile.calculate_fwhm(sig, gam, eta)
                area = PeakProfile.calculate_area(amp, sig, gam, eta)

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
        self.btn_quick_save.config(state=tk.NORMAL)
        self.btn_clear_fit.config(state=tk.NORMAL)

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
        self.btn_quick_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)
        self.update_info("Fit cleared. Peak selections preserved.\n")
        self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    # ==================== Save Operations ====================

    def save_results(self):
        """Save fitting results to user-selected directory"""
        if self.fit_results is None:
            messagebox.showwarning("No Results", "Please fit peaks before saving!")
            return

        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            self._save_results_to_dir(save_dir)
            messagebox.showinfo("Success",
                              f"Results saved to:\n{save_dir}")
            self.update_info(f"Results saved to: {save_dir}\n")
            self.status_label.config(text="Results saved!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")

    def quick_save_results(self):
        """Quickly save fitting results to source file directory"""
        if self.fit_results is None:
            messagebox.showwarning("No Results", "Please fit peaks before saving!")
            return

        if self.filepath is None:
            messagebox.showwarning("No File", "No source file path available!")
            return

        try:
            save_dir = os.path.dirname(self.filepath)
            self._save_results_to_dir(save_dir)

            self.update_info(f"Quick saved to: {save_dir}\n")
            self.status_label.config(text="Quick saved!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to quick save:\n{str(e)}")

    def _save_results_to_dir(self, save_dir):
        """Internal method to save results to a specific directory"""
        self.fit_results['File'] = self.filename
        csv_path = os.path.join(save_dir, f"{self.filename}_fit_results.csv")
        self.fit_results.to_csv(csv_path, index=False)

        fig_path = os.path.join(save_dir, f"{self.filename}_fit_plot.png")
        self.fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

        return csv_path, fig_path

    # ==================== Utility Methods ====================

    def update_info(self, message):
        """Update info text"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)


# ==================== Main Entry Point ====================
def main(icon_path=None):
    """
    Main entry point for the application

    Parameters:
    -----------
    icon_path : str, optional
        Path to custom icon file (.ico, .png, .gif)
    """
    root = tk.Tk()
    app = PeakFittingGUI(root, icon_path=icon_path)
    root.mainloop()


if __name__ == "__main__":
    main()
