# -*- coding: utf-8 -*-
"""
Manual Peak Fitting for XRD Data
Click on peaks to fit them and get peak position, FWHM, and area
@author: candicewang928@gmail.com
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline
import os
import pandas as pd
from scipy.special import wofz

# ---------- Peak Functions ----------
def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile"""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian"""
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def pseudo_voigt_with_bg(x, amplitude, center, sigma, gamma, eta, bg0, bg1):
    """Pseudo-Voigt with linear background"""
    bg = bg0 + bg1 * x
    return pseudo_voigt(x, amplitude, center, sigma, gamma, eta) + bg

# ---------- Multi-peak function ----------
def multi_pseudo_voigt(x, *params):
    """
    Multi-peak Pseudo-Voigt: params = [bg0, bg1, amp1, cen1, sig1, gam1, eta1, amp2, cen2, ...]
    First 2 params: linear background (intercept, slope)
    Then groups of 5 params per peak
    """
    bg = params[0] + params[1] * x
    n_peaks = (len(params) - 2) // 5
    y = bg.copy()
    for i in range(n_peaks):
        offset = 2 + i * 5
        amp, cen, sig, gam, eta = params[offset:offset+5]
        y += pseudo_voigt(x, amp, cen, sig, gam, eta)
    return y

# ---------- FWHM Calculation ----------
def calculate_fwhm(sigma, gamma, eta):
    """
    Calculate FWHM for Pseudo-Voigt profile
    Using the approximation formula
    """
    fwhm_g = 2.355 * sigma  # Gaussian FWHM
    fwhm_l = 2 * gamma      # Lorentzian FWHM
    # Thompson et al. approximation for Pseudo-Voigt FWHM
    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    return fwhm

# ---------- Fit single peak ----------
def fit_single_peak(x_local, y_local):
    """
    Fit a single peak with linear background
    Returns: fitted parameters, success flag
    """
    # Initial guesses
    bg0 = np.mean([y_local[0], y_local[-1]])
    bg1 = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])

    y_no_bg = y_local - (bg0 + bg1 * x_local)
    amplitude_guess = np.max(y_no_bg)
    center_guess = x_local[np.argmax(y_no_bg)]
    sigma_guess = 0.05
    gamma_guess = 0.05
    eta_guess = 0.5

    p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess, bg0, bg1]

    bounds_lower = [0, x_local.min(), 0.001, 0.001, 0, 0, -np.inf]
    bounds_upper = [np.inf, x_local.max(), 1.0, 1.0, 1.0, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(pseudo_voigt_with_bg, x_local, y_local,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=100000)
        return popt, True
    except Exception as e:
        print(f"   Fit failed: {e}")
        return None, False

# ---------- Fit single peak with fixed background ----------
def fit_single_peak_with_fixed_bg(x_local, y_local, bg_intercept, bg_slope):
    """
    Fit a single peak with user-defined fixed background
    Returns: fitted parameters in same format as fit_single_peak
    """
    # Subtract the fixed background
    bg = bg_intercept + bg_slope * x_local
    y_subtracted = y_local - bg

    # Initial guesses for peak only
    amplitude_guess = np.max(y_subtracted)
    center_guess = x_local[np.argmax(y_subtracted)]
    sigma_guess = 0.05
    gamma_guess = 0.05
    eta_guess = 0.5

    p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess]
    bounds_lower = [0, x_local.min(), 0.001, 0.001, 0]
    bounds_upper = [np.inf, x_local.max(), 1.0, 1.0, 1.0]

    try:
        popt, pcov = curve_fit(pseudo_voigt, x_local, y_subtracted,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=100000)
        # Return in same format as fit_single_peak: [amp, cen, sig, gam, eta, bg0, bg1]
        return list(popt) + [bg_intercept, bg_slope], True
    except Exception as e:
        print(f"   Fit failed: {e}")
        return None, False

# ---------- Fit multiple peaks ----------
def fit_multi_peak(x_local, y_local, peak_positions):
    """
    Fit multiple peaks simultaneously with shared background
    Returns: fitted parameters, success flag
    """
    n_peaks = len(peak_positions)

    # Background initial guess
    bg0 = np.mean([y_local[0], y_local[-1]])
    bg1 = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])

    # Initial guess for each peak
    p0 = [bg0, bg1]
    bounds_lower = [0, -np.inf]
    bounds_upper = [np.inf, np.inf]

    for pos in peak_positions:
        # Find local maximum near the position
        idx = np.argmin(np.abs(x_local - pos))
        search_range = min(10, len(x_local)//4)
        left_idx = max(0, idx - search_range)
        right_idx = min(len(x_local), idx + search_range)
        local_max_idx = left_idx + np.argmax(y_local[left_idx:right_idx])

        amp_guess = y_local[local_max_idx] - bg0
        cen_guess = x_local[local_max_idx]
        sig_guess = 0.05
        gam_guess = 0.05
        eta_guess = 0.5

        p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
        bounds_lower.extend([0, x_local.min(), 0.001, 0.001, 0])
        bounds_upper.extend([np.inf, x_local.max(), 1.0, 1.0, 1.0])

    try:
        popt, pcov = curve_fit(multi_pseudo_voigt, x_local, y_local,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=200000)
        return popt, True
    except Exception as e:
        print(f"   Multi-peak fit failed: {e}")
        return None, False

# ---------- Fit multiple peaks with fixed background ----------
def fit_multi_peak_with_fixed_bg(x_local, y_local, peak_positions, bg_intercept, bg_slope):
    """
    Fit multiple peaks with user-defined fixed background
    Returns: fitted parameters in same format as fit_multi_peak
    """
    n_peaks = len(peak_positions)

    # Subtract the fixed background
    bg = bg_intercept + bg_slope * x_local
    y_subtracted = y_local - bg

    # Initial guess for each peak (no background params needed for fitting)
    p0 = []
    bounds_lower = []
    bounds_upper = []

    for pos in peak_positions:
        idx = np.argmin(np.abs(x_local - pos))
        search_range = min(10, len(x_local)//4)
        left_idx = max(0, idx - search_range)
        right_idx = min(len(x_local), idx + search_range)
        local_max_idx = left_idx + np.argmax(y_subtracted[left_idx:right_idx])

        amp_guess = y_subtracted[local_max_idx]
        cen_guess = x_local[local_max_idx]
        sig_guess = 0.05
        gam_guess = 0.05
        eta_guess = 0.5

        p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
        bounds_lower.extend([0, x_local.min(), 0.001, 0.001, 0])
        bounds_upper.extend([np.inf, x_local.max(), 1.0, 1.0, 1.0])

    # Define fitting function for multiple peaks without background
    def multi_peak_no_bg(x, *params):
        y = np.zeros_like(x)
        n = len(params) // 5
        for i in range(n):
            offset = i * 5
            amp, cen, sig, gam, eta = params[offset:offset+5]
            y += pseudo_voigt(x, amp, cen, sig, gam, eta)
        return y

    try:
        popt, pcov = curve_fit(multi_peak_no_bg, x_local, y_subtracted,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=200000)
        # Return in format: [bg0, bg1, amp1, cen1, sig1, gam1, eta1, ...]
        result = [bg_intercept, bg_slope] + list(popt)
        return result, True
    except Exception as e:
        print(f"   Multi-peak fit failed: {e}")
        return None, False

# ---------- Interactive Peak Selection ----------
class PeakSelector:
    def __init__(self, x, y, filename, save_dir):
        self.x = x
        self.y = y
        self.filename = filename
        self.save_dir = save_dir
        self.results = []
        self.peak_count = 0
        self.fit_lines = []  # Store fit plot objects
        self.picking_enabled = True
        # Multi-peak selection
        self.multi_peak_positions = []  # Temporary storage for Ctrl+click positions
        self.multi_peak_markers = []    # Temporary markers for multi-peak selection
        # Manual background selection
        self.bg_mode = False  # Whether in background selection mode
        self.bg_points = []   # Background points (x, y)
        self.bg_markers = []  # Background markers
        self.bg_line_plot = None  # Background line plot
        self.bg_fitted = False  # Whether background has been fitted
        self.bg_spline = None  # Spline function for background
        self.bg_range = None  # Range of background (min_x, max_x)

    def run(self):
        """Run interactive peak selection with zoom/pan support"""
        # Create figure with space for buttons
        self.fig = plt.figure(figsize=(14, 8))

        # Main plot area (adjusted for two button rows)
        self.ax = self.fig.add_axes([0.1, 0.18, 0.85, 0.72])
        self.ax.plot(self.x, self.y, 'b-', linewidth=0.8, label='Data')
        self.ax.set_xlabel('2θ (degree)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'{self.filename}\nClick on peaks to fit (use toolbar to zoom/pan)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')

        # Add buttons (two rows)
        # Top row - Background controls
        ax_open = self.fig.add_axes([0.1, 0.06, 0.08, 0.04])
        ax_bg = self.fig.add_axes([0.2, 0.06, 0.08, 0.04])
        ax_fit_bg = self.fig.add_axes([0.3, 0.06, 0.08, 0.04])
        ax_clear_bg = self.fig.add_axes([0.4, 0.06, 0.08, 0.04])

        # Bottom row - Peak controls
        ax_multi = self.fig.add_axes([0.1, 0.01, 0.08, 0.04])
        ax_undo = self.fig.add_axes([0.2, 0.01, 0.08, 0.04])
        ax_clear = self.fig.add_axes([0.3, 0.01, 0.08, 0.04])
        ax_finish = self.fig.add_axes([0.4, 0.01, 0.08, 0.04])

        self.btn_open = Button(ax_open, 'Open')
        self.btn_open.on_clicked(self.on_open_file)

        self.btn_bg = Button(ax_bg, 'Sel BG')
        self.btn_bg.on_clicked(self.on_select_bg)

        self.btn_fit_bg = Button(ax_fit_bg, 'Fit BG')
        self.btn_fit_bg.on_clicked(self.on_fit_bg)

        self.btn_clear_bg = Button(ax_clear_bg, 'Clr BG')
        self.btn_clear_bg.on_clicked(self.on_clear_bg)

        self.btn_multi = Button(ax_multi, 'Fit Peak')
        self.btn_multi.on_clicked(self.on_fit_multi_peak)

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.on_undo)

        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.on_clear)

        self.btn_finish = Button(ax_finish, 'Save')
        self.btn_finish.on_clicked(self.on_finish)

        # Connect mouse click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # Connect scroll event for zoom
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        print("\n" + "="*60)
        print("Interactive Peak Fitting Mode")
        print("="*60)
        print("- SCROLL WHEEL: zoom in/out at cursor position")
        print("- Use toolbar to PAN (hand icon)")
        print("")
        print("Workflow:")
        print("1. Click 'Sel BG' → click background points")
        print("2. Click 'Fit BG' to fit smooth curve through points")
        print("3. Click on peak positions (cyan triangles)")
        print("4. Click 'Fit Peak' to fit peaks")
        print("="*60 + "\n")

        plt.show(block=True)

    def on_scroll(self, event):
        """Handle scroll wheel for zooming"""
        if event.inaxes != self.ax:
            return

        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Get mouse position
        xdata = event.xdata
        ydata = event.ydata

        # Zoom factor
        if event.button == 'up':
            scale_factor = 0.8  # Zoom in
        elif event.button == 'down':
            scale_factor = 1.25  # Zoom out
        else:
            return

        # Calculate new limits centered on mouse position
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        # Keep mouse position at the same relative location
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse click - select background or peak positions"""
        # Ignore clicks outside the main axes
        if event.inaxes != self.ax:
            return

        # Only respond to left click (button 1)
        if event.button != 1:
            return

        # Check if toolbar is in zoom/pan mode
        toolbar = self.fig.canvas.manager.toolbar
        if toolbar.mode != '':
            return  # Don't pick peaks while zooming/panning

        x_click = event.xdata
        y_click = event.ydata

        # Background selection mode - stays in this mode until user clicks other button
        if self.bg_mode:
            self.bg_points.append((x_click, y_click))
            marker, = self.ax.plot(x_click, y_click, 'ms', markersize=5, alpha=0.8)
            self.bg_markers.append(marker)
            self.fig.canvas.draw()

            print(f"   BG point {len(self.bg_points)} at 2θ = {x_click:.4f}")
            return

        # Peak selection mode - add to peak positions (no Ctrl needed)
        self.multi_peak_positions.append(x_click)

        # Mark with cyan marker
        marker, = self.ax.plot(x_click, y_click, 'c^', markersize=7, alpha=0.8)
        self.multi_peak_markers.append(marker)
        self.fig.canvas.draw()

        print(f"   Peak position {len(self.multi_peak_positions)} at 2θ = {x_click:.4f}")

    def fit_and_plot_peak(self, x_pos):
        """Fit a peak at the clicked position and plot result immediately"""
        # Find the index closest to clicked position
        idx = np.argmin(np.abs(self.x - x_pos))

        # Extract local region (window around the peak)
        window = 50  # Points on each side
        left = max(0, idx - window)
        right = min(len(self.x), idx + window)

        x_local = self.x[left:right]
        y_local = self.y[left:right]

        # Fit the peak with manual or auto background
        if len(self.bg_points) == 2:
            # Use manual background
            x1, y1 = self.bg_points[0]
            x2, y2 = self.bg_points[1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            intercept = y1 - slope * x1
            popt, success = fit_single_peak_with_fixed_bg(x_local, y_local, intercept, slope)
        else:
            # Auto background
            popt, success = fit_single_peak(x_local, y_local)

        if success:
            self.peak_count += 1
            amplitude, center, sigma, gamma, eta, bg0, bg1 = popt

            # Calculate FWHM
            fwhm = calculate_fwhm(sigma, gamma, eta)

            # Calculate area (integrate peak without background)
            x_fine = np.linspace(x_local.min(), x_local.max(), 1000)
            y_peak = pseudo_voigt(x_fine, amplitude, center, sigma, gamma, eta)
            area = trapezoid(y_peak, x_fine)

            # Plot fit on main figure
            x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
            y_fit = pseudo_voigt_with_bg(x_smooth, *popt)
            bg_line = bg0 + bg1 * x_smooth

            # Plot and store references for undo
            fit_line, = self.ax.plot(x_smooth, y_fit, 'r-', linewidth=2, alpha=0.8)
            bg_plot, = self.ax.plot(x_smooth, bg_line, 'g--', linewidth=1, alpha=0.6)
            peak_marker, = self.ax.plot(center, pseudo_voigt_with_bg(center, *popt), 'r*', markersize=15)

            # Add text annotation
            text_y = pseudo_voigt_with_bg(center, *popt)
            annotation = self.ax.annotate(
                f'#{self.peak_count}\n2θ={center:.3f}\nFWHM={fwhm:.4f}\nArea={area:.1f}',
                xy=(center, text_y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )

            # Store plot objects for undo
            self.fit_lines.append({
                'lines': [fit_line, bg_plot, peak_marker],
                'annotation': annotation,
                'result': {
                    'Peak #': self.peak_count,
                    'Center (2θ)': center,
                    'FWHM': fwhm,
                    'Area': area,
                    'Amplitude': amplitude,
                    'Sigma': sigma,
                    'Gamma': gamma,
                    'Eta': eta,
                    'x_local': x_local,
                    'y_local': y_local,
                    'popt': popt
                }
            })

            # Update canvas
            self.fig.canvas.draw()

            print(f"   Peak {self.peak_count}: Center={center:.4f}, FWHM={fwhm:.4f}, Area={area:.2f}")
        else:
            print(f"   Fitting failed at 2θ = {x_pos:.4f}")

    def on_undo(self, event):
        """Remove the last fitted peak"""
        if len(self.fit_lines) == 0:
            print("   Nothing to undo")
            return

        # Get last fit
        last_fit = self.fit_lines.pop()

        # Remove plot objects
        for line in last_fit['lines']:
            line.remove()
        last_fit['annotation'].remove()

        self.peak_count -= 1
        self.fig.canvas.draw()
        print(f"   Removed last peak. {len(self.fit_lines)} peaks remaining.")

    def on_clear(self, event):
        """Clear all fitted peaks and background"""
        for fit_data in self.fit_lines:
            for line in fit_data['lines']:
                line.remove()
            fit_data['annotation'].remove()

        self.fit_lines = []
        self.peak_count = 0

        # Also clear multi-peak selection
        for marker in self.multi_peak_markers:
            marker.remove()
        self.multi_peak_markers = []
        self.multi_peak_positions = []

        # Also clear background
        for marker in self.bg_markers:
            marker.remove()
        self.bg_markers = []
        self.bg_points = []
        if self.bg_line_plot:
            self.bg_line_plot.remove()
            self.bg_line_plot = None
        self.bg_mode = False
        self.bg_fitted = False
        self.bg_spline = None
        self.bg_range = None

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        print("   All cleared.")

    def on_select_bg(self, event):
        """Enter background selection mode"""
        # Clear previous background selection
        for marker in self.bg_markers:
            marker.remove()
        self.bg_markers = []
        self.bg_points = []

        self.bg_mode = True
        print("   Background selection mode: click 2 points to define background line")

        self.fig.canvas.draw()

    def on_clear_bg(self, event):
        """Clear manual background and return to auto mode"""
        for marker in self.bg_markers:
            marker.remove()
        self.bg_markers = []
        self.bg_points = []

        if self.bg_line_plot:
            self.bg_line_plot.remove()
            self.bg_line_plot = None

        self.bg_mode = False
        self.bg_fitted = False
        self.bg_spline = None
        self.bg_range = None
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        print("   Background cleared. Ready for peak selection.")

    def on_fit_bg(self, event):
        """Fit smooth curve through selected background points"""
        if len(self.bg_points) < 2:
            print("   Need at least 2 background points!")
            return

        # Sort points by x
        points = sorted(self.bg_points, key=lambda p: p[0])
        x_bg = np.array([p[0] for p in points])
        y_bg = np.array([p[1] for p in points])

        # Remove old background line
        if self.bg_line_plot:
            self.bg_line_plot.remove()

        # Fit smooth background - always use linear fit for zero curvature
        # This ensures the background is always flat/smooth without oscillations
        # Linear fit (least squares) through all selected points
        coeffs = np.polyfit(x_bg, y_bg, 1)  # degree=1 for linear
        self.bg_spline = np.poly1d(coeffs)

        # Plot fitted background - only within the range of selected points
        x_smooth = np.linspace(x_bg.min(), x_bg.max(), 500)
        y_smooth = self.bg_spline(x_smooth)
        self.bg_line_plot, = self.ax.plot(x_smooth, y_smooth, 'm-', linewidth=2,
                                           alpha=0.7, label='Fitted BG')
        self.ax.legend(loc='upper right')

        # Store background range for peak fitting
        self.bg_range = (x_bg.min(), x_bg.max())

        self.bg_fitted = True
        self.bg_mode = False  # Exit background selection mode
        self.fig.canvas.draw()

        print(f"   Background fitted with {len(points)} points (range: {x_bg.min():.2f} - {x_bg.max():.2f}). Ready for peak selection.")

    def fit_peaks_with_spline_bg(self, x_local, y_local, peak_positions):
        """Fit peaks using spline background"""
        n_peaks = len(peak_positions)

        # Subtract spline background
        bg = self.bg_spline(x_local)
        y_subtracted = y_local - bg

        # Initial guesses
        p0 = []
        bounds_lower = []
        bounds_upper = []

        for pos in peak_positions:
            idx = np.argmin(np.abs(x_local - pos))
            search_range = min(10, len(x_local)//4)
            left_idx = max(0, idx - search_range)
            right_idx = min(len(x_local), idx + search_range)
            local_max_idx = left_idx + np.argmax(y_subtracted[left_idx:right_idx])

            amp_guess = max(y_subtracted[local_max_idx], 1)
            cen_guess = x_local[local_max_idx]
            sig_guess = 0.05
            gam_guess = 0.05
            eta_guess = 0.5

            p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
            bounds_lower.extend([0, x_local.min(), 0.001, 0.001, 0])
            bounds_upper.extend([np.inf, x_local.max(), 1.0, 1.0, 1.0])

        # Fitting function for peaks only
        def multi_peak_no_bg(x, *params):
            y = np.zeros_like(x)
            n = len(params) // 5
            for i in range(n):
                offset = i * 5
                amp, cen, sig, gam, eta = params[offset:offset+5]
                y += pseudo_voigt(x, amp, cen, sig, gam, eta)
            return y

        try:
            popt, pcov = curve_fit(multi_peak_no_bg, x_local, y_subtracted,
                                   p0=p0, bounds=(bounds_lower, bounds_upper),
                                   maxfev=200000)
            return list(popt), True
        except Exception as e:
            print(f"   Fit failed: {e}")
            return None, False

    def on_open_file(self, event):
        """Open a new file"""
        new_file = select_file_dialog(os.path.dirname(self.save_dir))
        if new_file:
            # Close current figure
            plt.close(self.fig)
            # Open new file
            run_peak_fitting(new_file)

    def on_fit_multi_peak(self, event):
        """Fit selected peaks (single or multiple)"""
        if len(self.multi_peak_positions) < 1:
            print("   No peaks selected. Use Ctrl+Click to select peaks first.")
            return

        # Sort positions
        positions = sorted(self.multi_peak_positions)
        n_peaks = len(positions)

        # Determine fitting range (extend beyond outer peaks)
        min_pos = min(positions)
        max_pos = max(positions)

        # Find indices for the range
        idx_min = np.argmin(np.abs(self.x - min_pos))
        idx_max = np.argmin(np.abs(self.x - max_pos))

        # Add window on each side
        window = 30
        left = max(0, idx_min - window)
        right = min(len(self.x), idx_max + window)

        x_local = self.x[left:right]
        y_local = self.y[left:right]

        print(f"   Fitting {n_peaks} peak(s)...")

        # Fit peaks with manual spline or auto background
        if self.bg_fitted and self.bg_spline is not None:
            # Use spline background
            popt, success = self.fit_peaks_with_spline_bg(x_local, y_local, positions)
            if success:
                # Background is from spline
                x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
                bg_line = self.bg_spline(x_smooth)
                # Reconstruct total fit
                y_fit = bg_line.copy()
                for i in range(n_peaks):
                    offset = i * 5
                    amp, cen, sig, gam, eta = popt[offset:offset+5]
                    y_fit += pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)
                # For storage compatibility, prepend dummy bg params
                popt = [0, 0] + list(popt)
                bg0, bg1 = 0, 0  # Dummy values, actual bg is from spline
        else:
            # Auto background
            popt, success = fit_multi_peak(x_local, y_local, positions)
            if success:
                bg0, bg1 = popt[0], popt[1]
                x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
                y_fit = multi_pseudo_voigt(x_smooth, *popt)
                bg_line = bg0 + bg1 * x_smooth

        if success:

            # Plot total fit and background
            fit_line, = self.ax.plot(x_smooth, y_fit, 'r-', linewidth=2, alpha=0.8)
            bg_plot, = self.ax.plot(x_smooth, bg_line, 'g--', linewidth=1, alpha=0.6)

            plot_objects = [fit_line, bg_plot]
            annotations = []

            # Plot and annotate individual peaks
            colors = plt.cm.Set1(np.linspace(0, 1, n_peaks))
            for i in range(n_peaks):
                offset = 2 + i * 5
                amp, cen, sig, gam, eta = popt[offset:offset+5]

                # Calculate individual peak curve
                y_single = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta) + bg_line

                # Plot individual peak
                peak_plot, = self.ax.plot(x_smooth, y_single, ':', linewidth=1.5,
                                          color=colors[i], alpha=0.8)
                plot_objects.append(peak_plot)

                # Calculate FWHM and area
                fwhm = calculate_fwhm(sig, gam, eta)
                y_peak_only = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)
                area = trapezoid(y_peak_only, x_smooth)

                # Calculate peak height at center (handle spline vs linear background)
                if self.bg_fitted and self.bg_spline is not None:
                    peak_height = pseudo_voigt(cen, amp, cen, sig, gam, eta) + self.bg_spline(cen)
                else:
                    peak_height = pseudo_voigt_with_bg(cen, amp, cen, sig, gam, eta, bg0, bg1)

                # Mark peak center
                peak_marker, = self.ax.plot(cen, peak_height,
                                            '*', markersize=12, color=colors[i])
                plot_objects.append(peak_marker)

                # Add annotation
                self.peak_count += 1
                text_y = peak_height
                annotation = self.ax.annotate(
                    f'#{self.peak_count}\n2θ={cen:.3f}\nFWHM={fwhm:.4f}\nArea={area:.1f}',
                    xy=(cen, text_y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
                annotations.append(annotation)

                # Store result - for spline background, approximate with linear bg at center
                if self.bg_fitted and self.bg_spline is not None:
                    bg_at_center = float(self.bg_spline(cen))
                    # Approximate linear background for individual plot display
                    stored_bg0 = bg_at_center
                    stored_bg1 = 0
                else:
                    stored_bg0 = bg0
                    stored_bg1 = bg1

                self.fit_lines.append({
                    'lines': plot_objects if i == n_peaks-1 else [],  # Only store once
                    'annotation': annotation,
                    'result': {
                        'Peak #': self.peak_count,
                        'Center (2θ)': cen,
                        'FWHM': fwhm,
                        'Area': area,
                        'Amplitude': amp,
                        'Sigma': sig,
                        'Gamma': gam,
                        'Eta': eta,
                        'x_local': x_local,
                        'y_local': y_local,
                        'popt': [amp, cen, sig, gam, eta, stored_bg0, stored_bg1]
                    }
                })

                print(f"   Peak {self.peak_count}: Center={cen:.4f}, FWHM={fwhm:.4f}, Area={area:.2f}")

            # Store plot objects in the last result for proper undo
            if n_peaks > 0:
                self.fit_lines[-1]['lines'] = plot_objects

        # Clear temporary markers
        for marker in self.multi_peak_markers:
            marker.remove()
        self.multi_peak_markers = []
        self.multi_peak_positions = []

        self.fig.canvas.draw()

    def on_finish(self, event):
        """Save results and close"""
        if len(self.fit_lines) == 0:
            print("\nNo peaks fitted!")
            plt.close(self.fig)
            return

        # Collect results
        self.results = [fit_data['result'] for fit_data in self.fit_lines]

        # Save results
        self.save_results()

        # Save the current figure
        fig_path = os.path.join(self.save_dir, f"{self.filename}_manual_fit.png")
        self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   Figure saved to: {fig_path}")

        plt.close(self.fig)

        # Show individual peak fits
        self.show_individual_fits()

    def save_results(self):
        """Save fitting results to CSV"""
        if not self.results:
            return

        # Create DataFrame with main results
        df_data = []
        for r in self.results:
            df_data.append({
                'Peak #': r['Peak #'],
                'Center (2θ)': r['Center (2θ)'],
                'FWHM': r['FWHM'],
                'Area': r['Area'],
                'Amplitude': r['Amplitude'],
                'Sigma': r['Sigma'],
                'Gamma': r['Gamma'],
                'Eta': r['Eta']
            })

        df = pd.DataFrame(df_data)
        csv_path = os.path.join(self.save_dir, f"{self.filename}_manual_fit.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n   Results saved to: {csv_path}")

    def show_individual_fits(self):
        """Display individual fitted peaks in subplots"""
        n_peaks = len(self.results)
        if n_peaks == 0:
            return

        # Create subplot layout
        cols = min(3, n_peaks)
        rows = int(np.ceil(n_peaks / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_peaks == 1:
            axs = [axs]
        else:
            axs = axs.flatten() if n_peaks > 1 else [axs]

        for i, result in enumerate(self.results):
            ax = axs[i]
            x_local = result['x_local']
            y_local = result['y_local']
            popt = result['popt']

            # Plot data
            ax.plot(x_local, y_local, 'ko', markersize=3, label='Data')

            # Plot fit
            x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
            y_fit = pseudo_voigt_with_bg(x_smooth, *popt)
            ax.plot(x_smooth, y_fit, 'r-', linewidth=2, label='Fit')

            # Plot background
            bg_line = popt[5] + popt[6] * x_smooth
            ax.plot(x_smooth, bg_line, 'g--', linewidth=1, label='Background')

            # Plot peak without background
            y_peak_only = pseudo_voigt(x_smooth, popt[0], popt[1], popt[2], popt[3], popt[4])
            ax.fill_between(x_smooth, bg_line, y_fit, alpha=0.3, color='red', label='Peak Area')

            # Labels
            ax.set_xlabel('2θ (degree)')
            ax.set_ylabel('Intensity')
            ax.set_title(f"Peak {result['Peak #']}\n2θ={result['Center (2θ)']:.4f}, FWHM={result['FWHM']:.4f}, Area={result['Area']:.1f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(n_peaks, len(axs)):
            fig.delaxes(axs[j])

        plt.suptitle(f"{self.filename} - Individual Peak Fits", fontsize=14)
        plt.tight_layout()
        plt.show()

# ---------- Main function for single file ----------
def run_peak_fitting(file_path, save_dir=None):
    """
    Main function to run manual peak fitting on a single file

    Parameters:
    -----------
    file_path : str
        Path to the data file (.xy or .dat)
    save_dir : str, optional
        Directory to save results. If None, creates 'fit_output' in the same folder
    """
    # Read data
    try:
        with open(file_path, encoding='latin1') as f:
            data = np.genfromtxt(f, comments="#")
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Set save directory
    if save_dir is None:
        folder = os.path.dirname(file_path)
        save_dir = os.path.join(folder, "fit_output")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading: {filename}")
    print(f"Data range: 2θ = {x.min():.2f} to {x.max():.2f}")

    # Start interactive selection
    selector = PeakSelector(x, y, filename, save_dir)
    selector.run()

# ---------- File Selection Dialog ----------
def select_file_dialog(initial_dir=None):
    """
    Open a file selection dialog
    Returns the selected file path or None if cancelled
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select XRD Data File",
        filetypes=[
            ("XRD Data Files", "*.xy *.dat *.txt"),
            ("XY Files", "*.xy"),
            ("DAT Files", "*.dat"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None

# ---------- Launch empty GUI ----------
def launch_gui():
    """
    Launch GUI without loading a file first.
    User can open file through the 'Open File' button.
    """
    # Create figure with space for buttons
    fig = plt.figure(figsize=(14, 8))

    # Main plot area
    ax = fig.add_axes([0.1, 0.18, 0.85, 0.72])
    ax.set_xlabel('2θ (degree)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('Peak Fitting Tool\nClick "Open File" to load data', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.5, 'No data loaded\n\nClick "Open File" to start',
            transform=ax.transAxes, fontsize=16, ha='center', va='center',
            color='gray', alpha=0.7)

    # Add Open File button
    ax_open = fig.add_axes([0.4, 0.4, 0.2, 0.08])
    btn_open = Button(ax_open, 'Open File')

    def on_open(event):
        file_path = select_file_dialog()
        if file_path:
            plt.close(fig)
            run_peak_fitting(file_path)

    btn_open.on_clicked(on_open)

    print("\n" + "="*60)
    print("Peak Fitting Tool")
    print("="*60)
    print("Click 'Open File' to load XRD data")
    print("="*60 + "\n")

    plt.show(block=True)

# ---------- Main ----------
def main():
    """
    Launch peak fitting GUI
    """
    import sys

    # Get file path from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

        # Check if path exists
        if not os.path.exists(file_path):
            print(f"Path not found: {file_path}")
            print("\nUsage: python curve_fitting.py [data_file]")
            print("Or run without arguments to open GUI first")
            return

        # If it's a directory, open file dialog in that directory
        if os.path.isdir(file_path):
            print(f"Opening file selection dialog in: {file_path}")
            selected_file = select_file_dialog(initial_dir=file_path)
            if not selected_file:
                print("No file selected. Exiting.")
                return
            file_path = selected_file

        run_peak_fitting(file_path)
    else:
        # No argument - launch empty GUI
        launch_gui()

if __name__ == "__main__":
    main()
