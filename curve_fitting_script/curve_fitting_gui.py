# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - File Selection & Mouse Zoom
@author: candicewang928@gmail.com
Enhanced with GUI interface and zoom functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import warnings

# Suppress matplotlib font warnings for emojis
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ---------- Peak profile functions ----------
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian"""
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def multi_pseudo_voigt(x, *params):
    """
    Multi-peak with linear background
    params = [bg0, bg1, amp1, cen1, sig1, gam1, eta1, amp2, cen2, ...]
    """
    bg = params[0] + params[1] * x
    n_peaks = (len(params) - 2) // 5
    y = bg.copy()
    for i in range(n_peaks):
        offset = 2 + i * 5
        amp, cen, sig, gam, eta = params[offset:offset+5]
        y += pseudo_voigt(x, amp, cen, sig, gam, eta)
    return y

def calculate_fwhm(sigma, gamma, eta):
    """Calculate FWHM from Pseudo-Voigt parameters"""
    fwhm_g = 2.355 * sigma  # Gaussian FWHM
    fwhm_l = 2 * gamma      # Lorentzian FWHM
    return eta * fwhm_l + (1 - eta) * fwhm_g

def calculate_area(amplitude, sigma, gamma, eta):
    """Calculate integrated area"""
    area_g = amplitude * sigma * np.sqrt(2 * np.pi)
    area_l = amplitude * np.pi * gamma
    return eta * area_l + (1 - eta) * area_g

# ---------- Background fitting functions ----------
def linear_background(x, a, b):
    """Linear background: y = a + b*x"""
    return a + b * x

def polynomial_background(x, *params):
    """Polynomial background"""
    return sum(p * x**i for i, p in enumerate(params))

def chebyshev_background(x, *params):
    """Chebyshev polynomial background"""
    # Normalize x to [-1, 1]
    x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    result = np.zeros_like(x)
    for i, p in enumerate(params):
        result += p * np.cos(i * np.arccos(np.clip(x_norm, -1, 1)))
    return result

# ---------- Main GUI Application ----------
class PeakFittingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive XRD Peak Fitting Tool")
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
        self.bg_fitted = False
        self.bg_params = None
        self.bg_line = None
        self.bg_type = 'linear'
        self.bg_order = 2
        self.selecting_bg = False

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI components"""
        # Top control panel
        control_frame = tk.Frame(self.master, bg='#BA55D3', height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # Buttons with beautiful styling
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
                                  bg='#32CD32', fg='white',
                                  command=self.save_results, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_clear_fit = tk.Button(control_frame, text="Clear Fit",
                                       bg='#FF8C00', fg='white',
                                       command=self.clear_fit, state=tk.DISABLED, **btn_style)
        self.btn_clear_fit.pack(side=tk.LEFT, padx=5, pady=8)

        # Status label
        self.status_label = tk.Label(control_frame, text="Please load a file to start",
                                     bg='#BA55D3', fg='white',
                                     font=('Arial', 11, 'bold'))
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Background fitting control panel
        bg_frame = tk.Frame(self.master, bg='#E6D5F5', height=50)
        bg_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        bg_frame.pack_propagate(False)

        bg_label = tk.Label(bg_frame, text="Background Fitting:",
                           bg='#E6D5F5', fg='#4B0082',
                           font=('Arial', 10, 'bold'))
        bg_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Background type selection
        self.bg_type_var = tk.StringVar(value='linear')
        bg_types = [('Linear', 'linear'), ('Polynomial', 'polynomial'), ('Chebyshev', 'chebyshev')]
        for text, value in bg_types:
            rb = tk.Radiobutton(bg_frame, text=text, variable=self.bg_type_var,
                               value=value, bg='#E6D5F5', fg='#4B0082',
                               font=('Arial', 9), command=self.on_bg_type_change)
            rb.pack(side=tk.LEFT, padx=5)

        # Polynomial order
        tk.Label(bg_frame, text="Order:", bg='#E6D5F5', fg='#4B0082',
                font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.bg_order_var = tk.StringVar(value='2')
        order_combo = ttk.Combobox(bg_frame, textvariable=self.bg_order_var,
                                   values=['1', '2', '3', '4', '5'], width=3)
        order_combo.pack(side=tk.LEFT, padx=2)

        btn_bg_style = {
            'font': ('Arial', 9, 'bold'),
            'width': 14,
            'height': 1,
            'relief': tk.RAISED,
            'bd': 2
        }

        self.btn_select_bg = tk.Button(bg_frame, text="Select BG Points",
                                        bg='#6A5ACD', fg='white',
                                        command=self.toggle_bg_selection,
                                        state=tk.DISABLED, **btn_bg_style)
        self.btn_select_bg.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_fit_bg = tk.Button(bg_frame, text="Fit Background",
                                    bg='#4169E1', fg='white',
                                    command=self.fit_background,
                                    state=tk.DISABLED, **btn_bg_style)
        self.btn_fit_bg.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_subtract_bg = tk.Button(bg_frame, text="Subtract BG",
                                         bg='#2E8B57', fg='white',
                                         command=self.subtract_background,
                                         state=tk.DISABLED, **btn_bg_style)
        self.btn_subtract_bg.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_clear_bg = tk.Button(bg_frame, text="Clear BG",
                                      bg='#DC143C', fg='white',
                                      command=self.clear_background,
                                      state=tk.DISABLED, **btn_bg_style)
        self.btn_clear_bg.pack(side=tk.LEFT, padx=5, pady=8)

        # Main plot area
        plot_frame = tk.Frame(self.master, bg='white')
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='white')
        self.ax.set_facecolor('#FAF0FF')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
        self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_title('Click on peaks to select | Use toolbar to zoom/pan',
                         fontsize=14, fontweight='bold', color='#9370DB')

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add matplotlib navigation toolbar
        toolbar_frame = tk.Frame(plot_frame, bg='#E6D5F5')
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.config(bg='#E6D5F5')

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Info panel at bottom
        info_frame = tk.Frame(self.master, bg='#F0E6FA', height=80)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        info_frame.pack_propagate(False)

        self.info_text = tk.Text(info_frame, height=4, bg='#FAF0FF',
                                 fg='#4B0082', font=('Courier', 10),
                                 relief=tk.SUNKEN, bd=2)
        self.info_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.info_text.insert('1.0', 'Welcome! Load your XRD data file to begin peak fitting.\n')
        self.info_text.insert('2.0', 'Use the toolbar buttons to zoom and pan the plot.\n')
        self.info_text.insert('3.0', 'Click on peaks in the plot to select them for fitting.\n')
        self.info_text.config(state=tk.DISABLED)

    def on_bg_type_change(self):
        """Handle background type change"""
        self.bg_type = self.bg_type_var.get()

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
            # Try reading the file
            with open(filepath, encoding='latin1') as f:
                data = np.genfromtxt(f, comments="#")

            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns (2theta, Intensity)")

            self.x = data[:, 0]
            self.y = data[:, 1]
            self.y_original = self.y.copy()  # Keep original for background subtraction
            self.filepath = filepath
            self.filename = os.path.splitext(os.path.basename(filepath))[0]

            # Reset state
            self.reset_peaks()
            self.clear_background()
            self.fitted = False

            # Plot data - using line instead of points
            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=1.5,
                        label='Data')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar to zoom/pan',
                            fontsize=14, fontweight='bold', color='#9370DB')
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)
            self.canvas.draw()

            # Enable buttons
            self.btn_fit.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_select_bg.config(state=tk.NORMAL)
            self.btn_clear_bg.config(state=tk.NORMAL)

            # Update status
            self.status_label.config(text=f"Loaded: {self.filename}")
            self.update_info(f"File loaded: {self.filename}\n"
                           f"Data points: {len(self.x)}\n"
                           f"Click on peaks to select them for fitting\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.update_info(f"Error loading file: {str(e)}\n")

    def on_click(self, event):
        """Handle mouse clicks to select peaks or background points"""
        if event.inaxes != self.ax or self.x is None:
            return

        # Check if we're in zoom/pan mode
        if self.toolbar.mode != '':
            return

        # Find nearest data point
        x_click = event.xdata
        idx = np.argmin(np.abs(self.x - x_click))
        point_x = self.x[idx]
        point_y = self.y[idx]

        if self.selecting_bg:
            # Select background points
            marker, = self.ax.plot(point_x, point_y, 's', color='#4169E1',
                                  markersize=12, markeredgecolor='#FFD700',
                                  markeredgewidth=2, zorder=10)
            self.bg_points.append((point_x, point_y))
            self.bg_markers.append(marker)
            self.canvas.draw()

            self.update_info(f"Background point {len(self.bg_points)} selected at 2theta = {point_x:.4f}\n")

            # Enable fit background if enough points
            if len(self.bg_points) >= 2:
                self.btn_fit_bg.config(state=tk.NORMAL)
        elif not self.fitted:
            # Select peaks
            marker, = self.ax.plot(point_x, point_y, '*', color='#FF1493',
                                  markersize=20, markeredgecolor='#FFD700',
                                  markeredgewidth=2, zorder=10)
            text = self.ax.text(point_x, point_y * 1.05, f'P{len(self.selected_peaks)+1}',
                               ha='center', fontsize=11, color='#FF1493',
                               fontweight='bold', zorder=11,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE4E1',
                                       edgecolor='#FF69B4', linewidth=2))

            self.selected_peaks.append(idx)
            self.peak_markers.append(marker)
            self.peak_texts.append(text)
            self.canvas.draw()

            self.update_info(f"Peak {len(self.selected_peaks)} selected at 2theta = {point_x:.4f}\n")
            self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def toggle_bg_selection(self):
        """Toggle background point selection mode"""
        self.selecting_bg = not self.selecting_bg
        if self.selecting_bg:
            self.btn_select_bg.config(bg='#FFD700', text="Stop Selection")
            self.status_label.config(text="Selecting background points...")
            self.update_info("Background selection mode: Click on points to define background\n")
        else:
            self.btn_select_bg.config(bg='#6A5ACD', text="Select BG Points")
            self.status_label.config(text=f"{len(self.bg_points)} BG points selected")

    def fit_background(self):
        """Fit background to selected points"""
        if len(self.bg_points) < 2:
            messagebox.showwarning("Insufficient Points", "Please select at least 2 background points!")
            return

        try:
            bg_x = np.array([p[0] for p in self.bg_points])
            bg_y = np.array([p[1] for p in self.bg_points])

            self.bg_type = self.bg_type_var.get()
            self.bg_order = int(self.bg_order_var.get())

            x_smooth = np.linspace(self.x.min(), self.x.max(), 2000)

            if self.bg_type == 'linear':
                popt, _ = curve_fit(linear_background, bg_x, bg_y)
                y_bg = linear_background(x_smooth, *popt)
                self.bg_func = lambda x: linear_background(x, *popt)

            elif self.bg_type == 'polynomial':
                # Fit polynomial of specified order
                coeffs = np.polyfit(bg_x, bg_y, self.bg_order)
                y_bg = np.polyval(coeffs, x_smooth)
                self.bg_func = lambda x: np.polyval(coeffs, x)

            elif self.bg_type == 'chebyshev':
                # Fit Chebyshev polynomial
                p0 = [1.0] * (self.bg_order + 1)
                popt, _ = curve_fit(chebyshev_background, bg_x, bg_y, p0=p0)
                y_bg = chebyshev_background(x_smooth, *popt)
                self.bg_func = lambda x: chebyshev_background(x, *popt)

            # Remove old background line if exists
            if self.bg_line is not None:
                self.bg_line.remove()

            # Plot fitted background
            self.bg_line, = self.ax.plot(x_smooth, y_bg, '--', color='#4169E1',
                                        linewidth=2.5, label='Background Fit', zorder=5)
            self.ax.legend(fontsize=10, loc='best', framealpha=0.9)
            self.canvas.draw()

            self.bg_fitted = True
            self.btn_subtract_bg.config(state=tk.NORMAL)

            self.update_info(f"Background fitted using {self.bg_type} method (order={self.bg_order})\n")
            self.status_label.config(text="Background fitted")

        except Exception as e:
            messagebox.showerror("Fitting Error", f"Failed to fit background:\n{str(e)}")
            self.update_info(f"Background fitting failed: {str(e)}\n")

    def subtract_background(self):
        """Subtract fitted background from data"""
        if not self.bg_fitted:
            messagebox.showwarning("No Background", "Please fit background first!")
            return

        try:
            # Subtract background
            self.y = self.y_original - self.bg_func(self.x)

            # Replot data
            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=1.5,
                        label='Data (BG subtracted)')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename} (Background Subtracted)\nClick on peaks to select',
                            fontsize=14, fontweight='bold', color='#9370DB')
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)

            # Re-add peak markers if any
            for i, idx in enumerate(self.selected_peaks):
                marker, = self.ax.plot(self.x[idx], self.y[idx], '*', color='#FF1493',
                                      markersize=20, markeredgecolor='#FFD700',
                                      markeredgewidth=2, zorder=10)
                text = self.ax.text(self.x[idx], self.y[idx] * 1.05, f'P{i+1}',
                                   ha='center', fontsize=11, color='#FF1493',
                                   fontweight='bold', zorder=11,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE4E1',
                                           edgecolor='#FF69B4', linewidth=2))
                self.peak_markers[i] = marker
                self.peak_texts[i] = text

            self.canvas.draw()

            # Clear background markers
            self.bg_points = []
            self.bg_markers = []
            self.bg_line = None
            self.bg_fitted = False
            self.btn_subtract_bg.config(state=tk.DISABLED)
            self.btn_fit_bg.config(state=tk.DISABLED)

            self.update_info("Background subtracted from data\n")
            self.status_label.config(text="Background subtracted")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to subtract background:\n{str(e)}")

    def clear_background(self):
        """Clear background selection and fit"""
        # Remove background markers
        for marker in self.bg_markers:
            try:
                marker.remove()
            except:
                pass

        # Remove background line
        if self.bg_line is not None:
            try:
                self.bg_line.remove()
            except:
                pass

        self.bg_points = []
        self.bg_markers = []
        self.bg_line = None
        self.bg_fitted = False
        self.selecting_bg = False

        self.btn_select_bg.config(bg='#6A5ACD', text="Select BG Points")
        self.btn_fit_bg.config(state=tk.DISABLED)
        self.btn_subtract_bg.config(state=tk.DISABLED)

        if self.x is not None:
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)
            self.canvas.draw()

        self.update_info("Background selection cleared\n")

    def fit_peaks(self):
        """Fit all selected peaks"""
        if len(self.selected_peaks) == 0:
            messagebox.showwarning("No Peaks", "Please select at least one peak first!")
            return

        self.update_info(f"Fitting {len(self.selected_peaks)} peaks...\n")

        try:
            # Background estimate
            bg0 = np.percentile(self.y, 10)
            bg1 = 0

            # Initial parameters
            p0 = [bg0, bg1]
            bounds_lower = [0, -np.inf]
            bounds_upper = [np.inf, np.inf]

            for idx in self.selected_peaks:
                amp_guess = self.y[idx] - bg0
                cen_guess = self.x[idx]
                sig_guess = 0.05
                gam_guess = 0.05
                eta_guess = 0.5

                p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
                bounds_lower.extend([0, self.x.min(), 0.001, 0.001, 0])
                bounds_upper.extend([np.inf, self.x.max(), 1.0, 1.0, 1.0])

            # Perform fitting
            popt, pcov = curve_fit(multi_pseudo_voigt, self.x, self.y,
                                  p0=p0, bounds=(bounds_lower, bounds_upper),
                                  maxfev=100000)

            # Plot fit
            x_smooth = np.linspace(self.x.min(), self.x.max(), 2000)
            y_fit = multi_pseudo_voigt(x_smooth, *popt)
            bg_line = popt[0] + popt[1] * x_smooth

            line1, = self.ax.plot(x_smooth, y_fit, color='#BA55D3', linewidth=3,
                                label='Total Fit', zorder=5)
            line2, = self.ax.plot(x_smooth, bg_line, '--', color='#FF69B4',
                                linewidth=2, label='Background', zorder=4)

            self.fit_lines.extend([line1, line2])

            # Extract and plot individual peaks
            n_peaks = len(self.selected_peaks)
            results = []

            info_msg = "Fitting Results:\n" + "="*50 + "\n"

            for i in range(n_peaks):
                offset = 2 + i * 5
                amp, cen, sig, gam, eta = popt[offset:offset+5]

                # Plot individual peak
                y_single = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)
                line, = self.ax.plot(x_smooth, y_single + popt[0], ':',
                                    linewidth=2, alpha=0.8,
                                    label=f'Peak {i+1}', zorder=3)
                self.fit_lines.append(line)

                # Calculate metrics
                fwhm = calculate_fwhm(sig, gam, eta)
                area = calculate_area(amp, sig, gam, eta)

                results.append({
                    'Peak': i + 1,
                    'Center_2theta': cen,
                    'FWHM': fwhm,
                    'Area': area,
                    'Amplitude': amp,
                    'Sigma': sig,
                    'Gamma': gam,
                    'Eta': eta
                })

                info_msg += f"Peak {i+1}: 2theta={cen:.4f}, FWHM={fwhm:.4f}, Area={area:.1f}\n"

            self.fit_results = pd.DataFrame(results)
            self.fitted = True

            self.ax.legend(fontsize=10, loc='best', framealpha=0.9)
            self.ax.set_title(f'{self.filename} - Fit Complete',
                            fontsize=14, fontweight='bold', color='#32CD32')
            self.canvas.draw()

            self.update_info(info_msg)
            self.status_label.config(text="Fitting successful!")

            # Enable save button
            self.btn_save.config(state=tk.NORMAL)
            self.btn_clear_fit.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
            self.update_info(f"Fitting failed: {str(e)}\n")

    def clear_fit(self):
        """Clear fitting results but keep peak selections"""
        # Remove fit lines
        for line in self.fit_lines:
            line.remove()
        self.fit_lines = []

        self.fitted = False
        self.fit_results = None

        self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar to zoom/pan',
                         fontsize=14, fontweight='bold', color='#9370DB')
        self.ax.legend(fontsize=11, loc='best', framealpha=0.9)
        self.canvas.draw()

        self.btn_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)
        self.update_info("Fit cleared. Peak selections preserved.\n")
        self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def reset_peaks(self):
        """Clear all peak selections and fits"""
        # Remove markers and texts
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

        # Remove fit lines
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

        if self.x is not None:
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar to zoom/pan',
                            fontsize=14, fontweight='bold', color='#9370DB')
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)
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

        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            # Save CSV
            self.fit_results['File'] = self.filename
            csv_path = os.path.join(save_dir, f"{self.filename}_fit_results.csv")
            self.fit_results.to_csv(csv_path, index=False)

            # Save figure
            fig_path = os.path.join(save_dir, f"{self.filename}_fit_plot.png")
            self.fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')

            messagebox.showinfo("Success",
                              f"Results saved successfully!\n\n"
                              f"CSV: {csv_path}\n"
                              f"Plot: {fig_path}")

            self.update_info(f"Results saved to:\n{save_dir}\n")
            self.status_label.config(text="Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
            self.update_info(f"Save failed: {str(e)}\n")

    def update_info(self, message):
        """Update info text area"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

# ---------- Main ----------
def main():
    root = tk.Tk()
    app = PeakFittingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
