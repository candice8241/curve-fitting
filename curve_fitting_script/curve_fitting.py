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
        self.bg_line = None
        self.bg_connect_line = None  # Line connecting BG points
        self.selecting_bg = False

        # Undo stack for tracking actions
        self.undo_stack = []

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

        # Status label
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

        # Coordinate display label in bg_frame
        self.coord_label = tk.Label(bg_frame, text="",
                                    bg='#E6D5F5', fg='#4B0082',
                                    font=('Courier', 9))
        self.coord_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Main plot area
        plot_frame = tk.Frame(self.master, bg='white')
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='white')
        self.ax.set_facecolor('#FAF0FF')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
        self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_title('Click on peaks to select | Use toolbar or scroll to zoom/pan',
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

        # Connect scroll event for mouse wheel zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Connect mouse motion event for coordinate display
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Info panel at bottom
        info_frame = tk.Frame(self.master, bg='#F0E6FA', height=80)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        info_frame.pack_propagate(False)

        self.info_text = tk.Text(info_frame, height=4, bg='#FAF0FF',
                                 fg='#4B0082', font=('Courier', 10),
                                 relief=tk.SUNKEN, bd=2)
        self.info_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.info_text.insert('1.0', 'Welcome! Load your XRD data file to begin peak fitting.\n')
        self.info_text.insert('2.0', 'Use the toolbar buttons or mouse scroll wheel to zoom and pan the plot.\n')
        self.info_text.insert('3.0', 'Click on peaks in the plot to select them for fitting.\n')
        self.info_text.config(state=tk.DISABLED)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax or self.x is None:
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

        # Calculate relative position of mouse in current view
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        # Set new limits
        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata - new_height * rely, ydata + new_height * (1 - rely)]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Display mouse coordinates in real-time"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
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
            self.undo_stack = []
            self.btn_undo.config(state=tk.DISABLED)

            # Plot data - using line instead of points (thinner line)
            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8,
                        label='Data')
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll to zoom/pan',
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
            # Select background points (smaller marker size)
            marker, = self.ax.plot(point_x, point_y, 's', color='#4169E1',
                                  markersize=6, markeredgecolor='#FFD700',
                                  markeredgewidth=1, zorder=10)
            self.bg_points.append((point_x, point_y))
            self.bg_markers.append(marker)

            # Update connecting line between BG points
            self.update_bg_connect_line()

            self.canvas.draw()

            # Add to undo stack
            self.undo_stack.append(('bg_point', len(self.bg_points) - 1))
            self.btn_undo.config(state=tk.NORMAL)

            self.update_info(f"Background point {len(self.bg_points)} selected at 2theta = {point_x:.4f}\n")

            # Enable subtract background if enough points
            if len(self.bg_points) >= 2:
                self.btn_subtract_bg.config(state=tk.NORMAL)
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

            # Add to undo stack
            self.undo_stack.append(('peak', len(self.selected_peaks) - 1))
            self.btn_undo.config(state=tk.NORMAL)

            self.update_info(f"Peak {len(self.selected_peaks)} selected at 2theta = {point_x:.4f}\n")
            self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def toggle_bg_selection(self):
        """Toggle background point selection mode"""
        self.selecting_bg = not self.selecting_bg
        if self.selecting_bg:
            self.btn_select_bg.config(bg='#FFD700', fg='#000000', text="Stop Selection")
            self.status_label.config(text="Selecting background points...")
            self.update_info("Background selection mode: Click on points to define background\n")
        else:
            self.btn_select_bg.config(bg='#B0A0D0', fg='#2F0060', text="Select BG Points")
            self.status_label.config(text=f"{len(self.bg_points)} BG points selected")

    def update_bg_connect_line(self):
        """Update the line connecting background points"""
        # Remove old connecting line if exists
        if self.bg_connect_line is not None:
            try:
                self.bg_connect_line.remove()
            except:
                pass
            self.bg_connect_line = None

        # Draw new connecting line if we have at least 2 points
        if len(self.bg_points) >= 2:
            # Sort points by x coordinate for proper line connection
            sorted_points = sorted(self.bg_points, key=lambda p: p[0])
            bg_x = [p[0] for p in sorted_points]
            bg_y = [p[1] for p in sorted_points]
            self.bg_connect_line, = self.ax.plot(bg_x, bg_y, '-', color='#4169E1',
                                                 linewidth=1.5, alpha=0.7, zorder=8)

    def undo_action(self):
        """Undo the last peak or background point selection"""
        if not self.undo_stack:
            return

        action_type, index = self.undo_stack.pop()

        if action_type == 'peak':
            # Undo peak selection
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
                self.update_info(f"Undone: Peak {index + 1} removed\n")
                self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

        elif action_type == 'bg_point':
            # Undo background point selection
            if self.bg_points and index == len(self.bg_points) - 1:
                self.bg_points.pop()
                marker = self.bg_markers.pop()
                try:
                    marker.remove()
                except:
                    pass

                # Update connecting line
                self.update_bg_connect_line()

                self.canvas.draw()
                self.update_info(f"Undone: Background point {index + 1} removed\n")

                # Disable subtract background if not enough points
                if len(self.bg_points) < 2:
                    self.btn_subtract_bg.config(state=tk.DISABLED)

        # Disable undo button if stack is empty
        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

    def subtract_background(self):
        """Subtract background using linear interpolation between selected points"""
        if len(self.bg_points) < 2:
            messagebox.showwarning("Insufficient Points", "Please select at least 2 background points!")
            return

        try:
            # Sort background points by x coordinate
            sorted_points = sorted(self.bg_points, key=lambda p: p[0])
            bg_x = np.array([p[0] for p in sorted_points])
            bg_y = np.array([p[1] for p in sorted_points])

            # Linear interpolation between points
            # For x values outside the range, use the nearest endpoint value
            bg_interp = np.interp(self.x, bg_x, bg_y)

            # Subtract background
            self.y = self.y_original - bg_interp

            # Replot data (thinner line)
            self.ax.clear()
            self.ax.plot(self.x, self.y, '-', color='#4B0082', linewidth=0.8,
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
            self.bg_connect_line = None
            self.btn_subtract_bg.config(state=tk.DISABLED)

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

        # Remove connecting line
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

        # Clear bg_point-related items from undo stack
        self.undo_stack = [item for item in self.undo_stack if item[0] != 'bg_point']
        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

        self.btn_select_bg.config(bg='#B0A0D0', fg='#2F0060', text="Select BG Points")
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
            # Estimate data range for better bounds
            x_range = self.x.max() - self.x.min()
            y_range = self.y.max() - self.y.min()

            # Average data spacing
            dx = np.mean(np.diff(self.x))

            # Initial parameters - only peaks, no background
            p0 = []
            bounds_lower = []
            bounds_upper = []

            for idx in self.selected_peaks:
                # Amplitude estimation
                amp_guess = max(self.y[idx], y_range * 0.01)
                cen_guess = self.x[idx]

                # Estimate sigma from local peak width
                half_max = self.y[idx] / 2
                left_idx = max(0, idx - 50)

                # Estimate width from nearby points
                local_width = dx * 10  # Default width
                for i in range(idx, left_idx, -1):
                    if self.y[i] < half_max:
                        local_width = abs(self.x[idx] - self.x[i]) * 2
                        break

                sig_guess = max(local_width / 2.355, dx * 2)
                gam_guess = sig_guess
                eta_guess = 0.5

                p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
                # Bounds based on data range
                bounds_lower.extend([0, self.x.min(), dx, dx, 0])
                bounds_upper.extend([y_range * 10, self.x.max(), x_range * 0.5, x_range * 0.5, 1.0])

            # Define fitting function for peaks only (no background)
            def multi_peak_only(x, *params):
                n_peaks = len(params) // 5
                y = np.zeros_like(x)
                for i in range(n_peaks):
                    offset = i * 5
                    amp, cen, sig, gam, eta = params[offset:offset+5]
                    y += pseudo_voigt(x, amp, cen, sig, gam, eta)
                return y

            # Perform fitting
            popt, pcov = curve_fit(multi_peak_only, self.x, self.y,
                                  p0=p0, bounds=(bounds_lower, bounds_upper),
                                  maxfev=10000)

            # Plot fit result
            x_smooth = np.linspace(self.x.min(), self.x.max(), 2000)
            y_fit = multi_peak_only(x_smooth, *popt)

            line1, = self.ax.plot(x_smooth, y_fit, color='#BA55D3', linewidth=2,
                                label='Fit', zorder=5)
            self.fit_lines.append(line1)

            # Extract fitting results and add text annotations
            n_peaks = len(self.selected_peaks)
            results = []

            info_msg = "Fitting Results:\n" + "="*50 + "\n"

            for i in range(n_peaks):
                offset = i * 5
                amp, cen, sig, gam, eta = popt[offset:offset+5]

                # Calculate metrics
                fwhm = calculate_fwhm(sig, gam, eta)
                area = calculate_area(amp, sig, gam, eta)

                # Add text annotation on plot
                peak_y = pseudo_voigt(cen, amp, cen, sig, gam, eta)
                text_annotation = self.ax.annotate(
                    f'P{i+1}\n2θ={cen:.3f}\nFWHM={fwhm:.4f}\nArea={area:.1f}',
                    xy=(cen, peak_y),
                    xytext=(cen + x_range*0.02, peak_y * 0.7),
                    fontsize=8,
                    color='#4B0082',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD',
                             edgecolor='#DAA520', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#DAA520', lw=1),
                    zorder=12
                )
                self.fit_lines.append(text_annotation)

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

        self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll to zoom/pan',
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

        # Clear peak-related items from undo stack
        self.undo_stack = [item for item in self.undo_stack if item[0] != 'peak']
        if not self.undo_stack:
            self.btn_undo.config(state=tk.DISABLED)

        if self.x is not None:
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll to zoom/pan',
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
