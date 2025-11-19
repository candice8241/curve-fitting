# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - Optimized Version
@author: candicewang928@gmail.com
Enhanced with better peak fitting, cleaner visualization, and improved performance
Modified: Only fits selected peaks with full tail coverage
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
from scipy.special import wofz
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

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile using Faddeeva function"""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

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

# ---------- Main GUI Application ----------
class PeakFittingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive XRD Peak Fitting Tool - Optimized")
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

        # Undo stack for tracking actions
        self.undo_stack = []

        # Fitting method: "pseudo_voigt" or "voigt"
        self.fit_method = tk.StringVar(value="pseudo_voigt")

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

        # Fit method selector
        tk.Label(bg_frame, text="Fit Method:",
                bg='#E6D5F5', fg='#4B0082',
                font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(20, 5), pady=10)

        fit_method_combo = ttk.Combobox(bg_frame, textvariable=self.fit_method,
                                        values=["pseudo_voigt", "voigt"],
                                        state="readonly", width=12)
        fit_method_combo.pack(side=tk.LEFT, padx=5, pady=8)

        # Coordinate display label in bg_frame
        self.coord_label = tk.Label(bg_frame, text="",
                                    bg='#E6D5F5', fg='#4B0082',
                                    font=('Courier', 9))
        self.coord_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Results display panel
        results_frame = tk.Frame(self.master, bg='#F5E6FF', height=120)
        results_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        results_frame.pack_propagate(False)

        results_label = tk.Label(results_frame, text="Fitting Results:",
                                bg='#F5E6FF', fg='#4B0082',
                                font=('Arial', 10, 'bold'))
        results_label.pack(side=tk.TOP, anchor='w', padx=10, pady=5)

        # Create Treeview for results table
        columns = ('Peak', '2theta', 'FWHM', 'Area', 'Amplitude', 'Sigma', 'Gamma', 'Eta')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=4)

        # Configure column headings and widths
        col_widths = {'Peak': 50, '2theta': 100, 'FWHM': 100, 'Area': 100,
                      'Amplitude': 100, 'Sigma': 80, 'Gamma': 80, 'Eta': 60}
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=col_widths.get(col, 80), anchor='center')

        # Add scrollbar
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Style the treeview
        style = ttk.Style()
        style.configure('Treeview', background='#FAF0FF', foreground='#4B0082',
                       font=('Courier', 9))
        style.configure('Treeview.Heading', font=('Arial', 9, 'bold'),
                       foreground='#4B0082')

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
                                  markersize=15, markeredgecolor='#FFD700',
                                  markeredgewidth=1.5, zorder=10)
            text = self.ax.text(point_x, point_y * 1.03, f'P{len(self.selected_peaks)+1}',
                               ha='center', fontsize=8, color='#FF1493',
                               fontweight='bold', zorder=11,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE4E1',
                                       edgecolor='#FF69B4', linewidth=1))

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
                                      markersize=15, markeredgecolor='#FFD700',
                                      markeredgewidth=1.5, zorder=10)
                text = self.ax.text(self.x[idx], self.y[idx] * 1.03, f'P{i+1}',
                                   ha='center', fontsize=8, color='#FF1493',
                                   fontweight='bold', zorder=11,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFE4E1',
                                           edgecolor='#FF69B4', linewidth=1))
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
        """Fit only selected peaks with full tail coverage, supports peak splitting"""
        if len(self.selected_peaks) == 0:
            messagebox.showwarning("No Peaks", "Please select at least one peak first!")
            return

        fit_method = self.fit_method.get()
        self.update_info(f"Fitting {len(self.selected_peaks)} peaks using {fit_method}...\n")
        self.status_label.config(text="Fitting in progress...")
        self.master.update()  # Update GUI to show status

        try:
            # Average data spacing
            dx = np.mean(np.diff(self.x))
            y_min = self.y.min()
            y_max = self.y.max()
            y_range = y_max - y_min

            # Sort peaks by position for better grouping detection
            sorted_indices = sorted(range(len(self.selected_peaks)),
                                   key=lambda i: self.x[self.selected_peaks[i]])
            sorted_peaks = [self.selected_peaks[i] for i in sorted_indices]

            # First pass: estimate FWHM for each peak
            fwhm_estimates = []
            is_shoulder = []  # Track if peak is likely a shoulder

            for idx in sorted_peaks:
                peak_height = self.y[idx] - y_min
                half_max = (self.y[idx] + y_min) / 2

                # Find left half maximum point
                left_idx = idx
                found_left = False
                for j in range(idx, max(0, idx - 100), -1):
                    if self.y[j] <= half_max:
                        left_idx = j
                        found_left = True
                        break

                # Find right half maximum point
                right_idx = idx
                found_right = False
                for j in range(idx, min(len(self.x), idx + 100)):
                    if self.y[j] <= half_max:
                        right_idx = j
                        found_right = True
                        break

                fwhm_estimate = abs(self.x[right_idx] - self.x[left_idx])

                # Check if this is likely a shoulder peak
                # (can't find both half-max points or FWHM is very small)
                if not (found_left and found_right) or fwhm_estimate < dx * 3:
                    is_shoulder.append(True)
                    # Use a default FWHM based on typical peak width
                    fwhm_estimate = dx * 10  # Default width for shoulders
                else:
                    is_shoulder.append(False)

                if fwhm_estimate < dx * 2:
                    fwhm_estimate = dx * 5
                fwhm_estimates.append(fwhm_estimate)

            # Detect overlapping peaks (peaks that should be fit together)
            # Group peaks that are within 3x FWHM of each other
            peak_groups = []
            current_group = [0]

            for i in range(1, len(sorted_peaks)):
                prev_idx = sorted_peaks[i-1]
                curr_idx = sorted_peaks[i]
                distance = abs(self.x[curr_idx] - self.x[prev_idx])
                avg_fwhm = (fwhm_estimates[i-1] + fwhm_estimates[i]) / 2

                if distance < avg_fwhm * 3:
                    # Peaks are close, add to current group
                    current_group.append(i)
                else:
                    # Start new group
                    peak_groups.append(current_group)
                    current_group = [i]

            peak_groups.append(current_group)

            # For grouped peaks, refine shoulder detection and FWHM estimates
            for group in peak_groups:
                if len(group) > 1:
                    # Find the main peak (highest intensity) in this group
                    group_peaks = [sorted_peaks[i] for i in group]
                    main_peak_local_idx = np.argmax([self.y[idx] for idx in group_peaks])
                    main_peak_fwhm = fwhm_estimates[group[main_peak_local_idx]]

                    # Mark other peaks as shoulders
                    for local_i, global_i in enumerate(group):
                        if local_i != main_peak_local_idx:
                            is_shoulder[global_i] = True
                            # Keep shoulder's own FWHM estimate if reasonable,
                            # otherwise use main peak's FWHM as reference
                            if fwhm_estimates[global_i] < dx * 3:
                                fwhm_estimates[global_i] = main_peak_fwhm * 1.0  # Same as main peak

            # Report overlapping peaks
            for group in peak_groups:
                if len(group) > 1:
                    # Map back to original peak numbers
                    original_nums = [sorted_indices[g] + 1 for g in group]
                    self.update_info(f"Peaks {original_nums} will be fit together (overlapping)\n")

            # Create combined fitting window based on groups
            fit_mask = np.zeros(len(self.x), dtype=bool)
            group_windows = []  # Store window for each group

            for group in peak_groups:
                # Find the extent of this group
                group_peaks = [sorted_peaks[i] for i in group]
                group_fwhms = [fwhm_estimates[i] for i in group]

                # Window extends from leftmost peak - 2*FWHM to rightmost peak + 2*FWHM
                left_center = self.x[min(group_peaks)]
                right_center = self.x[max(group_peaks)]
                left_fwhm = group_fwhms[group.index(group_peaks.index(min(group_peaks)))] if len(group) > 0 else group_fwhms[0]
                right_fwhm = group_fwhms[-1]

                window_left = left_center - left_fwhm * 2
                window_right = right_center + right_fwhm * 2

                # Find indices
                left_mask = self.x >= window_left
                right_mask = self.x <= window_right
                window_mask = left_mask & right_mask

                if np.any(window_mask):
                    left = np.argmax(window_mask)
                    right = len(self.x) - np.argmax(window_mask[::-1])
                    fit_mask[left:right] = True
                    group_windows.append((left, right))

            # Also create individual peak windows for plotting
            peak_windows = []
            for i, idx in enumerate(sorted_peaks):
                window_half = fwhm_estimates[i] * 2
                x_center = self.x[idx]

                left_mask = self.x >= (x_center - window_half)
                right_mask = self.x <= (x_center + window_half)
                window_mask = left_mask & right_mask

                if np.any(window_mask):
                    left = np.argmax(window_mask)
                    right = len(self.x) - np.argmax(window_mask[::-1])
                    peak_windows.append((left, right))
                else:
                    peak_windows.append((max(0, idx - 50), min(len(self.x), idx + 50)))

            # Extract data only within fitting regions
            x_fit = self.x[fit_mask]
            y_fit = self.y[fit_mask]

            # Initial parameters
            p0 = []
            bounds_lower = []
            bounds_upper = []

            use_voigt = (fit_method == "voigt")
            n_params_per_peak = 4 if use_voigt else 5

            for i, idx in enumerate(sorted_peaks):
                cen_guess = self.x[idx]
                fwhm_estimate = fwhm_estimates[i]
                sig_guess = fwhm_estimate / 2.355
                gam_guess = fwhm_estimate / 2

                # Estimate amplitude based on local height
                local_height = self.y[idx] - y_min

                # For shoulder/emerging peaks, use different strategy
                if is_shoulder[i]:
                    # Find the main peak in the same group
                    main_peak_height = y_range * 0.3  # Default
                    main_peak_amp = y_range * 0.3  # Default amplitude
                    for group in peak_groups:
                        if i in group:
                            group_peaks = [sorted_peaks[g] for g in group]
                            heights = [self.y[p] - y_min for p in group_peaks]
                            main_peak_height = max(heights)
                            main_peak_amp = main_peak_height
                            break

                    # For emerging peak: use a fraction of main peak amplitude
                    # This is more reliable than local height which is dominated by main peak
                    # Start with 30% of main peak as initial guess for shoulder
                    amp_guess = main_peak_amp * 0.3

                    # But also check if local height suggests something different
                    # If local height is significant, use it as a floor
                    local_based_guess = local_height * 0.4
                    if local_based_guess > amp_guess:
                        amp_guess = local_based_guess

                    # Minimum amplitude should be reasonable
                    min_amp = main_peak_height * 0.02  # At least 2% of main peak
                    if amp_guess < min_amp:
                        amp_guess = min_amp

                    # Wide amplitude bounds for shoulder - key for emerging peaks
                    amp_lower = y_range * 0.0001  # Very very small lower bound
                    amp_upper = main_peak_height * 1.5  # Can be up to 1.5x main peak
                    # User clicked position is the peak center - only allow small movement
                    center_tolerance = fwhm_estimate * 0.15  # Very tight: 15% of FWHM
                else:
                    amp_guess = local_height
                    if amp_guess <= 0:
                        amp_guess = y_range * 0.1

                    amp_lower = amp_guess * 0.01
                    amp_upper = amp_guess * 100
                    # For main peaks, also keep center close to clicked position
                    center_tolerance = fwhm_estimate * 0.2  # 20% of FWHM

                # Set sigma/gamma bounds - wider for shoulders to allow more shape flexibility
                if is_shoulder[i]:
                    sig_lower = dx * 0.05  # Allow narrower peaks for shoulders
                    gam_lower = dx * 0.05
                    sig_upper = fwhm_estimate * 8  # Allow wider range for shoulders
                    gam_upper = fwhm_estimate * 8
                else:
                    sig_lower = dx * 0.1
                    gam_lower = dx * 0.1
                    sig_upper = fwhm_estimate * 5
                    gam_upper = fwhm_estimate * 5

                if use_voigt:
                    p0.extend([amp_guess, cen_guess, sig_guess, gam_guess])
                    bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower])
                    bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper])
                else:
                    eta_guess = 0.5
                    p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
                    bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower, 0])
                    bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper, 1.0])

            # Define fitting function
            if use_voigt:
                def multi_peak_func(x, *params):
                    n_peaks = len(params) // 4
                    y = np.zeros_like(x)
                    for i in range(n_peaks):
                        offset = i * 4
                        amp, cen, sig, gam = params[offset:offset+4]
                        y += voigt(x, amp, cen, sig, gam)
                    return y
            else:
                def multi_peak_func(x, *params):
                    n_peaks = len(params) // 5
                    y = np.zeros_like(x)
                    for i in range(n_peaks):
                        offset = i * 5
                        amp, cen, sig, gam, eta = params[offset:offset+5]
                        y += pseudo_voigt(x, amp, cen, sig, gam, eta)
                    return y

            # Perform fitting with robust method
            # Use Trust Region Reflective with soft_l1 loss for better handling of overlapping peaks
            try:
                popt, pcov = curve_fit(multi_peak_func, x_fit, y_fit,
                                      p0=p0, bounds=(bounds_lower, bounds_upper),
                                      method='trf', loss='soft_l1',
                                      maxfev=100000)
            except Exception:
                # Fallback to default method if trf fails
                popt, pcov = curve_fit(multi_peak_func, x_fit, y_fit,
                                      p0=p0, bounds=(bounds_lower, bounds_upper),
                                      maxfev=100000)

            # Plot fit result for each group window
            plotted_regions = set()
            for left, right in group_windows:
                region_key = (left, right)
                if region_key in plotted_regions:
                    continue
                plotted_regions.add(region_key)

                x_region = self.x[left:right]
                x_smooth = np.linspace(x_region.min(), x_region.max(), 500)
                y_fit_smooth = multi_peak_func(x_smooth, *popt)

                if len(plotted_regions) == 1:
                    line1, = self.ax.plot(x_smooth, y_fit_smooth, color='#FF0000', linewidth=1.5,
                                        label='Fit', zorder=5, alpha=0.9)
                else:
                    line1, = self.ax.plot(x_smooth, y_fit_smooth, color='#FF0000', linewidth=1.5,
                                        zorder=5, alpha=0.9)
                self.fit_lines.append(line1)

            # Plot individual peak components - use group window range for better visibility
            for i in range(len(sorted_peaks)):
                offset = i * n_params_per_peak

                # Find which group this peak belongs to
                peak_group_idx = None
                for g_idx, group in enumerate(peak_groups):
                    if i in group:
                        peak_group_idx = g_idx
                        break

                # Use group window for plotting range
                if peak_group_idx is not None and peak_group_idx < len(group_windows):
                    left, right = group_windows[peak_group_idx]
                    x_peak_smooth = np.linspace(self.x[left], self.x[right], 500)
                else:
                    # Fallback to wider range (3x FWHM)
                    if use_voigt:
                        amp, cen, sig, gam = popt[offset:offset+4]
                        fwhm = 2.355 * sig
                    else:
                        amp, cen, sig, gam, eta = popt[offset:offset+5]
                        fwhm = calculate_fwhm(sig, gam, eta)
                    plot_range = fwhm * 3
                    x_peak_smooth = np.linspace(cen - plot_range, cen + plot_range, 500)
                    x_peak_smooth = x_peak_smooth[(x_peak_smooth >= self.x.min()) &
                                                   (x_peak_smooth <= self.x.max())]

                if use_voigt:
                    amp, cen, sig, gam = popt[offset:offset+4]
                    y_component = voigt(x_peak_smooth, amp, cen, sig, gam)
                else:
                    amp, cen, sig, gam, eta = popt[offset:offset+5]
                    y_component = pseudo_voigt(x_peak_smooth, amp, cen, sig, gam, eta)

                # Map back to original peak number
                original_idx = sorted_indices[i]
                line_comp, = self.ax.plot(x_peak_smooth, y_component, '--',
                                         linewidth=0.8, alpha=0.6, zorder=4,
                                         label=f'Peak {original_idx+1}')
                self.fit_lines.append(line_comp)

            # Extract fitting results
            n_peaks = len(sorted_peaks)
            results = []

            info_msg = f"Fitting Results ({fit_method}):\n" + "="*50 + "\n"

            for i in range(n_peaks):
                offset = i * n_params_per_peak
                original_idx = sorted_indices[i]

                if use_voigt:
                    amp, cen, sig, gam = popt[offset:offset+4]
                    fwhm = 2.355 * sig
                    area = amp * sig * np.sqrt(2 * np.pi)
                    eta = "N/A"

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
                else:
                    amp, cen, sig, gam, eta = popt[offset:offset+5]
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

                shoulder_note = " (shoulder)" if is_shoulder[i] else ""
                info_msg += f"Peak {original_idx+1}{shoulder_note}: 2theta={cen:.4f}, FWHM={fwhm:.5f}, Area={area:.1f}\n"

            # Sort results by original peak number
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

            self.ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
            self.ax.set_title(f'{self.filename} - Fit Complete ({fit_method})',
                            fontsize=14, fontweight='bold', color='#32CD32')
            self.canvas.draw()

            self.update_info(info_msg)
            self.status_label.config(text="Fitting successful!")

            self.btn_save.config(state=tk.NORMAL)
            self.btn_clear_fit.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
            self.update_info(f"Fitting failed: {str(e)}\n")
            self.status_label.config(text="Fitting failed")

    def clear_fit(self):
        """Clear fitting results but keep peak selections"""
        # Remove fit lines
        for line in self.fit_lines:
            try:
                line.remove()
            except:
                pass
        self.fit_lines = []

        self.fitted = False
        self.fit_results = None

        # Clear results table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

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

        # Clear results table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

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
