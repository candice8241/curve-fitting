# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - Main Application Module

@author: candicewang928@gmail.com
Enhanced with better peak fitting algorithm and group-based background fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import warnings

# Import from our modular structure
from .smoothing import apply_smoothing
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

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class PeakFittingGUI:
    """
    Interactive GUI for XRD peak fitting with advanced features.

    Features:
    - Interactive peak selection
    - Automatic peak detection
    - Background subtraction (manual and automatic)
    - Data smoothing (Gaussian and Savitzky-Golay)
    - Group-based peak fitting using DBSCAN clustering
    - Voigt and Pseudo-Voigt profile fitting
    - Batch file processing
    """

    def __init__(self, master):
        """Initialize the GUI application."""
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
        self.y_smoothed = None

        # Original data storage
        self.y_original = None

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

        # File navigation buttons
        nav_btn_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 3,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 3
        }

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

    # File loading and navigation methods
    def load_file(self):
        """Load XRD data file via file dialog"""
        filepath = filedialog.askopenfilename(
            title="Select XRD Data File",
            filetypes=[("XY files", "*.xy"), ("DAT files", "*.dat"),
                       ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return

        # Scan folder for all compatible files
        self._scan_folder(filepath)

        # Load the selected file
        self.load_file_by_path(filepath)

    def _scan_folder(self, filepath):
        """Scan folder and build file list for navigation"""
        folder = os.path.dirname(filepath)
        all_files = []

        # Collect all compatible files in the folder
        extensions = ['.xy', '.dat', '.txt']
        for file in sorted(os.listdir(folder)):
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(folder, file)
                if os.path.isfile(full_path):
                    all_files.append(full_path)

        self.file_list = all_files
        # Find current file index
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

            self.btn_fit.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_select_bg.config(state=tk.NORMAL)
            self.btn_clear_bg.config(state=tk.NORMAL)
            self.btn_auto_bg.config(state=tk.NORMAL)
            self.btn_auto_find.config(state=tk.NORMAL)
            self.btn_overlap_mode.config(state=tk.NORMAL)
            self.btn_apply_smooth.config(state=tk.NORMAL)
            self.btn_reset_smooth.config(state=tk.NORMAL)

            # Enable navigation buttons if there are multiple files
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

    # Mouse interaction methods
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

            elif event.button == 3:  # Right click - remove nearest point
                if len(self.bg_points) > 0:
                    point_distances = [abs(x_click - p[0]) for p in self.bg_points]
                    nearest_point_idx = np.argmin(point_distances)
                    min_point_dist = point_distances[nearest_point_idx]

                    # Also check distance to line segments
                    min_segment_dist = float('inf')
                    nearest_segment_point_idx = None

                    if len(self.bg_points) >= 2:
                        sorted_points = sorted(enumerate(self.bg_points), key=lambda x: x[1][0])
                        for i in range(len(sorted_points) - 1):
                            idx1, (x1, y1) = sorted_points[i]
                            idx2, (x2, y2) = sorted_points[i + 1]

                            if x1 <= x_click <= x2:
                                t = (x_click - x1) / (x2 - x1) if x2 != x1 else 0
                                y_on_segment = y1 + t * (y2 - y1)
                                y_dist = abs(point_y - y_on_segment)

                                if y_dist < min_segment_dist:
                                    min_segment_dist = y_dist
                                    dist_to_start = abs(x_click - x1)
                                    dist_to_end = abs(x_click - x2)
                                    nearest_segment_point_idx = idx1 if dist_to_start < dist_to_end else idx2

                    y_range = np.max(self.y) - np.min(self.y)
                    segment_threshold = y_range * 0.05

                    if nearest_segment_point_idx is not None and min_segment_dist < segment_threshold:
                        delete_idx = nearest_segment_point_idx
                    else:
                        delete_idx = nearest_point_idx

                    removed_point = self.bg_points.pop(delete_idx)
                    marker_tuple = self.bg_markers.pop(delete_idx)

                    try:
                        marker_tuple[0].remove()
                        marker_tuple[1].remove()
                    except:
                        pass

                    for i, (marker, text) in enumerate(self.bg_markers):
                        text.set_text(f'BG{i+1}')

                    self.update_bg_connect_line()
                    self.canvas.draw()

                    self.update_info(f"BG point removed at 2theta = {removed_point[0]:.4f}\n")

                    if len(self.bg_points) < 2:
                        self.btn_subtract_bg.config(state=tk.DISABLED)
        elif not self.fitted:
            if event.button == 1:  # Left click - add peak
                search_window = max(5, 10)

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

            elif event.button == 3:  # Right click - remove nearest peak
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

    # Background methods
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
        """Automatically select background points across the data range"""
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        try:
            self.clear_background()

            n_points = 10
            bg_points = find_background_points_auto(self.x, self.y, n_points=n_points)

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

    # Undo and reset methods
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

    #  Auto peak finding
    def auto_find_peaks(self):
        """Automatically find all peaks in the data using scipy.signal.find_peaks"""
        if self.x is None or self.y is None:
            messagebox.showwarning("No Data", "Please load a file first!")
            return

        self.reset_peaks()

        try:
            # Smooth data for better peak detection
            from scipy.signal import savgol_filter
            if len(self.y) > 15:
                window_length = min(15, len(self.y) // 2 * 2 + 1)
                y_smooth = savgol_filter(self.y, window_length, 3)
            else:
                y_smooth = self.y

            # Calculate adaptive thresholds
            y_range = np.max(self.y) - np.min(self.y)
            y_std = np.std(self.y)
            dx = np.mean(np.diff(self.x))

            height_threshold = np.min(self.y) + y_range * 0.05
            prominence_threshold = y_range * 0.02
            min_distance = max(5, int(0.1 / dx)) if dx > 0 else 5

            peaks, properties = find_peaks(
                y_smooth,
                height=height_threshold,
                prominence=prominence_threshold,
                distance=min_distance,
                width=2
            )

            if len(peaks) == 0:
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

            # Filter peaks - above local baseline
            filtered_peaks = []
            for idx in peaks:
                window = 40
                left = max(0, idx - window)
                right = min(len(self.y), idx + window)

                edge_n = max(3, (right - left) // 10)
                local_baseline = (np.mean(self.y[left:left+edge_n]) +
                                 np.mean(self.y[right-edge_n:right])) / 2

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

            self.update_info(f"Auto-detected {len(peaks)} peaks\n")
            self.status_label.config(text=f"{len(peaks)} peaks auto-detected")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Auto peak detection failed:\n{str(e)}")
            self.update_info(f"Auto detection error: {traceback.format_exc()}\n")

    # Smoothing methods
    def on_smoothing_changed(self):
        """Called when smoothing settings change"""
        pass

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

    # Utility methods
    def update_info(self, message):
        """Update info text"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    # Peak fitting method
    def fit_peaks(self):
        """Fit selected peaks using group-based fitting"""
        from .gui_fitting import fit_peaks_method
        fit_peaks_method(self)

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

    # Save methods
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
            # Save to the same directory as the source file
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


def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = PeakFittingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
