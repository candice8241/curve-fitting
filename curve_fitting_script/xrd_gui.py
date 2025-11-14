# -*- coding: utf-8 -*-
"""
Complete XRD Batch Processing Suite - GUI Application
Provides a graphical interface for XRD data processing including:
- 1D Integration from 2D patterns
- Peak fitting
- Volume calculation
- Birch-Murnaghan equation of state fitting

Created on Mon Nov 10 15:11:31 2025
@author: 16961
"""

import tkinter as tk
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
import re
import warnings
import math
import random
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from batch_integration import BatchIntegrator
from peak_fitting import BatchFitter
from birch_murnaghan_batch import BirchMurnaghanFitter
from batch_cal_volume import XRayDiffractionAnalyzer
from batch_appearance import ModernButton, ModernTab, CuteSheepProgressBar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')


class XRDProcessingGUI:
    """Main GUI application for XRD data processing"""

    def __init__(self, root):
        self.root = root
        self.root.title("XRD Data Post-Processing")
        self.root.geometry("1100x950")
        self.root.resizable(True, True)

        # Color scheme
        self.colors = {
            'bg': '#F8F7FF',
            'card_bg': '#FFFFFF',
            'primary': '#B794F6',
            'primary_hover': '#D4BBFF',
            'secondary': '#E0AAFF',
            'accent': '#FF6B9D',
            'text_dark': '#2B2D42',
            'text_light': '#8B8BA7',
            'border': '#E8E4F3',
            'success': '#06D6A0',
            'error': '#EF476F',
            'light_purple': '#E6D9F5',
            'active_module': '#C8B3E6'
        }

        self.root.configure(bg=self.colors['bg'])

        # Integration and fitting variables
        self.poni_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.input_pattern = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.dataset_path = tk.StringVar(value="entry/data/data")
        self.npt = tk.IntVar(value=4000)
        self.unit = tk.StringVar(value='2th_deg')
        self.fit_method = tk.StringVar(value='pseudo')

        # Phase analysis variables
        self.phase_peak_csv = tk.StringVar()
        self.phase_volume_csv = tk.StringVar()
        self.phase_volume_system = tk.StringVar(value='FCC')
        self.phase_volume_output = tk.StringVar()
        self.phase_wavelength = tk.DoubleVar(value=0.4133)
        self.phase_tolerance_1 = tk.DoubleVar(value=0.3)
        self.phase_tolerance_2 = tk.DoubleVar(value=0.4)
        self.phase_tolerance_3 = tk.DoubleVar(value=0.01)
        self.phase_n_points = tk.IntVar(value=4)

        # Birch-Murnaghan variables
        self.bm_input_file = tk.StringVar()
        self.bm_output_dir = tk.StringVar()
        self.bm_order = tk.StringVar(value='3')

        self.current_module = "integration"
        self.setup_ui()

    def setup_ui(self):
        """Setup the main user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(header_content, text="🐱", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Data Post-Processing",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 20, 'bold')).pack(side=tk.LEFT)

        # Tab bar with module selectors
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'])
        tab_frame.pack(fill=tk.X, padx=30, pady=(15, 0))

        # XRD type tabs
        xrd_tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        xrd_tabs_container.pack(side=tk.TOP, anchor=tk.W, pady=(0, 8))

        self.powder_tab = ModernTab(xrd_tabs_container, "Powder XRD",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 40))

        self.single_tab = ModernTab(xrd_tabs_container, "Single Crystal",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=(0, 40))

        self.radial_tab = ModernTab(xrd_tabs_container, "Radial",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT)

        # Module selector buttons
        module_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        module_container.pack(side=tk.TOP, anchor=tk.W)

        self.integration_module_btn = tk.Button(
            module_container,
            text="1D Integration & Peak Fitting",
            font=('Comic Sans MS', 10),
            bg=self.colors['active_module'] if self.current_module == "integration" else self.colors['light_purple'],
            fg=self.colors['text_dark'],
            activebackground=self.colors['primary_hover'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=10,
            command=lambda: self.show_module("integration")
        )
        self.integration_module_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.analysis_module_btn = tk.Button(
            module_container,
            text="Cal_Volume & BM_Fitting",
            font=('Comic Sans MS', 10),
            bg=self.colors['active_module'] if self.current_module == "analysis" else self.colors['light_purple'],
            fg=self.colors['text_dark'],
            activebackground=self.colors['primary_hover'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=10,
            command=lambda: self.show_module("analysis")
        )
        self.analysis_module_btn.pack(side=tk.LEFT)

        # Scrollable container
        container = tk.Frame(self.root, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        canvas = tk.Canvas(container, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        self.scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.setup_powder_content()

    def setup_powder_content(self):
        """Setup content for powder XRD tab"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Container for dynamic content
        self.dynamic_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        prog_cont = tk.Frame(main_frame, bg=self.colors['bg'])
        prog_cont.pack(fill=tk.X, pady=(15, 15))

        prog_inner = tk.Frame(prog_cont, bg=self.colors['bg'])
        prog_inner.pack(expand=True)

        self.progress = CuteSheepProgressBar(prog_inner, width=780, height=80)
        self.progress.pack()

        # Log area
        log_card = self.create_card_frame(main_frame)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_content = tk.Frame(log_card, bg=self.colors['card_bg'], padx=20, pady=12)
        log_content.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_content, bg=self.colors['card_bg'])
        log_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(log_header, text="🐰", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(log_header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(log_content, height=10, wrap=tk.WORD,
                                                  font=('Comic Sans MS', 10),
                                                  bg='#FAFAFA', fg='#B794F6',
                                                  relief='flat', borderwidth=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Show integration module by default
        self.show_module("integration")

    def show_module(self, module_type):
        """Switch between different processing modules"""
        self.current_module = module_type

        # Update button colors
        if module_type == "integration":
            self.integration_module_btn.config(bg=self.colors['active_module'])
            self.analysis_module_btn.config(bg=self.colors['light_purple'])
        else:
            self.integration_module_btn.config(bg=self.colors['light_purple'])
            self.analysis_module_btn.config(bg=self.colors['active_module'])

        # Clear dynamic frame
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        # Setup appropriate module
        if module_type == "integration":
            self.setup_integration_module()
        elif module_type == "analysis":
            self.setup_analysis_module()

    def setup_integration_module(self):
        """Setup integration and fitting module UI"""
        # Integration Settings
        integration_card = self.create_card_frame(self.dynamic_frame)
        integration_card.pack(fill=tk.X, pady=(0, 15))

        content1 = tk.Frame(integration_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content1.pack(fill=tk.BOTH, expand=True)

        header1 = tk.Frame(content1, bg=self.colors['card_bg'])
        header1.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header1, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header1, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content1, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker(content1, "Mask File", self.mask_path,
                               [("EDF files", "*.edf"), ("All files", "*.*")])
        self.create_file_picker(content1, "Input Pattern", self.input_pattern,
                               [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)
        self.create_folder_picker(content1, "Output Directory", self.output_dir)
        self.create_entry(content1, "Dataset Path", self.dataset_path)

        param_frame = tk.Frame(content1, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(5, 0))

        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                   width=18, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(unit_cont, textvariable=self.unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Fitting Settings
        fitting_card = self.create_card_frame(self.dynamic_frame)
        fitting_card.pack(fill=tk.X, pady=(0, 15))

        content2 = tk.Frame(fitting_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content2.pack(fill=tk.BOTH, expand=True)

        header2 = tk.Frame(content2, bg=self.colors['card_bg'])
        header2.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header2, text="🐹", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header2, text="Peak Fitting Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        fit_cont = tk.Frame(content2, bg=self.colors['card_bg'])
        fit_cont.pack(fill=tk.X)
        tk.Label(fit_cont, text="Fitting Method", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(fit_cont, textvariable=self.fit_method,
                    values=['pseudo', 'voigt'], width=22, state='readonly',
                    font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Buttons
        btn_frame = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        btn_cont = tk.Frame(btn_frame, bg=self.colors['bg'])
        btn_cont.pack(expand=True)

        btns = tk.Frame(btn_cont, bg=self.colors['bg'])
        btns.pack()

        ModernButton(btns, "Run Integration", self.run_integration, icon="🐿️",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=200, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(btns, "Run Fitting", self.run_fitting, icon="🐻",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=200, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(btns, "Full Pipeline", self.run_full_pipeline, icon="🦔",
                    bg_color=self.colors['primary'], hover_color=self.colors['accent'],
                    width=200, height=45).pack(side=tk.LEFT, padx=8)

    def setup_analysis_module(self):
        """Setup analysis module UI (volume calculation and BM fitting)"""
        # Phase Analysis Section
        phase_card = self.create_card_frame(self.dynamic_frame)
        phase_card.pack(fill=tk.X, pady=(0, 15))

        content3 = tk.Frame(phase_card, bg=self.colors['card_bg'], padx=15, pady=12)
        content3.pack(fill=tk.BOTH, expand=True)

        header3 = tk.Frame(content3, bg=self.colors['card_bg'])
        header3.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header3, text="🔬", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header3, text="Calculate Unit Cell Volume",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # Row 1: Peak CSV input with Run button and tolerance controls
        row1 = tk.Frame(content3, bg=self.colors['card_bg'])
        row1.pack(fill=tk.X, pady=(5, 8))

        peak_csv_frame = tk.Frame(row1, bg=self.colors['card_bg'])
        peak_csv_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        tk.Label(peak_csv_frame, text="Input CSV (Peak Data)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        peak_input_frame = tk.Frame(peak_csv_frame, bg=self.colors['card_bg'])
        peak_input_frame.pack(fill=tk.X)

        tk.Entry(peak_input_frame, textvariable=self.phase_peak_csv, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        ModernButton(peak_input_frame, "Browse",
                    lambda: self.browse_file(self.phase_peak_csv, [("CSV files", "*.csv")]),
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary'],
                    width=75, height=28).pack(side=tk.LEFT, padx=(5, 0))

        # Run button for separating peaks
        run_separate_frame = tk.Frame(peak_csv_frame, bg=self.colors['card_bg'])
        run_separate_frame.pack(fill=tk.X, pady=(5, 0))

        ModernButton(run_separate_frame, "Separate Original & New Peaks",
                    self.separate_peaks, icon="🔀",
                    bg_color="#06D6A0", hover_color="#05B88A",
                    width=260, height=35).pack(anchor=tk.W)

        # Tolerance and N controls (right side of row1)
        tolerance_frame = tk.Frame(row1, bg='#F0E6FF', relief='raised', borderwidth=2, padx=12, pady=10)
        tolerance_frame.pack(side=tk.LEFT)

        tk.Label(tolerance_frame, text="Peak Tolerances & N Points", bg='#F0E6FF',
                fg=self.colors['primary'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        tol_grid = tk.Frame(tolerance_frame, bg='#F0E6FF')
        tol_grid.pack(fill=tk.X)

        for i, (label, var) in enumerate([("Tol-1:", self.phase_tolerance_1),
                                          ("Tol-2:", self.phase_tolerance_2),
                                          ("Tol-3:", self.phase_tolerance_3)]):
            row = tk.Frame(tol_grid, bg='#F0E6FF')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg='#F0E6FF', font=('Comic Sans MS', 8), width=6).pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, font=('Comic Sans MS', 8), width=8).pack(side=tk.LEFT)

        # N Pressure Points
        n_row = tk.Frame(tol_grid, bg='#F0E6FF')
        n_row.pack(fill=tk.X, pady=(5, 0))
        tk.Label(n_row, text="N:", bg='#F0E6FF', font=('Comic Sans MS', 8), width=6).pack(side=tk.LEFT)
        ttk.Spinbox(n_row, from_=1, to=20, textvariable=self.phase_n_points,
                   width=8, font=('Comic Sans MS', 8)).pack(side=tk.LEFT)

        # Row 2: Volume CSV input with wavelength
        row2 = tk.Frame(content3, bg=self.colors['card_bg'])
        row2.pack(fill=tk.X, pady=(8, 0))

        volume_csv_frame = tk.Frame(row2, bg=self.colors['card_bg'])
        volume_csv_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        tk.Label(volume_csv_frame, text="Input CSV (Volume Calculation)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        volume_row = tk.Frame(volume_csv_frame, bg=self.colors['card_bg'])
        volume_row.pack(fill=tk.X)

        tk.Entry(volume_row, textvariable=self.phase_volume_csv, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        ModernButton(volume_row, "Browse",
                    lambda: self.browse_file(self.phase_volume_csv, [("CSV files", "*.csv")]),
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary'],
                    width=75, height=28).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Combobox(volume_row, textvariable=self.phase_volume_system,
                    values=['FCC', 'BCC', 'SC', 'Hexagonal', 'Tetragonal',
                           'Orthorhombic', 'Monoclinic', 'Triclinic'],
                    width=12, state='readonly', font=('Comic Sans MS', 9)).pack(side=tk.LEFT, padx=(10, 0))

        # Wavelength (right side of row2)
        wavelength_frame = tk.Frame(row2, bg='#F0E6FF', relief='raised', borderwidth=2, padx=12, pady=10)
        wavelength_frame.pack(side=tk.LEFT)

        tk.Label(wavelength_frame, text="Wavelength (Å)", bg='#F0E6FF',
                font=('Comic Sans MS', 9)).pack(anchor=tk.W, pady=(0, 2))
        tk.Entry(wavelength_frame, textvariable=self.phase_wavelength,
                font=('Comic Sans MS', 9), width=10).pack()

        # Row 3: Output directory
        self.create_folder_picker(content3, "Output Directory", self.phase_volume_output)

        # Phase analysis button
        btn_frame2 = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        btn_frame2.pack(fill=tk.X, pady=(10, 15))

        btn_cont2 = tk.Frame(btn_frame2, bg=self.colors['bg'])
        btn_cont2.pack(expand=True)

        ModernButton(btn_cont2, "Calculate Unit Cell Volume",
                    self.run_phase_analysis, icon="🔬",
                    bg_color="#06D6A0", hover_color="#05B88A",
                    width=280, height=45).pack()

        # Birch-Murnaghan Section
        bm_card = self.create_card_frame(self.dynamic_frame)
        bm_card.pack(fill=tk.X, pady=(0, 15))

        content4 = tk.Frame(bm_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content4.pack(fill=tk.BOTH, expand=True)

        header4 = tk.Frame(content4, bg=self.colors['card_bg'])
        header4.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header4, text="⚗️", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header4, text="Birch-Murnaghan EOS",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content4, "Input CSV (P-V Data)",
                               self.bm_input_file, [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_folder_picker(content4, "Output Directory", self.bm_output_dir)

        order_cont = tk.Frame(content4, bg=self.colors['card_bg'])
        order_cont.pack(fill=tk.X, pady=(5, 0))
        tk.Label(order_cont, text="BM Order", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(order_cont, textvariable=self.bm_order,
                    values=['2', '3'], width=18, state='readonly',
                    font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # BM button
        btn_frame3 = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        btn_frame3.pack(fill=tk.X, pady=(10, 0))

        btn_cont3 = tk.Frame(btn_frame3, bg=self.colors['bg'])
        btn_cont3.pack(expand=True)

        ModernButton(btn_cont3, "Birch-Murnaghan Fit",
                    self.run_birch_murnaghan, icon="⚗️",
                    bg_color="#FF6B9D", hover_color="#FF8FB3",
                    width=250, height=45).pack()

    def create_card_frame(self, parent, bg=None, **kwargs):
        """Create a styled card frame"""
        if bg is None:
            bg = self.colors['card_bg']
        card = tk.Frame(parent, bg=bg, relief='flat', borderwidth=0, **kwargs)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        return card

    def create_file_picker(self, parent, label, variable, filetypes, pattern=False):
        """Create a file picker widget"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if pattern:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_pattern(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        else:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_file(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_folder_picker(self, parent, label, variable):
        """Create a folder picker widget"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        btn = ModernButton(input_frame, "Browse",
                         lambda: self.browse_folder(variable),
                         bg_color=self.colors['secondary'],
                         hover_color=self.colors['primary'],
                         width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_entry(self, parent, label, variable):
        """Create a text entry widget"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        entry = tk.Entry(container, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1)
        entry.pack(fill=tk.X, ipady=3)

    def browse_file(self, variable, filetypes):
        """Open file browser dialog"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            variable.set(filename)

    def browse_pattern(self, variable, filetypes):
        """Open file browser and create pattern"""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            folder = os.path.dirname(filename)
            ext = os.path.splitext(filename)[1]
            pattern = os.path.join(folder, f"*{ext}")
            variable.set(pattern)

    def browse_folder(self, variable):
        """Open folder browser dialog"""
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def log(self, message):
        """Add message to log"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def show_success(self, message):
        """Show success dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Success")
        dialog.geometry("450x300")
        dialog.configure(bg=self.colors['card_bg'])
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="✅", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 64)).pack(pady=(30, 20))

        tk.Label(dialog, text=message, bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Comic Sans MS', 13, 'bold'),
                wraplength=400).pack(pady=(10, 30))

        ModernButton(dialog, "OK", dialog.destroy,
                    bg_color=self.colors['primary'],
                    hover_color=self.colors['primary_hover'],
                    width=120, height=40).pack()

    def switch_tab(self, tab_name):
        """Switch between XRD type tabs"""
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if tab_name == "powder":
            self.setup_powder_content()
        else:
            self.setup_placeholder(tab_name.replace("_", " ").title(), "Coming soon...")

    def setup_placeholder(self, title, message):
        """Setup placeholder for unimplemented features"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, pady=50)

        card = self.create_card_frame(main_frame)
        card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=50, pady=50)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="🔬", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 48)).pack(pady=(0, 20))

        tk.Label(content, text=title, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 20, 'bold')).pack(pady=(0, 10))

        tk.Label(content, text=message, bg=self.colors['card_bg'],
                fg=self.colors['text_light'], font=('Comic Sans MS', 12)).pack()

    # Processing functions
    def separate_peaks(self):
        """Separate original and new peaks from CSV data"""
        if not self.phase_peak_csv.get():
            messagebox.showerror("Error", "Please select peak CSV file first")
            return
        threading.Thread(target=self._separate_peaks_thread, daemon=True).start()

    def _separate_peaks_thread(self):
        """Background thread for peak separation"""
        try:
            self.progress.start()
            self.log("🔀 Separating original and new peaks...")

            csv_path = self.phase_peak_csv.get()
            output_dir = os.path.dirname(csv_path)

            # Create analyzer instance
            analyzer = XRayDiffractionAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get(),
                csv_path=csv_path
            )

            # Read and process data
            df = pd.read_csv(csv_path)
            pressure_data = analyzer.read_pressure_peak_data(csv_path)
            pressures = sorted(pressure_data.keys())

            transition_pressure, before_pressures, after_pressures = analyzer.find_phase_transition_point(
                pressure_data=pressure_data, tolerance=analyzer.peak_tolerance_1)

            # Identify original and new peaks
            original_peaks = analyzer.build_original_peak_dataset(
                pressure_data, tracked_new_peak_dataset=None,
                tolerance=analyzer.peak_tolerance_3)

            new_peaks = analyzer.collect_tracked_new_peaks(
                pressure_data, transition_pressure,
                after_pressures, new_peaks_ref=None, tolerance=analyzer.peak_tolerance_2)

            # Save original peaks
            original_csv = os.path.join(output_dir, "original_peaks.csv")
            if 'Peak Index' in df.columns:
                df_original = df[df['Peak Index'].isin(original_peaks)]
            else:
                df_original = df.head(len(original_peaks))
            df_original.to_csv(original_csv, index=False)
            self.log(f"✅ Original peaks saved: {original_csv}")

            # Save new peaks
            new_csv = os.path.join(output_dir, "new_peaks.csv")
            if 'Peak Index' in df.columns:
                df_new = df[df['Peak Index'].isin(new_peaks)]
            else:
                df_new = df.tail(len(new_peaks))
            df_new.to_csv(new_csv, index=False)
            self.log(f"✅ New peaks saved: {new_csv}")

            self.log(f"📊 Original peaks: {len(original_peaks)}, New peaks: {len(new_peaks)}")
            if transition_pressure:
                self.log(f"⚡ Phase transition detected at: {transition_pressure} GPa")

            self.show_success("Peak separation completed!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_integration(self):
        """Start integration process"""
        if not self.poni_path.get() or not self.mask_path.get() or not self.input_pattern.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_integration_thread, daemon=True).start()

    def _run_integration_thread(self):
        """Background thread for integration"""
        try:
            self.progress.start()
            self.log("🔁 Starting Batch Integration...")
            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=self.dataset_path.get() or None
            )
            self.log("✅ Integration completed!")
            self.show_success("Integration completed successfully!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_fitting(self):
        """Start peak fitting process"""
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please specify output directory")
            return
        threading.Thread(target=self._run_fitting_thread, daemon=True).start()

    def _run_fitting_thread(self):
        """Background thread for fitting"""
        try:
            self.progress.start()
            self.log("📈 Starting Batch Peak Fitting...")
            fitter = BatchFitter(folder=self.output_dir.get(), fit_method=self.fit_method.get())
            fitter.run_batch_fitting()
            self.log("✅ Fitting completed!")
            self.show_success("Peak fitting completed successfully!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_full_pipeline(self):
        """Run full integration and fitting pipeline"""
        if not self.poni_path.get() or not self.mask_path.get() or not self.input_pattern.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_full_pipeline_thread, daemon=True).start()

    def _run_full_pipeline_thread(self):
        """Background thread for full pipeline"""
        try:
            self.progress.start()
            self.log("🔁 Step 1/2: Batch Integration...")
            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=self.dataset_path.get() or None
            )
            self.log("✅ Integration complete!")

            self.log("📈 Step 2/2: Peak Fitting...")
            fitter = BatchFitter(folder=self.output_dir.get(), fit_method=self.fit_method.get())
            fitter.run_batch_fitting()
            self.log("✅ Full pipeline completed!")
            self.show_success("Full pipeline completed successfully!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_phase_analysis(self):
        """Start phase analysis and volume calculation"""
        if not self.phase_volume_csv.get() or not self.phase_volume_output.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_phase_analysis_thread, daemon=True).start()

    def _run_phase_analysis_thread(self):
        """Background thread for phase analysis"""
        try:
            self.progress.start()
            self.log("🔬 Starting Volume Calculation...")

            # Create analyzer instance
            analyzer = XRayDiffractionAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get()
            )

            # Read peak data
            pressure_data = analyzer.read_pressure_peak_data(self.phase_volume_csv.get())
            self.log(f"📊 Loaded data for {len(pressure_data)} pressure points")

            # Calculate volumes
            results_df = analyzer.calculate_volumes_from_peaks(
                pressure_data,
                crystal_system=self.phase_volume_system.get(),
                output_dir=self.phase_volume_output.get()
            )

            self.log(f"✅ Volume calculation completed!")
            self.log(f"📈 Calculated {len(results_df)} P-V data points")
            self.log(f"💾 Results saved to: {self.phase_volume_output.get()}")

            self.show_success("Volume calculation completed successfully!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_birch_murnaghan(self):
        """Start Birch-Murnaghan fitting"""
        if not self.bm_input_file.get() or not self.bm_output_dir.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_birch_murnaghan_thread, daemon=True).start()

    def _run_birch_murnaghan_thread(self):
        """Background thread for BM fitting"""
        try:
            self.progress.start()
            order = int(self.bm_order.get())
            order_str = f"{order}rd order" if order == 3 else "2nd order"
            self.log(f"⚗️ Starting {order_str} Birch-Murnaghan Fitting...")

            # Use BirchMurnaghanFitter class
            fitter = BirchMurnaghanFitter(
                output_dir=self.bm_output_dir.get(),
                order=order
            )

            # Load data
            data = pd.read_csv(self.bm_input_file.get())
            P = data.iloc[:, 0].values
            V = data.iloc[:, 1].values

            self.log(f"📊 Loaded {len(P)} pressure-volume data points")

            # Perform fitting
            results = fitter.fit(P=P, V=V)

            self.log(f"\n✅ BM fitting completed!")
            self.log(f"📊 V₀ = {results['V0']:.4f} ± {results['V0_err']:.4f} Å³")
            self.log(f"📊 K₀ = {results['K0']:.2f} ± {results['K0_err']:.2f} GPa")

            if order == 3:
                self.log(f"📊 K₀' = {results['K0_prime']:.3f} ± {results['K0_prime_err']:.3f}")
            else:
                self.log(f"📊 K₀' = 4.0 (fixed)")

            self.log(f"📈 R² = {results['r_squared']:.6f}")
            self.log(f"💾 Results saved to: {self.bm_output_dir.get()}")

            self.show_success(f"{order_str} BM fitting completed successfully!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
