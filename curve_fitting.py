# -*- coding: utf-8 -*-
"""
Complete XRD Batch Processing Suite - Corrected Version
Fixed Birch-Murnaghan parameter issue
Created on Mon Nov 10 15:11:31 2025

@author: felicity
"""

import tkinter as tk
from tkinter import font as tkFont
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
import re
import sys
import ctypes
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
from batch_cal_volume import XRayDiffractionAnalyzer as XRDAnalyzer
from batch_appearance import ModernButton, ModernTab, CuteSheepProgressBar
from azimuthal_integration import AzimuthalIntegrator, get_preset_sectors
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil

warnings.filterwarnings('ignore')



class XRDProcessingGUI:
    def __init__(self, root):
        self.root = root

        self.root.title("XRD Data Post-Processing")
        self.root.geometry("1100x950")
        self.root.resizable(True, True)

        try:
            icon_path = r'D:\HEPS\ID31\dioptas_data\github_felicity\batch\ChatGPT Image.ico'  # 替换为你的图标路径
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"无法加载图标: {e}")

        # Color scheme definition
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

        # Radial XRD (Azimuthal Integration) variables
        self.radial_poni_path = tk.StringVar()
        self.radial_mask_path = tk.StringVar()
        self.radial_input_pattern = tk.StringVar()
        self.radial_output_dir = tk.StringVar()
        self.radial_dataset_path = tk.StringVar(value="entry/data/data")
        self.radial_npt = tk.IntVar(value=4000)
        self.radial_unit = tk.StringVar(value='2th_deg')
        self.radial_azimuth_start = tk.DoubleVar(value=0.0)
        self.radial_azimuth_end = tk.DoubleVar(value=90.0)
        self.radial_sector_label = tk.StringVar(value="Sector_1")
        self.radial_preset = tk.StringVar(value='custom')
        self.radial_mode = tk.StringVar(value='single')  # 'single' or 'multiple'

        self.current_module = "integration"
        self.setup_ui()


    def setup_ui(self):
        """Setup main user interface"""
        # Header section
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(header_content, text="🐱", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Complete Suite",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 20, 'bold')).pack(side=tk.LEFT)

        # Tab bar with increased spacing
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        tab_frame.pack(fill=tk.X, padx=30, pady=(15, 0))

        tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        tabs_container.pack(side=tk.LEFT)

        self.powder_tab = ModernTab(tabs_container, "Powder XRD",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 15))

        self.single_tab = ModernTab(tabs_container, "Single Crystal",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=15)

        self.radial_tab = ModernTab(tabs_container, "Radial",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT, padx=15)

        # Scrollable container setup
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
        """Setup powder XRD tab content"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Module selector buttons directly under Powder XRD tab
        module_frame = tk.Frame(main_frame, bg=self.colors['bg'], height=60)
        module_frame.pack(fill=tk.X, pady=(5, 15))

        btn_container = tk.Frame(module_frame, bg=self.colors['bg'])
        btn_container.pack()

        self.integration_module_btn = tk.Button(
            btn_container,
            text="1D Integration & Peak Fitting",
            font=('Comic Sans MS', 10),
            bg=self.colors['active_module'] if self.current_module == "integration" else self.colors['light_purple'],
            fg=self.colors['text_dark'],
            activebackground=self.colors['primary_hover'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=12,
            command=lambda: self.show_module("integration")
        )
        self.integration_module_btn.pack(side=tk.LEFT, padx=8)

        self.analysis_module_btn = tk.Button(
            btn_container,
            text="Cal_Volume & BM_Fitting",
            font=('Comic Sans MS', 10),
            bg=self.colors['active_module'] if self.current_module == "analysis" else self.colors['light_purple'],
            fg=self.colors['text_dark'],
            activebackground=self.colors['primary_hover'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=12,
            command=lambda: self.show_module("analysis")
        )
        self.analysis_module_btn.pack(side=tk.LEFT, padx=8)

        # Container for dynamic content
        self.dynamic_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        # Progress bar section
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

        # Update button colors to reflect active module
        if module_type == "integration":
            self.integration_module_btn.config(bg=self.colors['active_module'])
            self.analysis_module_btn.config(bg=self.colors['light_purple'])
        else:
            self.integration_module_btn.config(bg=self.colors['light_purple'])
            self.analysis_module_btn.config(bg=self.colors['active_module'])

        # Clear existing content
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        # Load appropriate module content
        if module_type == "integration":
            self.setup_integration_module()
        elif module_type == "analysis":
            self.setup_analysis_module()

    def setup_integration_module(self):
        """Setup integration and peak fitting module UI"""
        # Integration Settings Card
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

        # Fitting Settings Card
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

        # Action Buttons
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
        """Setup phase analysis and Birch-Murnaghan fitting module UI"""
        # Phase Analysis Section
        phase_card = self.create_card_frame(self.dynamic_frame)
        phase_card.pack(fill=tk.X, pady=(0, 15))

        content3 = tk.Frame(phase_card, bg=self.colors['card_bg'], padx=10, pady=10)
        content3.pack(fill=tk.BOTH, expand=True)

        header3 = tk.Frame(content3, bg=self.colors['card_bg'])
        header3.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header3, text="🔬", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header3, text="Phase Transition Analysis & Volume Calculation",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # Row 1: Peak CSV input with Run button and tolerance controls
        row1 = tk.Frame(content3, bg=self.colors['card_bg'])
        row1.pack(fill=tk.X, anchor=tk.W, pady=(5, 8))

        peak_csv_frame = tk.Frame(row1, bg=self.colors['card_bg'])
        peak_csv_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

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
                    self.separate_peaks, icon="🐶",
                    bg_color="#FFB6C1", hover_color="#9966CC",
                    width=350, height=45).pack(anchor=tk.E, padx=200)

        # Tolerance and N controls (right side of row1)
        tolerance_frame = tk.Frame(row1, bg='#F0E6FF', relief='raised', borderwidth=2, padx=10, pady=10)
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
        volume_csv_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

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

        ModernButton(btn_cont2, "Calculate Volume & Fit Lattice Parameters",
                    self.run_phase_analysis, icon="🦊",
                    bg_color="#FFBCD9", hover_color="#7851A9",
                    width=400, height=45).pack()

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
        """Create a file picker widget with browse button"""
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
        """Create a folder picker widget with browse button"""
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
        """Open file browser and create pattern from selected file"""
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
        """Log message to the log text widget"""
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
        """Switch between main tabs"""
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if tab_name == "powder":
            self.setup_powder_content()
        elif tab_name == "radial":
            self.setup_radial_content()
        else:
            self.setup_placeholder(tab_name.replace("_", " ").title(), "Coming soon...")

    def setup_placeholder(self, title, message):
        """Setup placeholder content for unimplemented tabs"""
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

    # ==================== Processing Functions ====================

    def separate_peaks(self):
        """Separate original and new peaks from input CSV"""
        if not self.phase_peak_csv.get():
            messagebox.showerror("Error", "Please select peak CSV file first")
            return
        threading.Thread(target=self._separate_peaks_thread, daemon=True).start()

    def _separate_peaks_thread(self):
        """Background thread for peak separation"""
        try:
            self.progress.start()
            self.log("🔀 Starting peak separation process...")

            csv_path = self.phase_peak_csv.get()

            # Initialize XRayDiffractionAnalyzer with GUI parameters
            analyzer = XRDAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get()
            )

            # Read pressure-peak data
            self.log(f"📄 Reading data from: {os.path.basename(csv_path)}")
            pressure_data = analyzer.read_pressure_peak_data(csv_path)
            self.log(f"✓ Loaded {len(pressure_data)} pressure points")

            # Find phase transition point
            self.log("🔍 Identifying phase transition...")
            transition_pressure, before_pressures, after_pressures = analyzer.find_phase_transition_point()

            if transition_pressure is None:
                self.log("⚠️ No phase transition detected")
                messagebox.showwarning("Warning", "No phase transition detected in the data")
                return

            self.log(f"✓ Phase transition detected at {transition_pressure:.2f} GPa")

            # Identify new peaks at transition
            transition_peaks = pressure_data[transition_pressure]
            prev_pressure = before_pressures[-1]
            prev_peaks = pressure_data[prev_pressure]

            tolerance_windows = [(p - analyzer.peak_tolerance_1, p + analyzer.peak_tolerance_1)
                                for p in prev_peaks]
            new_peaks_at_transition = []

            for peak in transition_peaks:
                in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
                if not in_any_window:
                    new_peaks_at_transition.append(peak)

            self.log(f"✓ Found {len(new_peaks_at_transition)} new peaks at transition")

            # Generate output file paths
            base_filename = csv_path.replace('.csv', '')
            new_peaks_dataset_csv = f"{base_filename}_new_peaks_dataset.csv"
            original_peaks_dataset_csv = f"{base_filename}_original_peaks_dataset.csv"

            # Collect tracked new peaks and export to CSV
            self.log("📊 Tracking new peaks across pressure points...")
            stable_count, tracked_new_peaks = analyzer.collect_tracked_new_peaks(
                pressure_data,
                transition_pressure,
                after_pressures,
                new_peaks_at_transition,
                analyzer.peak_tolerance_2,
                output_csv=new_peaks_dataset_csv
            )

            self.log(f"✓ {stable_count} stable new peaks identified")
            self.log(f"💾 New peaks dataset saved to: {os.path.basename(new_peaks_dataset_csv)}")

            # Build original peaks dataset and export to CSV
            self.log("📊 Building original peaks dataset...")
            original_peak_dataset = analyzer.build_original_peak_dataset(
                pressure_data,
                tracked_new_peaks,
                analyzer.peak_tolerance_3,
                output_csv=original_peaks_dataset_csv
            )

            self.log(f"✓ Original peaks dataset constructed for {len(original_peak_dataset)} pressure points")
            self.log(f"💾 Original peaks dataset saved to: {os.path.basename(original_peaks_dataset_csv)}")

            # Summary
            self.log("\n" + "="*60)
            self.log("✅ Peak separation completed successfully!")
            self.log("="*60)
            self.log(f"📍 Transition pressure: {transition_pressure:.2f} GPa")
            self.log(f"📊 New peaks CSV: {os.path.basename(new_peaks_dataset_csv)}")
            self.log(f"📊 Original peaks CSV: {os.path.basename(original_peaks_dataset_csv)}")
            self.log("="*60 + "\n")

            self.show_success(f"Peak separation completed!\n\n"
                            f"Transition at {transition_pressure:.2f} GPa\n"
                            f"Files saved to input directory")

        except Exception as e:
            self.log(f"❌ Error during peak separation: {str(e)}")
            messagebox.showerror("Error", f"Peak separation failed:\n{str(e)}")
        finally:
            self.progress.stop()

    def run_integration(self):
        """Run 1D integration"""
        if not self.poni_path.get() or not self.mask_path.get() or not self.input_pattern.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_integration_thread, daemon=True).start()

    def _run_integration_thread(self):
        """Background thread for integration"""
        try:
            self.progress.start()
            self.log("🔁 Starting Batch Integration")
            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=self.dataset_path.get() or None
            )
            self.log("✅ Integration completed!")
            self.show_success("Integration completed!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_fitting(self):
        """Run peak fitting"""
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please specify output directory")
            return
        threading.Thread(target=self._run_fitting_thread, daemon=True).start()

    def _run_fitting_thread(self):
        """Background thread for peak fitting"""
        try:
            self.progress.start()
            self.log("📈 Starting Batch Fitting")
            fitter = BatchFitter(folder=self.output_dir.get(), fit_method=self.fit_method.get())
            fitter.run_batch_fitting()
            self.log("✅ Fitting completed!")
            self.show_success("Fitting completed!")
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
            self.log("🔁 Step 1/2: Integration")
            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get()
            )
            self.log("✅ Integration done")

            self.log("📈 Step 2/2: Fitting")
            fitter = BatchFitter(folder=self.output_dir.get(), fit_method=self.fit_method.get())
            fitter.run_batch_fitting()
            self.log("✅ Pipeline completed!")
            self.show_success("Full pipeline completed!")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress.stop()

    def run_phase_analysis(self):
        """Run volume calculation and lattice parameter fitting"""
        if not self.phase_volume_csv.get() or not self.phase_volume_output.get():
            messagebox.showerror("Error", "Please fill all required fields (Input CSV and Output Directory)")
            return
        threading.Thread(target=self._run_phase_analysis_thread, daemon=True).start()

    def _run_phase_analysis_thread(self):
        """Background thread for phase analysis and volume calculation"""
        try:
            self.progress.start()
            self.log("🔬 Starting Volume Calculation & Lattice Parameter Fitting")

            csv_path = self.phase_volume_csv.get()
            output_dir = self.phase_volume_output.get()

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Map GUI crystal system selection to analyzer format
            system_mapping = {
                'FCC': 'cubic_FCC',
                'BCC': 'cubic_BCC',
                'SC': 'cubic_SC',
                'Hexagonal': 'Hexagonal',
                'Tetragonal': 'Tetragonal',
                'Orthorhombic': 'Orthorhombic',
                'Monoclinic': 'Monoclinic',
                'Triclinic': 'Triclinic'
            }

            crystal_system = system_mapping.get(self.phase_volume_system.get(), 'cubic_FCC')

            self.log(f"📄 Input CSV: {os.path.basename(csv_path)}")
            self.log(f"🔷 Crystal system: {self.phase_volume_system.get()}")
            self.log(f"📏 Wavelength: {self.phase_wavelength.get()} Å")
            self.log(f"📁 Output directory: {output_dir}")

            # Initialize analyzer with GUI parameters
            analyzer = XRDAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get()
            )

            self.log("\n" + "="*60)
            self.log("Starting analysis...")
            self.log("="*60 + "\n")

            # Call the analyze function in auto mode
            results = analyzer.analyze(
                csv_path=csv_path,
                original_system=crystal_system,
                new_system=crystal_system,
                auto_mode=True
            )

            if results is None:
                self.log("❌ Analysis failed - no results returned")
                messagebox.showerror("Error", "Analysis failed to complete")
                return

            # Move generated output files to specified output directory
            input_dir = os.path.dirname(csv_path)
            base_filename = os.path.splitext(os.path.basename(csv_path))[0]

            # Look for generated files and move them
            generated_files = []

            # Check for different result file patterns
            possible_files = [
                f"{base_filename}_original_peaks_lattice.csv",
                f"{base_filename}_new_peaks_lattice.csv",
                f"{base_filename}_lattice_results.csv",
                f"{base_filename}_new_peaks_dataset.csv",
                f"{base_filename}_original_peaks_dataset.csv"
            ]

            for filename in possible_files:
                source_path = os.path.join(input_dir, filename)
                if os.path.exists(source_path):
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    generated_files.append(filename)
                    self.log(f"📋 Copied: {filename}")

            # Log summary
            self.log("\n" + "="*60)
            self.log("✅ Volume Calculation & Lattice Fitting Completed!")
            self.log("="*60)

            if 'transition_pressure' in results:
                self.log(f"📍 Phase transition pressure: {results['transition_pressure']:.2f} GPa")

            self.log(f"📁 Output location: {output_dir}")
            self.log(f"📊 Generated {len(generated_files)} result file(s)")

            for f in generated_files:
                self.log(f"   - {f}")

            self.log("="*60 + "\n")

            # Show success message
            success_msg = f"Volume calculation completed!\n\n"
            if 'transition_pressure' in results:
                success_msg += f"Transition at {results['transition_pressure']:.2f} GPa\n"
            success_msg += f"{len(generated_files)} file(s) saved to output directory"

            self.show_success(success_msg)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"❌ Error during analysis: {str(e)}")
            self.log(f"Details:\n{error_details}")
            messagebox.showerror("Error", f"Volume calculation failed:\n{str(e)}")
        finally:
            self.progress.stop()

    def run_birch_murnaghan(self):
        """Run Birch-Murnaghan equation of state fitting"""
        if not self.bm_input_file.get() or not self.bm_output_dir.get():
            messagebox.showerror("Error", "Please fill all required fields")
            return
        threading.Thread(target=self._run_birch_murnaghan_thread, daemon=True).start()

    def _run_birch_murnaghan_thread(self):
        """Background thread for Birch-Murnaghan fitting - Single Phase"""
        try:
            self.progress.start()
            order = int(self.bm_order.get())
            order_str = f"{order}rd order" if order == 3 else "2nd order"
            self.log(f"⚗️ Starting {order_str} Single-Phase BM Fitting")

            input_file_path = self.bm_input_file.get()
            output_directory = self.bm_output_dir.get()

            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # ==================== Step 1: Load P-V Data ====================
            self.log(f"📄 Reading data from: {os.path.basename(input_file_path)}")
            df = pd.read_csv(input_file_path)

            # Check for required columns
            if 'V_atomic' not in df.columns or 'Pressure (GPa)' not in df.columns:
                raise ValueError("CSV must contain 'V_atomic' and 'Pressure (GPa)' columns")

            V_data = df['V_atomic'].dropna().values
            P_data = df['Pressure (GPa)'].dropna().values

            # Ensure data pairing
            min_len = min(len(V_data), len(P_data))
            V_data = V_data[:min_len]
            P_data = P_data[:min_len]

            self.log(f"✓ Loaded {len(V_data)} data points")
            self.log(f"   Volume range: {V_data.min():.4f} - {V_data.max():.4f} Å³/atom")
            self.log(f"   Pressure range: {P_data.min():.2f} - {P_data.max():.2f} GPa")

            # ==================== Step 2: Initialize Fitter ====================
            # The BirchMurnaghanFitter only takes configuration parameters
            fitter = BirchMurnaghanFitter(
                V0_bounds=(0.8, 1.3),
                B0_bounds=(50, 500),
                B0_prime_bounds=(2.5, 6.5),
                max_iterations=10000
            )

            self.log(f"\n🔧 Fitting {order_str} Birch-Murnaghan equation...")

            # ==================== Step 3: Fit Single Phase ====================
            results = fitter.fit_single_phase(V_data, P_data, phase_name="Single Phase")

            # Extract results based on order
            if order == 2:
                if results['2nd_order'] is None:
                    raise ValueError("2nd order fitting failed")
                fit_results = results['2nd_order']
            else:  # order == 3
                if results['3rd_order'] is None:
                    raise ValueError("3rd order fitting failed")
                fit_results = results['3rd_order']

            # ==================== Step 4: Display Results ====================
            self.log(f"\n{'='*60}")
            self.log(f"✅ {order_str} BM Fitting Results:")
            self.log(f"{'='*60}")
            self.log(f"📊 V₀ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom")
            self.log(f"📊 B₀ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa")
            self.log(f"📊 B₀' = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}")
            self.log(f"📈 R² = {fit_results['R_squared']:.6f}")
            self.log(f"📉 RMSE = {fit_results['RMSE']:.4f} GPa")
            self.log(f"{'='*60}\n")

            # ==================== Step 5: Generate Plots ====================
            self.log("📈 Generating plots...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Left plot: P-V curve with fit
            ax1.scatter(V_data, P_data, s=80, c='blue', marker='o',
                       label='Experimental Data', alpha=0.7, edgecolors='black', linewidths=1.5)

            # Generate smooth fit curve
            V_fit = np.linspace(V_data.min()*0.95, V_data.max()*1.05, 200)
            if order == 2:
                P_fit = fitter.birch_murnaghan_2nd(V_fit, fit_results['V0'], fit_results['B0'])
                color = 'red'
            else:
                P_fit = fitter.birch_murnaghan_3rd(V_fit, fit_results['V0'],
                                                  fit_results['B0'], fit_results['B0_prime'])
                color = 'green'

            ax1.plot(V_fit, P_fit, color=color, linewidth=2.5,
                    label=f'{order_str} BM Fit', alpha=0.8)
            ax1.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Pressure P (GPa)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{order_str} Birch-Murnaghan Equation of State',
                         fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Add text box with fitting parameters
            textstr = f"$V_0$ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa\n"
            textstr += f"$B_0'$ = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}\n"
            textstr += f"$R^2$ = {fit_results['R_squared']:.6f}\n"
            textstr += f"RMSE = {fit_results['RMSE']:.4f} GPa"

            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.8, edgecolor='black'))

            # Right plot: Residuals
            residuals = P_data - fit_results['fitted_P']
            ax2.scatter(V_data, residuals, s=60, c='blue', marker='o',
                       alpha=0.7, edgecolors='black', linewidths=1.5)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
            ax2.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals (GPa)', fontsize=12, fontweight='bold')
            ax2.set_title('Fitting Residuals Analysis', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=10)

            # Add RMSE text box to residuals plot
            rmse_text = f"RMSE = {fit_results['RMSE']:.4f} GPa"
            ax2.text(0.05, 0.95, rmse_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightblue', alpha=0.8, edgecolor='black'))

            plt.tight_layout()

            # Save figure
            fig_path = os.path.join(output_directory, f'BM_{order}rd_order_single_phase_fit.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"💾 Plot saved: {os.path.basename(fig_path)}")

            # ==================== Step 7: Summary ====================
            self.log(f"\n{'='*60}")
            self.log("✨ All tasks completed successfully!")
            self.log(f"{'='*60}")
            self.log(f"📁 Output directory: {output_directory}")
            self.log(f"   - {os.path.basename(fig_path)} : P-V curve and residuals")
            self.log(f"{'='*60}\n")

            # Show success dialog
            success_msg = f"{order_str} BM fitting completed!\n\n"
            success_msg += f"V₀ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom\n"
            success_msg += f"B₀ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa\n"
            success_msg += f"B₀' = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}\n"
            success_msg += f"R² = {fit_results['R_squared']:.6f}\n\n"
            success_msg += "Results saved to output directory"

            self.show_success(success_msg)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"❌ Error during BM fitting: {str(e)}")
            self.log(f"\nDetails:\n{error_details}")
            messagebox.showerror("Error", f"BM fitting failed:\n\n{str(e)}")
        finally:
            self.progress.stop()

    # ==================== Radial XRD (Azimuthal Integration) Functions ====================

    def setup_radial_content(self):
        """Setup Radial XRD tab content with azimuthal integration"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title and description
        title_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(10, 20))

        title_card = self.create_card_frame(title_frame)
        title_card.pack(fill=tk.X)

        title_content = tk.Frame(title_card, bg=self.colors['card_bg'], padx=20, pady=15)
        title_content.pack(fill=tk.X)

        tk.Label(title_content, text="🎯", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 24)).pack(side=tk.LEFT, padx=(0, 10))

        title_text = tk.Frame(title_content, bg=self.colors['card_bg'])
        title_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(title_text, text="Azimuthal Integration",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 16, 'bold')).pack(anchor=tk.W)

        tk.Label(title_text, text="Integrate diffraction rings over selected azimuthal angle ranges",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Comic Sans MS', 10)).pack(anchor=tk.W)

        # Azimuthal angle reference diagram (text-based)
        ref_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        ref_frame.pack(fill=tk.X, pady=(0, 15))

        ref_card = self.create_card_frame(ref_frame)
        ref_card.pack(fill=tk.X)

        ref_content = tk.Frame(ref_card, bg=self.colors['card_bg'], padx=20, pady=12)
        ref_content.pack(fill=tk.X)

        tk.Label(ref_content, text="📐 Azimuthal Angle Reference:",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W)

        ref_text = "  0° = Right (→)  |  90° = Top (↑)  |  180° = Left (←)  |  270° = Bottom (↓)"
        tk.Label(ref_content, text=ref_text,
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Courier', 9)).pack(anchor=tk.W, pady=(5, 0))

        tk.Label(ref_content, text="  Counter-clockwise rotation from right horizontal",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Comic Sans MS', 8, 'italic')).pack(anchor=tk.W)

        # Integration Settings Card
        settings_card = self.create_card_frame(main_frame)
        settings_card.pack(fill=tk.X, pady=(0, 15))

        content1 = tk.Frame(settings_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content1.pack(fill=tk.BOTH, expand=True)

        header1 = tk.Frame(content1, bg=self.colors['card_bg'])
        header1.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header1, text="⚙️", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header1, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content1, "PONI File", self.radial_poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker(content1, "Mask File (Optional)", self.radial_mask_path,
                               [("EDF files", "*.edf"), ("NPY files", "*.npy"), ("All files", "*.*")])
        self.create_file_picker(content1, "Input Pattern", self.radial_input_pattern,
                               [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)
        self.create_folder_picker(content1, "Output Directory", self.radial_output_dir)
        self.create_entry(content1, "Dataset Path", self.radial_dataset_path)

        param_frame = tk.Frame(content1, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(5, 0))

        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(npt_cont, from_=500, to=10000, textvariable=self.radial_npt,
                   width=18, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(unit_cont, textvariable=self.radial_unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Azimuthal Angle Settings Card
        azimuth_card = self.create_card_frame(main_frame)
        azimuth_card.pack(fill=tk.X, pady=(0, 15))

        content2 = tk.Frame(azimuth_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content2.pack(fill=tk.BOTH, expand=True)

        header2 = tk.Frame(content2, bg=self.colors['card_bg'])
        header2.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header2, text="📊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header2, text="Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # Mode selection (Single or Multiple sectors)
        mode_frame = tk.Frame(content2, bg=self.colors['card_bg'])
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(mode_frame, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        mode_buttons = tk.Frame(mode_frame, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.radial_mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_radial_mode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors (Preset)", variable=self.radial_mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_radial_mode).pack(side=tk.LEFT)

        # Container for dynamic content (changes based on mode)
        self.radial_dynamic_frame = tk.Frame(content2, bg=self.colors['card_bg'])
        self.radial_dynamic_frame.pack(fill=tk.X)

        # Initialize with single sector mode
        self.update_radial_mode()

        # Action Buttons
        btn_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        btn_cont = tk.Frame(btn_frame, bg=self.colors['bg'])
        btn_cont.pack(expand=True)

        btns = tk.Frame(btn_cont, bg=self.colors['bg'])
        btns.pack()

        ModernButton(btns, "Run Azimuthal Integration", self.run_azimuthal_integration,
                    icon="🎯", bg_color=self.colors['accent'], hover_color=self.colors['primary_hover'],
                    width=300, height=45).pack(side=tk.LEFT, padx=8)

        # Progress bar section
        prog_cont = tk.Frame(main_frame, bg=self.colors['bg'])
        prog_cont.pack(fill=tk.X, pady=(15, 15))

        prog_inner = tk.Frame(prog_cont, bg=self.colors['bg'])
        prog_inner.pack(expand=True)

        self.radial_progress = CuteSheepProgressBar(prog_inner, width=780, height=80)
        self.radial_progress.pack()

        # Log area
        log_card = self.create_card_frame(main_frame)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_content = tk.Frame(log_card, bg=self.colors['card_bg'], padx=20, pady=12)
        log_content.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_content, bg=self.colors['card_bg'])
        log_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(log_header, text="📝", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(log_header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.radial_log_text = scrolledtext.ScrolledText(log_content, height=10, wrap=tk.WORD,
                                                         font=('Comic Sans MS', 10),
                                                         bg='#FAFAFA', fg='#B794F6',
                                                         relief='flat', borderwidth=0, padx=10, pady=10)
        self.radial_log_text.pack(fill=tk.BOTH, expand=True)

    def update_radial_mode(self):
        """Update the azimuthal settings UI based on selected mode"""
        # Clear existing content
        for widget in self.radial_dynamic_frame.winfo_children():
            widget.destroy()

        if self.radial_mode.get() == 'single':
            # Single sector mode - custom angle inputs
            angle_frame = tk.Frame(self.radial_dynamic_frame, bg=self.colors['card_bg'])
            angle_frame.pack(fill=tk.X, pady=(5, 0))

            # Start angle
            start_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(start_cont, text="Start Angle (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
            ttk.Spinbox(start_cont, from_=0, to=360, textvariable=self.radial_azimuth_start,
                       width=18, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

            # End angle
            end_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            tk.Label(end_cont, text="End Angle (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
            ttk.Spinbox(end_cont, from_=0, to=360, textvariable=self.radial_azimuth_end,
                       width=18, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

            # Sector label
            label_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            label_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(label_cont, text="Sector Label", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
            tk.Entry(label_cont, textvariable=self.radial_sector_label,
                    font=('Comic Sans MS', 9), width=20).pack(anchor=tk.W)

        else:
            # Multiple sectors mode - preset selection
            preset_frame = tk.Frame(self.radial_dynamic_frame, bg=self.colors['card_bg'])
            preset_frame.pack(fill=tk.X, pady=(5, 0))

            tk.Label(preset_frame, text="Sector Preset:", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

            ttk.Combobox(preset_frame, textvariable=self.radial_preset,
                        values=['quadrants', 'octants', 'hemispheres', 'horizontal_vertical'],
                        width=25, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

            # Show preset description
            preset_info = {
                'quadrants': "4 sectors: 0-90°, 90-180°, 180-270°, 270-360°",
                'octants': "8 sectors: Every 45° from 0° to 360°",
                'hemispheres': "2 sectors: 0-180° (Right), 180-360° (Left)",
                'horizontal_vertical': "4 sectors: Horizontal & Vertical directions"
            }

            info_text = preset_info.get(self.radial_preset.get(), "Select a preset")
            tk.Label(preset_frame, text=f"ℹ️ {info_text}",
                    bg=self.colors['card_bg'], fg=self.colors['text_light'],
                    font=('Comic Sans MS', 8, 'italic')).pack(anchor=tk.W, pady=(5, 0))

    def radial_log(self, message):
        """Log message to the radial log text widget"""
        if hasattr(self, 'radial_log_text'):
            self.radial_log_text.config(state='normal')
            self.radial_log_text.insert(tk.END, message + "\n")
            self.radial_log_text.see(tk.END)
            self.radial_log_text.config(state='disabled')
            self.root.update()

    def run_azimuthal_integration(self):
        """Run azimuthal integration based on selected mode"""
        # Validate inputs
        if not self.radial_poni_path.get():
            messagebox.showerror("Error", "Please select PONI file")
            return
        if not self.radial_input_pattern.get():
            messagebox.showerror("Error", "Please select input H5 files")
            return
        if not self.radial_output_dir.get():
            messagebox.showerror("Error", "Please select output directory")
            return

        # Run in background thread
        threading.Thread(target=self._run_azimuthal_integration_thread, daemon=True).start()

    def _run_azimuthal_integration_thread(self):
        """Background thread for azimuthal integration"""
        try:
            self.radial_progress.start()
            mode = self.radial_mode.get()

            if mode == 'single':
                self.radial_log("🎯 Starting Single Sector Azimuthal Integration")
                self._run_single_sector()
            else:
                self.radial_log("🎯 Starting Multiple Sectors Azimuthal Integration")
                self._run_multiple_sectors()

            self.radial_log("✅ Azimuthal integration completed!")
            self.show_success("Azimuthal integration completed successfully!")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.radial_log(f"❌ Error: {str(e)}")
            self.radial_log(f"\nDetails:\n{error_details}")
            messagebox.showerror("Error", f"Azimuthal integration failed:\n{str(e)}")
        finally:
            self.radial_progress.stop()

    def _run_single_sector(self):
        """Run single sector azimuthal integration"""
        self.radial_log(f"📁 PONI file: {os.path.basename(self.radial_poni_path.get())}")
        if self.radial_mask_path.get():
            self.radial_log(f"🎭 Mask file: {os.path.basename(self.radial_mask_path.get())}")

        # Initialize integrator
        integrator = AzimuthalIntegrator(
            self.radial_poni_path.get(),
            self.radial_mask_path.get() if self.radial_mask_path.get() else None
        )

        azim_start = self.radial_azimuth_start.get()
        azim_end = self.radial_azimuth_end.get()
        sector_label = self.radial_sector_label.get()

        self.radial_log(f"📐 Azimuthal range: {azim_start}° to {azim_end}°")
        self.radial_log(f"🏷️  Sector label: {sector_label}")

        # Run batch integration
        output_files = integrator.batch_integrate_h5(
            input_pattern=self.radial_input_pattern.get(),
            output_dir=self.radial_output_dir.get(),
            azimuth_start=azim_start,
            azimuth_end=azim_end,
            npt=self.radial_npt.get(),
            unit=self.radial_unit.get(),
            dataset_path=self.radial_dataset_path.get(),
            sector_label=sector_label
        )

        self.radial_log(f"\n{'='*60}")
        self.radial_log(f"✨ Integration complete!")
        self.radial_log(f"📊 Generated {len(output_files)} files")
        self.radial_log(f"📁 Output directory: {self.radial_output_dir.get()}")
        self.radial_log(f"{'='*60}\n")

    def _run_multiple_sectors(self):
        """Run multiple sectors azimuthal integration"""
        self.radial_log(f"📁 PONI file: {os.path.basename(self.radial_poni_path.get())}")
        if self.radial_mask_path.get():
            self.radial_log(f"🎭 Mask file: {os.path.basename(self.radial_mask_path.get())}")

        # Initialize integrator
        integrator = AzimuthalIntegrator(
            self.radial_poni_path.get(),
            self.radial_mask_path.get() if self.radial_mask_path.get() else None
        )

        preset_name = self.radial_preset.get()
        sector_list = get_preset_sectors(preset_name)

        if not sector_list:
            raise ValueError(f"Invalid preset: {preset_name}")

        self.radial_log(f"📐 Preset: {preset_name}")
        self.radial_log(f"📊 Number of sectors: {len(sector_list)}")

        for start, end, label in sector_list:
            self.radial_log(f"   - {label}: {start}° to {end}°")

        # Run batch integration
        output_files = integrator.batch_integrate_multiple_sectors(
            input_pattern=self.radial_input_pattern.get(),
            output_dir=self.radial_output_dir.get(),
            sector_list=sector_list,
            npt=self.radial_npt.get(),
            unit=self.radial_unit.get(),
            dataset_path=self.radial_dataset_path.get()
        )

        self.radial_log(f"\n{'='*60}")
        self.radial_log(f"✨ Integration complete!")
        self.radial_log(f"📊 Generated {len(output_files)} files")
        self.radial_log(f"📁 Output directory: {self.radial_output_dir.get()}")
        self.radial_log(f"{'='*60}\n")


def launch_main_app():
    # Set AppUserModelID (important for taskbar icon)
    app_id = u"mycompany.myapp.xrdpostprocessor"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    # Create main window
    root = tk.Tk()

    # ✅ Set window icon (taskbar + title bar)
    icon_path = r"D:\HEPS\ID31\dioptas_data\github_felicity\batch\ChatGPT Image.ico"
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Failed to set icon: {e}")
    else:
        print("Icon file not found!")

    # ✅ Set window title
    root.title("XRD Data Post-Processing")

    # ✅ Set window size and allow resizing
    root.geometry("700x400")
    root.resizable(True, True)  # width & height resizable

    # ✅ Set background color to purple-pink
    root.configure(bg="#EDE9F3")  # light purple-pink (Thistle)

    # ✅ Define cute font style
    cute_font = tkFont.Font(family="Comic Sans MS", size=14, weight="bold")

    # ✅ Create adorable welcome label
    welcome_text = ("💜 Hey there, crystal cutie! Ready to sparkle your XRD data? 🌈\n"
                    "\n"
                    #"💜 Beam me up, XRD Commander – it's fitting time! ~ 💖✨\n"
                    "\n"
                    "📧 Contact: candicewang928@egmail.com")
    label = tk.Label(
        root,
        text=welcome_text,
        font=cute_font,
        bg="#F9EBF2",
        fg="#8E24AA",
        pady=90
    )
    label.pack(pady=40)

    root.mainloop()

def show_startup_window():
    splash = tk.Tk()
    splash.title("Loading...")

    # Set startup window size and position
    splash.geometry("300x100")
    tk.Label(splash, text="Starting up, please wait...").pack(pady=20)

    # Automatically close startup window and open the main app
    splash.after(100, lambda: [splash.destroy(), launch_main_app()])
    splash.mainloop()

if __name__ == "__main__":
    show_startup_window()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)  # Construct your main GUI here
    root.mainloop()

if __name__ == "__main__":
    main()
