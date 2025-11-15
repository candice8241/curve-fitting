# -*- coding: utf-8 -*-
"""
Powder XRD Module
Contains integration, peak fitting, phase analysis, and Birch-Murnaghan fitting
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from gui_base import GUIBase
from batch_appearance import ModernButton, CuteSheepProgressBar
from batch_integration import BatchIntegrator
from peak_fitting import BatchFitter
from batch_cal_volume import XRayDiffractionAnalyzer as XRDAnalyzer
from birch_murnaghan_batch import BirchMurnaghanFitter


class PowderXRDModule(GUIBase):
    """Powder XRD processing module with integration and analysis capabilities"""

    def __init__(self, parent, root):
        """
        Initialize Powder XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root Tk window for dialogs
        """
        super().__init__()
        self.parent = parent
        self.root = root
        self.current_module = "integration"

        # Initialize variables
        self._init_variables()

    def _init_variables(self):
        """Initialize all Tkinter variables"""
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

    def setup_ui(self):
        """Setup the complete powder XRD UI"""
        main_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Module selector buttons
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
        """Switch between integration and analysis modules"""
        self.current_module = module_type

        # Update button colors
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

    # ==================== Processing Functions ====================

    def log(self, message):
        """Log message to the log text widget"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

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

            self.show_success(self.root, f"Peak separation completed!\n\n"
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
            self.show_success(self.root, "Integration completed!")
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
            self.show_success(self.root, "Fitting completed!")
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
            self.show_success(self.root, "Full pipeline completed!")
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

            self.show_success(self.root, success_msg)

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
        """Background thread for Birch-Murnaghan fitting"""
        try:
            self.progress.start()
            order = int(self.bm_order.get())
            order_str = f"{order}rd order" if order == 3 else "2nd order"
            self.log(f"⚗️ Starting {order_str} Single-Phase BM Fitting")

            input_file_path = self.bm_input_file.get()
            output_directory = self.bm_output_dir.get()

            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Load P-V Data
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

            # Initialize Fitter
            fitter = BirchMurnaghanFitter(
                V0_bounds=(0.8, 1.3),
                B0_bounds=(50, 500),
                B0_prime_bounds=(2.5, 6.5),
                max_iterations=10000
            )

            self.log(f"\n🔧 Fitting {order_str} Birch-Murnaghan equation...")

            # Fit Single Phase
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

            # Display Results
            self.log(f"\n{'='*60}")
            self.log(f"✅ {order_str} BM Fitting Results:")
            self.log(f"{'='*60}")
            self.log(f"📊 V₀ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom")
            self.log(f"📊 B₀ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa")
            self.log(f"📊 B₀' = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}")
            self.log(f"📈 R² = {fit_results['R_squared']:.6f}")
            self.log(f"📉 RMSE = {fit_results['RMSE']:.4f} GPa")
            self.log(f"{'='*60}\n")

            # Generate Plots
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

            # Summary
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

            self.show_success(self.root, success_msg)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"❌ Error during BM fitting: {str(e)}")
            self.log(f"\nDetails:\n{error_details}")
            messagebox.showerror("Error", f"BM fitting failed:\n\n{str(e)}")
        finally:
            self.progress.stop()
