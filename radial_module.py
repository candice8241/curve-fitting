#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XRD Azimuthal Integration Module with GUI
==========================================
Author: candicewang928@gmail.com
Created: Nov 15, 2025

This module performs azimuthal integration on XRD diffraction ring data stored in HDF5 format.
Designed to be embedded in the main XRD processing application.
"""

import os
import glob
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import hdf5plugin
import h5py
import numpy as np
import pandas as pd
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

from prefernce import GUIBase, CuteSheepProgressBar


# ============================================================================
# Core Integration Class
# ============================================================================

class XRDAzimuthalIntegrator:
    """Class to handle azimuthal integration of XRD diffraction data."""

    def __init__(self, poni_file: str, mask_file: Optional[str] = None):
        """
        Initialize the azimuthal integrator.

        Parameters:
        -----------
        poni_file : str
            Path to the PONI calibration file
        mask_file : str, optional
            Path to the mask file (can be .npy, .edf, or .tif)
        """
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.ai = None
        self.mask = None

        self._load_calibration()
        if mask_file:
            self._load_mask()

    def _load_calibration(self):
        """Load the PONI calibration file."""
        if not os.path.exists(self.poni_file):
            raise FileNotFoundError(f"PONI file not found: {self.poni_file}")

        print(f"Loading calibration from: {self.poni_file}")
        self.ai = pyFAI.load(self.poni_file)
        print(f"  Detector: {self.ai.detector.name}")
        print(f"  Distance: {self.ai.dist * 1000:.2f} mm")
        print(f"  Wavelength: {self.ai.wavelength * 1e10:.4f} Å")

    def _load_mask(self):
        """Load the mask file."""
        if not os.path.exists(self.mask_file):
            print(f"Warning: Mask file not found: {self.mask_file}")
            return

        print(f"Loading mask from: {self.mask_file}")

        ext = os.path.splitext(self.mask_file)[1].lower()

        if ext == '.npy':
            self.mask = np.load(self.mask_file)
        elif ext in ['.edf', '.tif', '.tiff']:
            try:
                import fabio
                img = fabio.open(self.mask_file)
                self.mask = img.data
            except ImportError:
                print("Warning: fabio not installed. Cannot read mask file.")
                return
        elif ext in ['.h5', '.hdf5']:
            with h5py.File(self.mask_file, 'r') as f:
                for key in ['mask', 'data', 'entry/data/data']:
                    if key in f:
                        self.mask = f[key][:]
                        break
                else:
                    keys = list(f.keys())
                    if keys:
                        self.mask = f[keys[0]][:]
        else:
            print(f"Warning: Unsupported mask file format: {ext}")
            return

        print(f"  Mask shape: {self.mask.shape}")
        print(f"  Masked pixels: {np.sum(self.mask)}")

    def integrate_file(self, h5_file: str, output_dir: str,
                      npt: int = 2048,
                      unit: str = "q_A^-1",
                      output_format: str = "xy",
                      azimuth_range: Optional[tuple] = None,
                      sector_label: str = "") -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Integrate a single HDF5 file with optional azimuthal range.

        Parameters:
        -----------
        h5_file : str
            Path to the HDF5 file containing 2D diffraction data
        output_dir : str
            Directory to save the integrated data
        npt : int
            Number of points in the integrated pattern
        unit : str
            Unit for the radial axis
        output_format : str
            Output file format
        azimuth_range : tuple, optional
            Tuple of (start_angle, end_angle) in degrees
        sector_label : str
            Label for the sector in output filename

        Returns:
        --------
        tuple : (output_file_path, x_data, y_data)
        """
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

        print(f"\nProcessing: {os.path.basename(h5_file)}")

        data = self._read_h5_data(h5_file)

        print(f"  Integrating with {npt} points, unit={unit}")
        if azimuth_range:
            print(f"  Azimuthal range: {azimuth_range[0]}° to {azimuth_range[1]}°")

        result = self.ai.integrate1d(
            data,
            npt=npt,
            mask=self.mask,
            unit=unit,
            method="splitpixel",
            error_model="poisson",
            azimuth_range=azimuth_range
        )

        if isinstance(result, tuple):
            if len(result) == 2:
                q, intensity = result
            elif len(result) == 3:
                q, intensity, sigma = result
            else:
                q = result[0]
                intensity = result[1]
        else:
            q = result.radial
            intensity = result.intensity

        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        if sector_label:
            output_file = os.path.join(output_dir, f"{base_name}_{sector_label}.{output_format}")
        else:
            output_file = os.path.join(output_dir, f"{base_name}_integrated.{output_format}")

        self._save_data(q, intensity, output_file, unit, output_format)

        print(f"  Saved: {output_file}")
        return output_file, q, intensity

    def _read_h5_data(self, h5_file: str, dataset_path: str = None) -> np.ndarray:
        """Read 2D diffraction data from HDF5 file."""
        with h5py.File(h5_file, 'r') as f:
            if dataset_path and dataset_path in f:
                data = f[dataset_path][...]
                if data.ndim == 3:
                    data = data[0]
                return data

            common_paths = [
                'entry/data/data',
                'entry/instrument/detector/data',
                'data',
                'image',
                'diffraction'
            ]

            data = None
            for path in common_paths:
                if path in f:
                    data = f[path][...]
                    if data.ndim == 3:
                        data = data[0]
                    break

            if data is None:
                def find_2d_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if obj.ndim == 2 or (obj.ndim == 3 and obj.shape[0] == 1):
                            return name
                    return None

                for key in f.keys():
                    result = f[key].visititems(find_2d_dataset)
                    if result:
                        data = f[result][...]
                        if data.ndim == 3:
                            data = data[0]
                        break

                if data is None:
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][...]
                        if data.ndim == 3:
                            data = data[0]

            if data is None:
                raise ValueError(f"No suitable dataset found in {h5_file}")

            print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Intensity range: [{np.min(data):.1f}, {np.max(data):.1f}]")

            return data

    def _save_data(self, q: np.ndarray, intensity: np.ndarray,
                   output_file: str, unit: str, output_format: str):
        """Save the integrated data to file."""
        if output_format == "xy":
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity"
            np.savetxt(output_file, np.column_stack([q, intensity]),
                      header=header, fmt='%.6e')

        elif output_format == "chi":
            with open(output_file, 'w') as f:
                f.write(f"2-Theta Angle (Degrees)\n")
                f.write(f"Intensity\n")
                for q_val, int_val in zip(q, intensity):
                    f.write(f"{q_val:.6f} {int_val:.6f}\n")

        elif output_format == "dat":
            errors = np.sqrt(np.maximum(intensity, 1))
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity\n# Column 3: Error"
            np.savetxt(output_file, np.column_stack([q, intensity, errors]),
                      header=header, fmt='%.6e')

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def batch_process(self, h5_files: List[str], output_dir: str, **kwargs) -> List[str]:
        """Process multiple HDF5 files."""
        output_files = []
        total = len(h5_files)

        print(f"\n{'='*60}")
        print(f"Batch processing {total} file(s)")
        print(f"{'='*60}")

        for i, h5_file in enumerate(h5_files, 1):
            print(f"\n[{i}/{total}]", end=" ")
            try:
                output_file, _, _ = self.integrate_file(h5_file, output_dir, **kwargs)
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing {h5_file}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Completed: {len(output_files)}/{total} files processed successfully")
        print(f"{'='*60}\n")

        return output_files


# ============================================================================
# GUI Module (Refactored for embedding)
# ============================================================================

class AzimuthalIntegrationModule(GUIBase):
    """Azimuthal Integration module for embedding in main application"""

    def __init__(self, parent, root):
        """
        Initialize the module

        Args:
            parent: Parent frame to embed this module in
            root: Root Tk window (for dialogs and threading)
        """
        super().__init__()
        self.parent = parent
        self.root = root

        # Variables
        self._init_variables()

        # Processing state
        self.processing = False
        self.stop_processing = False

        # Custom sectors
        self.custom_sectors = []

    def _init_variables(self):
        """Initialize all Tkinter variables"""
        self.poni_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.input_pattern = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.dataset_path = tk.StringVar(value="entry/data/data")
        self.npt = tk.IntVar(value=4000)
        self.unit = tk.StringVar(value='2th_deg')
        self.azimuth_start = tk.DoubleVar(value=0.0)
        self.azimuth_end = tk.DoubleVar(value=90.0)
        self.sector_label = tk.StringVar(value="Sector_1")
        self.preset = tk.StringVar(value='quadrants')
        self.mode = tk.StringVar(value='single')
        self.multiple_mode = tk.StringVar(value='custom')
        self.output_csv = tk.BooleanVar(value=True)

    def setup_ui(self):
        """Setup the complete UI (called when tab is activated)"""
        # Clear any existing content
        for widget in self.parent.winfo_children():
            widget.destroy()

        # Title section
        self._create_title_section()

        # Reference section
        self._create_reference_section()

        # Settings section
        self._create_settings_section()

        # Azimuthal settings section
        self._create_azimuthal_section()

        # Action buttons
        self._create_action_buttons()

        # Progress bar
        self._create_progress_section()

        # Log section
        self._create_log_section()

    def _create_title_section(self):
        """Create title section"""
        title_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        title_card = self.create_card_frame(title_frame)
        title_card.pack(fill=tk.X)

        content = tk.Frame(title_card, bg=self.colors['card_bg'], padx=20, pady=15)
        content.pack(fill=tk.X)

        tk.Label(content, text="🎯", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 24)).pack(side=tk.LEFT, padx=(0, 10))

        text_frame = tk.Frame(content, bg=self.colors['card_bg'])
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(text_frame, text="Azimuthal Integration",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Arial', 16, 'bold')).pack(anchor=tk.W)

        tk.Label(text_frame, text="Integrate diffraction rings over selected azimuthal angle ranges",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Arial', 10)).pack(anchor=tk.W)

    def _create_reference_section(self):
        """Create azimuthal angle reference diagram"""
        ref_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        ref_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        ref_card = self.create_card_frame(ref_frame)
        ref_card.pack(fill=tk.X)

        content = tk.Frame(ref_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.X)

        tk.Label(content, text="📐 Azimuthal Angle Reference:",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        ref_text = "  0° = Right (→)  |  90° = Top (↑)  |  180° = Left (←)  |  270° = Bottom (↓)"
        tk.Label(content, text=ref_text,
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Courier', 9)).pack(anchor=tk.W, pady=(5, 0))

        tk.Label(content, text="  Counter-clockwise rotation from right horizontal",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Arial', 8, 'italic')).pack(anchor=tk.W)

    def _create_settings_section(self):
        """Create integration settings section"""
        settings_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        settings_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        card = self.create_card_frame(settings_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="⚙️", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # File pickers
        self._create_file_picker(content, "PONI File", self.poni_path,
                                [("PONI files", "*.poni"), ("All files", "*.*")])
        self._create_file_picker(content, "Mask File (Optional)", self.mask_path,
                                [("EDF files", "*.edf"), ("NPY files", "*.npy"), ("All files", "*.*")])
        self._create_file_picker(content, "Input Pattern", self.input_pattern,
                                [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)
        self._create_folder_picker(content, "Output Directory", self.output_dir)
        self._create_entry(content, "Dataset Path", self.dataset_path)

        # Parameters
        param_frame = tk.Frame(content, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(5, 0))

        # NPT
        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                   width=18, font=('Arial', 9)).pack(anchor=tk.W)

        # Unit
        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(unit_cont, textvariable=self.unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Arial', 9)).pack(anchor=tk.W)

        # CSV option
        csv_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        csv_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Checkbutton(csv_cont, text="Output CSV", variable=self.output_csv,
                      bg=self.colors['card_bg'], font=('Arial', 9, 'bold'),
                      activebackground=self.colors['card_bg']).pack(anchor=tk.W, pady=(0, 2))
        tk.Label(csv_cont, text="(Save merged results)", bg=self.colors['card_bg'],
                fg=self.colors['text_light'], font=('Arial', 7, 'italic')).pack(anchor=tk.W)

    def _create_azimuthal_section(self):
        """Create azimuthal angle settings section"""
        azimuth_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        azimuth_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        card = self.create_card_frame(azimuth_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="📊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Mode selection
        mode_frame = tk.Frame(content, bg=self.colors['card_bg'])
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(mode_frame, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        mode_buttons = tk.Frame(mode_frame, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Arial', 9),
                      command=self.update_mode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Arial', 9),
                      command=self.update_mode).pack(side=tk.LEFT)

        # Dynamic content container
        self.dynamic_frame = tk.Frame(content, bg=self.colors['card_bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize with single sector mode
        self.update_mode()

    def _create_action_buttons(self):
        """Create action buttons"""
        btn_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        btn_cont = tk.Frame(btn_frame, bg=self.colors['bg'])
        btn_cont.pack(expand=True)

        run_btn = tk.Button(btn_cont, text="🎯 Run Azimuthal Integration",
                           command=self.run_integration,
                           bg=self.colors['accent'], fg='white',
                           font=('Arial', 11, 'bold'), relief='flat',
                           padx=20, pady=12, cursor='hand2')
        run_btn.pack()

    def _create_progress_section(self):
        """Create progress bar section"""
        prog_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        prog_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        self.progress_bar = CuteSheepProgressBar(prog_frame, width=780, height=80)
        self.progress_bar.pack()

    def _create_log_section(self):
        """Create log section"""
        log_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 20))

        card = self.create_card_frame(log_frame)
        card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="📝", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(content, height=12, wrap=tk.WORD,
                                                  font=('Courier', 9),
                                                  bg='#FAFAFA', fg=self.colors['primary'],
                                                  relief='flat', borderwidth=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _create_file_picker(self, parent, label, variable, filetypes, pattern=False):
        """Create a file picker row"""
        row = tk.Frame(parent, bg=self.colors['card_bg'])
        row.pack(fill=tk.X, pady=5)

        tk.Label(row, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold'),
                width=20, anchor='w').pack(side=tk.LEFT)

        entry = tk.Entry(row, textvariable=variable, font=('Arial', 9), width=50)
        entry.pack(side=tk.LEFT, padx=5)

        def browse():
            if pattern:
                path = filedialog.askopenfilename(filetypes=filetypes)
                if path:
                    # Convert to pattern
                    dir_path = os.path.dirname(path)
                    ext = os.path.splitext(path)[1]
                    pattern_path = os.path.join(dir_path, f"*{ext}")
                    variable.set(pattern_path)
            else:
                path = filedialog.askopenfilename(filetypes=filetypes)
                if path:
                    variable.set(path)

        tk.Button(row, text="Browse", command=browse,
                 bg=self.colors['primary'], fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=10, pady=2).pack(side=tk.LEFT)

    def _create_folder_picker(self, parent, label, variable):
        """Create a folder picker row"""
        row = tk.Frame(parent, bg=self.colors['card_bg'])
        row.pack(fill=tk.X, pady=5)

        tk.Label(row, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold'),
                width=20, anchor='w').pack(side=tk.LEFT)

        entry = tk.Entry(row, textvariable=variable, font=('Arial', 9), width=50)
        entry.pack(side=tk.LEFT, padx=5)

        def browse():
            path = filedialog.askdirectory()
            if path:
                variable.set(path)

        tk.Button(row, text="Browse", command=browse,
                 bg=self.colors['primary'], fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=10, pady=2).pack(side=tk.LEFT)

    def _create_entry(self, parent, label, variable):
        """Create an entry row"""
        row = tk.Frame(parent, bg=self.colors['card_bg'])
        row.pack(fill=tk.X, pady=5)

        tk.Label(row, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold'),
                width=20, anchor='w').pack(side=tk.LEFT)

        entry = tk.Entry(row, textvariable=variable, font=('Arial', 9), width=50)
        entry.pack(side=tk.LEFT, padx=5)

    def update_mode(self):
        """Update UI based on selected mode"""
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        if self.mode.get() == 'single':
            self._setup_single_sector_ui()
        else:
            self._setup_multiple_sectors_ui()

    def _setup_single_sector_ui(self):
        """Setup UI for single sector mode"""
        angle_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        angle_frame.pack(fill=tk.X, pady=(5, 0))

        start_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
        start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(start_cont, text="Start Angle (°)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(start_cont, from_=0, to=360, textvariable=self.azimuth_start,
                   width=18, font=('Arial', 9)).pack(anchor=tk.W)

        end_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
        end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(end_cont, text="End Angle (°)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(end_cont, from_=0, to=360, textvariable=self.azimuth_end,
                   width=18, font=('Arial', 9)).pack(anchor=tk.W)

        label_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
        label_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(label_cont, text="Sector Label", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        tk.Entry(label_cont, textvariable=self.sector_label,
                font=('Arial', 9), width=20).pack(anchor=tk.W)

    def _setup_multiple_sectors_ui(self):
        """Setup UI for multiple sectors mode"""
        submode_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        submode_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Label(submode_frame, text="Multiple Sectors Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        submode_buttons = tk.Frame(submode_frame, bg=self.colors['card_bg'])
        submode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(submode_buttons, text="Preset Templates", variable=self.multiple_mode,
                      value='preset', bg=self.colors['card_bg'],
                      font=('Arial', 9),
                      command=self.update_multiple_submode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(submode_buttons, text="Custom Sectors", variable=self.multiple_mode,
                      value='custom', bg=self.colors['card_bg'],
                      font=('Arial', 9),
                      command=self.update_multiple_submode).pack(side=tk.LEFT)

        self.submode_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        self.submode_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.update_multiple_submode()

    def update_multiple_submode(self):
        """Update UI based on multiple sectors sub-mode"""
        for widget in self.submode_frame.winfo_children():
            widget.destroy()

        if self.multiple_mode.get() == 'preset':
            self._setup_preset_mode()
        else:
            self._setup_custom_sectors_mode()

    def _setup_preset_mode(self):
        """Setup preset templates mode"""
        preset_frame = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        preset_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(preset_frame, text="Select Preset:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ttk.Combobox(preset_frame, textvariable=self.preset,
                    values=['quadrants', 'octants', 'hemispheres'],
                    width=25, state='readonly', font=('Arial', 9)).pack(anchor=tk.W)

        preset_info = {
            'quadrants': "4 sectors: 0-90°, 90-180°, 180-270°, 270-360°",
            'octants': "8 sectors: Every 45° from 0° to 360°",
            'hemispheres': "2 sectors: 0-180° (Right), 180-360° (Left)"
        }

        info_text = preset_info.get(self.preset.get(), "Select a preset")
        tk.Label(preset_frame, text=f"ℹ️ {info_text}",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Arial', 8, 'italic')).pack(anchor=tk.W, pady=(5, 0))

    def _setup_custom_sectors_mode(self):
        """Setup custom sectors mode"""
        instruction_frame = tk.Frame(self.submode_frame, bg='#FFF8DC',
                                     relief='solid', borderwidth=1, padx=10, pady=8)
        instruction_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(instruction_frame, text="💡 Define custom azimuthal sectors. Add multiple rows for different angular ranges.",
                bg='#FFF8DC', fg=self.colors['text_dark'],
                font=('Arial', 8)).pack(anchor=tk.W)

        # Initialize default sectors if empty
        if not self.custom_sectors:
            self.custom_sectors = [
                [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1")],
                [tk.DoubleVar(value=90.0), tk.DoubleVar(value=180.0), tk.StringVar(value="Sector_2")],
                [tk.DoubleVar(value=180.0), tk.DoubleVar(value=270.0), tk.StringVar(value="Sector_3")],
                [tk.DoubleVar(value=270.0), tk.DoubleVar(value=360.0), tk.StringVar(value="Sector_4")]
            ]

        # Sectors display
        for idx, sector in enumerate(self.custom_sectors):
            row_frame = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
            row_frame.pack(fill=tk.X, pady=2)

            tk.Label(row_frame, text=f"#{idx+1}", bg=self.colors['card_bg'],
                    font=('Arial', 8), width=3).pack(side=tk.LEFT, padx=2)

            tk.Label(row_frame, text="Start:", bg=self.colors['card_bg'],
                    font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
            ttk.Spinbox(row_frame, from_=0, to=360, textvariable=sector[0],
                       width=8, font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

            tk.Label(row_frame, text="End:", bg=self.colors['card_bg'],
                    font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
            ttk.Spinbox(row_frame, from_=0, to=360, textvariable=sector[1],
                       width=8, font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

            tk.Label(row_frame, text="Label:", bg=self.colors['card_bg'],
                    font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
            tk.Entry(row_frame, textvariable=sector[2],
                    font=('Arial', 8), width=15).pack(side=tk.LEFT, padx=2)

            tk.Button(row_frame, text="✖", command=lambda i=idx: self._delete_sector(i),
                     bg='#FF6B6B', fg='white', font=('Arial', 8, 'bold'),
                     relief='flat', width=3).pack(side=tk.LEFT, padx=2)

        # Add sector button
        btn_frame = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(btn_frame, text="➕ Add Sector", command=self._add_sector,
                 bg=self.colors['success'], fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(btn_frame, text="🗑️ Clear All", command=self._clear_all_sectors,
                 bg=self.colors['error'], fg='white',
                 font=('Arial', 8, 'bold'), relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT)

    def _add_sector(self):
        """Add a new sector row"""
        new_sector = [
            tk.DoubleVar(value=0.0),
            tk.DoubleVar(value=90.0),
            tk.StringVar(value=f"Sector_{len(self.custom_sectors) + 1}")
        ]
        self.custom_sectors.append(new_sector)
        self.update_multiple_submode()

    def _delete_sector(self, index):
        """Delete a sector by index"""
        if len(self.custom_sectors) > 1:
            del self.custom_sectors[index]
            self.update_multiple_submode()
        else:
            messagebox.showwarning("Warning", "At least one sector must be defined!")

    def _clear_all_sectors(self):
        """Clear all sectors"""
        result = messagebox.askyesno("Confirm", "Clear all sectors and reset to default?")
        if result:
            self.custom_sectors = [
                [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1")]
            ]
            self.update_multiple_submode()

    def log(self, message):
        """Thread-safe log message"""
        def _log():
            try:
                if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
            except tk.TclError:
                pass

        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            try:
                self.root.after(0, _log)
            except tk.TclError:
                pass

    def run_integration(self):
        """Main entry point for running integration"""
        # Prevent duplicate runs
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already running!")
            return

        # Validate inputs
        if not self.poni_path.get():
            messagebox.showerror("Error", "Please select PONI file")
            return
        if not self.input_pattern.get():
            messagebox.showerror("Error", "Please select input H5 files")
            return
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select output directory")
            return

        dataset_path = self.dataset_path.get().strip()
        if not dataset_path:
            messagebox.showerror("Error", "Dataset path cannot be empty!")
            return

        # Gather parameters
        params = {
            'mode': self.mode.get(),
            'poni_path': self.poni_path.get(),
            'mask_path': self.mask_path.get() if self.mask_path.get() else None,
            'input_pattern': self.input_pattern.get(),
            'output_dir': self.output_dir.get(),
            'dataset_path': dataset_path,
            'npt': self.npt.get(),
            'unit': self.unit.get(),
            'output_csv': self.output_csv.get()
        }

        # Get sectors
        if params['mode'] == 'single':
            params['sectors'] = [(
                float(self.azimuth_start.get()),
                float(self.azimuth_end.get()),
                str(self.sector_label.get())
            )]
        else:
            if self.multiple_mode.get() == 'preset':
                preset_name = self.preset.get()
                params['sectors'] = self._get_preset_sectors(preset_name)
                params['preset_name'] = preset_name
            else:
                sectors = []
                for sector_data in self.custom_sectors:
                    start = sector_data[0].get()
                    end = sector_data[1].get()
                    label = sector_data[2].get()
                    sectors.append((float(start), float(end), str(label)))
                params['sectors'] = sectors

        # Set processing flag
        self.processing = True
        self.stop_processing = False

        # Run in background thread
        threading.Thread(target=self._run_integration_thread,
                        args=(params,), daemon=True).start()

    def _run_integration_thread(self, params):
        """Background thread for integration"""
        try:
            # Start progress bar
            def safe_progress_start():
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.start()
                except tk.TclError:
                    pass

            self.root.after(0, safe_progress_start)

            # Run integration
            if params['mode'] == 'single':
                self.log("🎯 Starting Single Sector Azimuthal Integration")
                self._run_single_sector(params)
            else:
                self.log("🎯 Starting Multiple Sectors Azimuthal Integration")
                self._run_multiple_sectors(params)

            if not self.stop_processing:
                self.log("✅ Azimuthal integration completed!")
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    "Azimuthal integration completed successfully!"))

        except Exception as e:
            if not self.stop_processing:
                import traceback
                error_details = traceback.format_exc()
                self.log(f"❌ Error: {str(e)}")
                self.log(f"\nDetails:\n{error_details}")
                self.root.after(0, lambda err=str(e): messagebox.showerror("Error",
                    f"Azimuthal integration failed:\n{err}"))

        finally:
            # Stop progress bar
            def safe_progress_stop():
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.stop()
                except tk.TclError:
                    pass

            self.root.after(0, safe_progress_stop)
            self.processing = False

    def _run_single_sector(self, params):
        """Run single sector integration"""
        self.log(f"📁 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🎭 Mask file: {os.path.basename(params['mask_path'])}")

        azim_start, azim_end, sector_label = params['sectors'][0]

        self.log(f"📐 Azimuthal range: {azim_start}° to {azim_end}°")
        self.log(f"🏷️  Sector label: {sector_label}")

        output_files = self._integrate_sector(params, azim_start, azim_end, sector_label)

        self.log(f"\n{'='*60}")
        self.log(f"✨ Integration complete!")
        self.log(f"📊 Generated {len(output_files)} files")
        self.log(f"📁 Output directory: {params['output_dir']}")
        self.log(f"{'='*60}\n")

    def _run_multiple_sectors(self, params):
        """Run multiple sectors integration"""
        self.log(f"📁 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🎭 Mask file: {os.path.basename(params['mask_path'])}")

        sector_list = params['sectors']

        if 'preset_name' in params:
            self.log(f"📐 Using preset: {params['preset_name']}")
        else:
            self.log(f"📐 Using custom sectors")

        self.log(f"📊 Number of sectors: {len(sector_list)}")

        for start, end, label in sector_list:
            self.log(f"   - {label}: {start}° to {end}°")

        all_output_files = []
        for start, end, label in sector_list:
            if self.stop_processing:
                self.log("⚠️ Processing stopped by user")
                break
            self.log(f"\n🔄 Processing {label}...")
            output_files = self._integrate_sector(params, start, end, label)
            all_output_files.extend(output_files)

        if not self.stop_processing:
            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"📊 Generated {len(all_output_files)} files total")
            self.log(f"📁 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _integrate_sector(self, params, azim_start, azim_end, sector_label):
        """Perform integration for a single sector"""
        # Initialize integrator
        integrator = XRDAzimuthalIntegrator(
            params['poni_path'],
            params['mask_path']
        )

        # Get input files
        input_files = sorted(glob.glob(params['input_pattern']))
        if not input_files:
            raise ValueError(f"No files found matching pattern: {params['input_pattern']}")

        self.log(f"   Found {len(input_files)} input files")

        output_dir = params['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        csv_data = {}
        output_files = []

        for idx, h5_file in enumerate(input_files):
            if self.stop_processing:
                break

            filename = os.path.basename(h5_file)
            self.log(f"   [{idx+1}/{len(input_files)}] {filename}")

            try:
                output_file, x_data, y_data = integrator.integrate_file(
                    h5_file,
                    output_dir,
                    npt=params['npt'],
                    unit=params['unit'],
                    output_format='xy',
                    azimuth_range=(azim_start, azim_end),
                    sector_label=sector_label
                )

                output_files.append(output_file)
                self.log(f"   ✓ Saved: {os.path.basename(output_file)}")

                csv_data[filename] = {
                    'x': x_data,
                    'y': y_data
                }

            except Exception as e:
                self.log(f"   ⚠️ Error processing {filename}: {str(e)}")
                continue

        # Save CSV if requested
        if params['output_csv'] and csv_data and not self.stop_processing:
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            self._save_csv(csv_data, csv_path, sector_label, params['unit'])
            output_files.append(csv_path)
            self.log(f"   💾 CSV saved: {csv_filename}")

        return output_files

    def _save_csv(self, csv_data, csv_path, sector_label, unit):
        """Save integrated data to CSV format"""
        if not csv_data:
            return

        first_key = list(csv_data.keys())[0]
        x_values = csv_data[first_key]['x']

        df_dict = {unit: x_values}

        for filename, data in csv_data.items():
            base_name = os.path.splitext(filename)[0]
            df_dict[base_name] = data['y']

        df = pd.DataFrame(df_dict)
        df.to_csv(csv_path, index=False)

    def _get_preset_sectors(self, preset_name):
        """Get preset sector configurations"""
        presets = {
            'quadrants': [
                (0, 90, "Q1_0-90"),
                (90, 180, "Q2_90-180"),
                (180, 270, "Q3_180-270"),
                (270, 360, "Q4_270-360")
            ],
            'octants': [
                (0, 45, "Oct1_0-45"),
                (45, 90, "Oct2_45-90"),
                (90, 135, "Oct3_90-135"),
                (135, 180, "Oct4_135-180"),
                (180, 225, "Oct5_180-225"),
                (225, 270, "Oct6_225-270"),
                (270, 315, "Oct7_270-315"),
                (315, 360, "Oct8_315-360")
            ],
            'hemispheres': [
                (0, 180, "Right_Hemisphere"),
                (180, 360, "Left_Hemisphere")
            ]
        }
        return presets.get(preset_name, [])


# ============================================================================
# Standalone Entry Point (for testing)
# ============================================================================

def main():
    """Standalone entry point for testing the module"""
    root = tk.Tk()
    root.title("XRD Azimuthal Integration Tool")
    root.geometry("900x900")
    root.configure(bg='#F8F3FF')

    # Create main frame
    main_frame = tk.Frame(root, bg='#F8F3FF')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

    # Initialize module
    app = AzimuthalIntegrationModule(main_frame, root)
    app.setup_ui()

    root.mainloop()


if __name__ == "__main__":
    main()
