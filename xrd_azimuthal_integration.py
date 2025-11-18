#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XRD Azimuthal Integration Module with GUI - Enhanced with Bin Mode
====================================================================
Author: candicewang928@gmail.com
Created: Nov 15, 2025
"""


import os
import glob
import threading
import weakref
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

from theme_module import GUIBase, CuteSheepProgressBar, ModernTab, ModernButton


# ============================================================================
# Core Integration Class (unchanged)
# ============================================================================

class XRDAzimuthalIntegrator:
    """Class to handle azimuthal integration of XRD diffraction data."""

    def __init__(self, poni_file: str, mask_file: Optional[str] = None):
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.ai = None
        self.mask = None
        self._load_calibration()
        if mask_file:
            self._load_mask()

    def _load_calibration(self):
        if not os.path.exists(self.poni_file):
            raise FileNotFoundError(f"PONI file not found: {self.poni_file}")
        print(f"Loading calibration from: {self.poni_file}")
        self.ai = pyFAI.load(self.poni_file)
        print(f"  Detector: {self.ai.detector.name}")
        print(f"  Distance: {self.ai.dist * 1000:.2f} mm")
        print(f"  Wavelength: {self.ai.wavelength * 1e10:.4f} Å")

    def _load_mask(self):
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
                      npt: int = 2048, unit: str = "q_A^-1",
                      output_format: str = "xy",
                      azimuth_range: Optional[tuple] = None,
                      sector_label: str = "") -> Tuple[str, np.ndarray, np.ndarray]:
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
        print(f"\nProcessing: {os.path.basename(h5_file)}")
        data = self._read_h5_data(h5_file)
        print(f"  Integrating with {npt} points, unit={unit}")
        if azimuth_range:
            print(f"  Azimuthal range: {azimuth_range[0]}° to {azimuth_range[1]}°")
        result = self.ai.integrate1d(data, npt=npt, mask=self.mask, unit=unit,
                                     method="splitpixel", error_model="poisson",
                                     azimuth_range=azimuth_range)
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
        with h5py.File(h5_file, 'r') as f:
            if dataset_path and dataset_path in f:
                data = f[dataset_path][...]
                if data.ndim == 3:
                    data = data[0]
                return data
            common_paths = ['entry/data/data', 'entry/instrument/detector/data',
                          'data', 'image', 'diffraction']
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
# GUI Module with Thread Safety (Fixed for Jupyter/IPython)
# ============================================================================

class AzimuthalIntegrationModule(GUIBase):
    """Azimuthal Integration module - Thread-safe version"""

    def __init__(self, parent, root):
        super().__init__()
        self.parent = parent
        self.root = root
        self._cleanup_lock = threading.Lock()
        self._is_destroyed = False
        self._init_variables()
        self.processing = False
        self.stop_processing = False
        self.custom_sectors = []
        self.sector_row_widgets = []

        # Track trace IDs for proper cleanup
        self._trace_ids = []

        # Register cleanup on window close
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        except:
            pass

    def _on_closing(self):
        """Clean shutdown handler"""
        with self._cleanup_lock:
            self._is_destroyed = True
            self.stop_processing = True

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def _init_variables(self):
        """Initialize all tkinter variables with thread safety"""
        try:
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

            # Bin mode variables
            self.bin_mode = tk.BooleanVar(value=False)
            self.bin_start = tk.DoubleVar(value=0.0)
            self.bin_end = tk.DoubleVar(value=360.0)
            self.bin_step = tk.DoubleVar(value=10.0)

            # Multi bin mode
            self.multi_bin_mode = tk.BooleanVar(value=False)
        except Exception as e:
            print(f"Warning: Error initializing variables: {e}")

    def _remove_all_traces(self):
        """Remove all registered trace callbacks to prevent accumulation"""
        for var, trace_id in self._trace_ids:
            try:
                var.trace_remove('write', trace_id)
            except:
                pass
        self._trace_ids = []

    def _add_trace(self, var, callback):
        """Add a trace and register it for later cleanup"""
        try:
            trace_id = var.trace_add('write', callback)
            self._trace_ids.append((var, trace_id))
        except:
            pass

    def setup_ui(self):
        """Setup UI with error handling"""
        try:
            for widget in self.parent.winfo_children():
                widget.destroy()
        except:
            pass

        self._create_title_section()
        self._create_reference_section()
        self._create_io_section()
        self._create_azimuthal_section()
        self._create_run_button_section()
        self._create_progress_section()
        self._create_log_section()

    def _create_title_section(self):
        """CENTERED title with larger font"""
        title_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        title_card = self.create_card_frame(title_frame)
        title_card.pack(fill=tk.X)

        content = tk.Frame(title_card, bg=self.colors['card_bg'], padx=20, pady=15)
        content.pack(fill=tk.X)

        center_container = tk.Frame(content, bg=self.colors['card_bg'])
        center_container.pack(expand=True)

        tk.Label(center_container, text="🎀", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 26)).pack(side=tk.LEFT, padx=(0, 10))

        text_frame = tk.Frame(center_container, bg=self.colors['card_bg'])
        text_frame.pack(side=tk.LEFT)

        tk.Label(text_frame, text="Azimuthal Integration",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 18, 'bold')).pack()

        tk.Label(text_frame, text="Integrate diffraction rings over selected azimuthal angle ranges",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Comic Sans MS', 11)).pack()

    def _create_reference_section(self):
        """Reference with larger font and CENTERED"""
        ref_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        ref_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        ref_card = self.create_card_frame(ref_frame)
        ref_card.pack(fill=tk.X)

        content = tk.Frame(ref_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.X)

        center_container = tk.Frame(content, bg=self.colors['card_bg'])
        center_container.pack(expand=True)

        tk.Label(center_container, text="🍓 Azimuthal Angle Reference:",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 10, 'bold')).pack()

        ref_text = "0° = Right (→)  |  90° = Top (↑)  |  180° = Left (←)  |  270° = Bottom (↓)"
        tk.Label(center_container, text=ref_text,
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10)).pack(pady=(5, 0))

        tk.Label(center_container, text="Counter-clockwise rotation from right horizontal",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Comic Sans MS', 9, 'italic')).pack()

    def _create_io_section(self):
        """Input/Output file configuration section"""
        io_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        io_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        card = self.create_card_frame(io_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 10))

        tk.Label(header, text="📁", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="File Configuration",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 10, 'bold')).pack(side=tk.LEFT)

        # PONI file
        poni_row = tk.Frame(content, bg=self.colors['card_bg'])
        poni_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(poni_row, text="PONI Calibration File:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        poni_entry_frame = tk.Frame(poni_row, bg=self.colors['card_bg'])
        poni_entry_frame.pack(fill=tk.X)

        tk.Entry(poni_entry_frame, textvariable=self.poni_path,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        tk.Button(poni_entry_frame, text="Browse 🔍",
                 command=lambda: self._browse_file(self.poni_path, "PONI files", "*.poni"),
                 bg='#D8A7D8', fg='white',
                 font=('Comic Sans MS', 9, 'bold'),
                 relief='flat', padx=10, pady=5,
                 cursor='hand2').pack(side=tk.LEFT)

        # Mask file (optional)
        mask_row = tk.Frame(content, bg=self.colors['card_bg'])
        mask_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(mask_row, text="Mask File (Optional):",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        mask_entry_frame = tk.Frame(mask_row, bg=self.colors['card_bg'])
        mask_entry_frame.pack(fill=tk.X)

        tk.Entry(mask_entry_frame, textvariable=self.mask_path,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        tk.Button(mask_entry_frame, text="Browse 🔍",
                 command=lambda: self._browse_file(self.mask_path, "Mask files", "*.npy *.h5 *.edf"),
                 bg='#D8A7D8', fg='white',
                 font=('Comic Sans MS', 9, 'bold'),
                 relief='flat', padx=10, pady=5,
                 cursor='hand2').pack(side=tk.LEFT)

        # Input H5 files
        input_row = tk.Frame(content, bg=self.colors['card_bg'])
        input_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(input_row, text="Input H5 Files Pattern:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        input_entry_frame = tk.Frame(input_row, bg=self.colors['card_bg'])
        input_entry_frame.pack(fill=tk.X)

        tk.Entry(input_entry_frame, textvariable=self.input_pattern,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        tk.Button(input_entry_frame, text="Select Files 📂",
                 command=self._select_input_files,
                 bg='#D8A7D8', fg='white',
                 font=('Comic Sans MS', 9, 'bold'),
                 relief='flat', padx=10, pady=5,
                 cursor='hand2').pack(side=tk.LEFT)

        # Output directory
        output_row = tk.Frame(content, bg=self.colors['card_bg'])
        output_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(output_row, text="Output Directory:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        output_entry_frame = tk.Frame(output_row, bg=self.colors['card_bg'])
        output_entry_frame.pack(fill=tk.X)

        tk.Entry(output_entry_frame, textvariable=self.output_dir,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        tk.Button(output_entry_frame, text="Browse 📁",
                 command=self._browse_directory,
                 bg='#D8A7D8', fg='white',
                 font=('Comic Sans MS', 9, 'bold'),
                 relief='flat', padx=10, pady=5,
                 cursor='hand2').pack(side=tk.LEFT)

        # Advanced settings row
        advanced_row = tk.Frame(content, bg=self.colors['card_bg'])
        advanced_row.pack(fill=tk.X, pady=(8, 0))

        # Dataset path
        dataset_col = tk.Frame(advanced_row, bg=self.colors['card_bg'])
        dataset_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        tk.Label(dataset_col, text="HDF5 Dataset Path:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
        tk.Entry(dataset_col, textvariable=self.dataset_path,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat').pack(fill=tk.X)

        # Number of points
        npt_col = tk.Frame(advanced_row, bg=self.colors['card_bg'])
        npt_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        tk.Label(npt_col, text="Number of Points:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
        tk.Entry(npt_col, textvariable=self.npt,
                font=('Comic Sans MS', 10),
                bg='white', relief='flat', width=10).pack(anchor=tk.W)

        # Unit selection
        unit_col = tk.Frame(advanced_row, bg=self.colors['card_bg'])
        unit_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(unit_col, text="Unit:",
                bg=self.colors['card_bg'],
                fg=self.colors['text_dark'],
                font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
        ttk.Combobox(unit_col, textvariable=self.unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'd*2_A^-2'],
                    width=12, state='readonly',
                    font=('Comic Sans MS', 10)).pack(anchor=tk.W)

    def _browse_file(self, var, description, filetypes):
        """Helper method for file browsing with error handling"""
        try:
            filename = filedialog.askopenfilename(
                title=f"Select {description}",
                filetypes=[(description, filetypes), ("All files", "*.*")]
            )
            if filename:
                var.set(filename)
        except Exception as e:
            self.log(f"Error browsing file: {e}")

    def _select_input_files(self):
        """Helper method for selecting input H5 files with error handling"""
        try:
            files = filedialog.askopenfilenames(
                title="Select Input H5 Files",
                filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
            )
            if files:
                directory = os.path.dirname(files[0])
                pattern = os.path.join(directory, "*.h5")
                self.input_pattern.set(pattern)
        except Exception as e:
            self.log(f"Error selecting files: {e}")

    def _browse_directory(self):
        """Helper method for directory browsing with error handling"""
        try:
            directory = filedialog.askdirectory(title="Select Output Directory")
            if directory:
                self.output_dir.set(directory)
        except Exception as e:
            self.log(f"Error browsing directory: {e}")

    def _create_azimuthal_section(self):
        """Azimuthal settings"""
        azimuth_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        azimuth_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        card = self.create_card_frame(azimuth_frame)
        card.pack(fill=tk.X)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 10))

        tk.Label(header, text="🍰", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Azimuthal Angle Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 10, 'bold')).pack(side=tk.LEFT)

        # Mode selection
        mode_frame = tk.Frame(content, bg=self.colors['card_bg'])
        mode_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(mode_frame, text="Integration Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 6))

        mode_buttons = tk.Frame(mode_frame, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Single Sector", variable=self.mode,
                      value='single', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 10),
                      command=self.update_mode).pack(side=tk.LEFT, padx=(0, 25))

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 10),
                      command=self.update_mode).pack(side=tk.LEFT)

        self.dynamic_frame = tk.Frame(content, bg=self.colors['card_bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        self.update_mode()

    def _create_run_button_section(self):
        """Run button directly on background"""
        center_container = tk.Frame(self.parent, bg=self.colors['bg'])
        center_container.pack(fill=tk.X, pady=(0, 10))

        self.run_btn = tk.Button(center_container, text="🌸 Run Azimuthal Integration",
                           command=self.run_integration,
                           bg='#E89FE9', fg='white',
                           font=('Comic Sans MS', 10, 'bold'), relief='flat',
                           padx=12, pady=5, cursor='hand2')
        self.run_btn.pack()

    def _create_progress_section(self):
        prog_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        prog_frame.pack(fill=tk.X, padx=0, pady=(10, 10))

        self.progress_bar = CuteSheepProgressBar(prog_frame, width=780, height=80)
        self.progress_bar.pack()

    def _create_log_section(self):
        """Log with larger font"""
        log_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 20))

        card = self.create_card_frame(log_frame)
        card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=20, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(content, bg=self.colors['card_bg'])
        header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header, text="🧸", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 16)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 12, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(content, height=12, wrap=tk.WORD,
                                                  font=('Comic Sans MS', 10),
                                                  bg='#FAFAFA', fg=self.colors['primary'],
                                                  relief='flat', borderwidth=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def update_mode(self):
        # Remove all existing traces before updating UI
        self._remove_all_traces()

        try:
            for widget in self.dynamic_frame.winfo_children():
                widget.destroy()
        except:
            pass

        if self.mode.get() == 'single':
            self._setup_single_sector_ui()
        else:
            self._setup_multiple_sectors_ui()

    def _setup_single_sector_ui(self):
        """Single sector with bin mode option"""
        # Add bin mode toggle
        bin_toggle_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        bin_toggle_frame.pack(fill=tk.X, pady=(10, 15))

        tk.Checkbutton(bin_toggle_frame,
                       text="🍰 Enable Bin Mode (Divide range into multiple bins)",
                       variable=self.bin_mode,
                       bg=self.colors['card_bg'],
                       font=('Comic Sans MS', 10, 'bold'),
                       command=self._update_bin_mode_ui).pack(anchor=tk.W)

        # Dynamic frame for bin/normal mode
        self.bin_mode_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        self.bin_mode_frame.pack(fill=tk.X, pady=(0, 10))

        self._update_bin_mode_ui()

    def _update_bin_mode_ui(self):
        """Update UI based on bin mode selection"""
        # Remove traces before rebuilding UI
        self._remove_all_traces()

        try:
            for widget in self.bin_mode_frame.winfo_children():
                widget.destroy()
        except:
            pass

        # Process pending events to prevent flicker
        self.bin_mode_frame.update_idletasks()

        if self.bin_mode.get():
            # BIN MODE UI
            bin_container = tk.Frame(self.bin_mode_frame, bg=self.colors['card_bg'])
            bin_container.pack(fill=tk.X)

            # Row 1: Total range
            range_frame = tk.Frame(bin_container, bg=self.colors['card_bg'])
            range_frame.pack(fill=tk.X, pady=(0, 10))

            start_cont = tk.Frame(range_frame, bg=self.colors['card_bg'])
            start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            tk.Label(start_cont, text="Total Range Start (°)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(start_cont, textvariable=self.bin_start,
                    font=('Comic Sans MS', 10), width=10).pack(anchor=tk.W)

            end_cont = tk.Frame(range_frame, bg=self.colors['card_bg'])
            end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(end_cont, text="Total Range End (°)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(end_cont, textvariable=self.bin_end,
                    font=('Comic Sans MS', 10), width=10).pack(anchor=tk.W)

            # Row 2: Bin size and calculated bin count
            step_frame = tk.Frame(bin_container, bg=self.colors['card_bg'])
            step_frame.pack(fill=tk.X)

            step_cont = tk.Frame(step_frame, bg=self.colors['card_bg'])
            step_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            tk.Label(step_cont, text="Bin Size (°/bin)",
                    bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'],
                    font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(step_cont, textvariable=self.bin_step,
                    font=('Comic Sans MS', 10), width=10).pack(anchor=tk.W)

            # Display calculated bin count
            info_cont = tk.Frame(step_frame, bg=self.colors['card_bg'])
            info_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def calculate_bins():
                try:
                    total_range = self.bin_end.get() - self.bin_start.get()
                    step = self.bin_step.get()
                    if step > 0 and total_range > 0:
                        n_bins = int(np.ceil(total_range / step))
                        return f"✨ Will generate {n_bins} bins"
                    return "⚠️ Invalid parameters"
                except:
                    return "⚠️ Invalid parameters"

            self.bin_info_label = tk.Label(info_cont,
                                           text=calculate_bins(),
                                           bg=self.colors['card_bg'],
                                           fg=self.colors['primary'],
                                           font=('Comic Sans MS', 10, 'italic'))
            self.bin_info_label.pack(anchor=tk.W, pady=(14, 0))

            # Add trace to update bin count in real-time with proper tracking
            def update_bin_info(*args):
                if hasattr(self, 'bin_info_label'):
                    try:
                        self.bin_info_label.config(text=calculate_bins())
                    except:
                        pass

            self._add_trace(self.bin_start, update_bin_info)
            self._add_trace(self.bin_end, update_bin_info)
            self._add_trace(self.bin_step, update_bin_info)

        else:
            # NORMAL SINGLE SECTOR MODE UI
            angle_frame = tk.Frame(self.bin_mode_frame, bg=self.colors['card_bg'])
            angle_frame.pack(fill=tk.X)

            start_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            start_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            tk.Label(start_cont, text="Start Angle (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(start_cont, textvariable=self.azimuth_start,
                    font=('Comic Sans MS', 10), width=10).pack(anchor=tk.W)

            end_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            end_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            tk.Label(end_cont, text="End Angle (°)", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(end_cont, textvariable=self.azimuth_end,
                    font=('Comic Sans MS', 10), width=10).pack(anchor=tk.W)

            label_cont = tk.Frame(angle_frame, bg=self.colors['card_bg'])
            label_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(label_cont, text="Sector Label", bg=self.colors['card_bg'],
                    fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))
            tk.Entry(label_cont, textvariable=self.sector_label,
                    font=('Comic Sans MS', 10), width=24).pack(anchor=tk.W)

    def _setup_multiple_sectors_ui(self):
        """Multiple sectors"""
        main_container = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        main_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # LEFT SIDE: Mode selector
        left_side = tk.Frame(main_container, bg=self.colors['card_bg'])
        left_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 20))

        tk.Label(left_side, text="Multiple Sectors Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 6))

        mode_buttons = tk.Frame(left_side, bg=self.colors['card_bg'])
        mode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(mode_buttons, text="Preset Templates", variable=self.multiple_mode,
                      value='preset', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 10),
                      command=self.update_multiple_submode).pack(anchor=tk.W, pady=2)

        tk.Radiobutton(mode_buttons, text="Custom Sectors", variable=self.multiple_mode,
                      value='custom', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 10),
                      command=self.update_multiple_submode).pack(anchor=tk.W, pady=2)

        # RIGHT SIDE
        self.submode_frame = tk.Frame(main_container, bg=self.colors['card_bg'])
        self.submode_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.update_multiple_submode()

    def update_multiple_submode(self):
        # Remove all existing traces before updating UI
        self._remove_all_traces()

        try:
            for widget in self.submode_frame.winfo_children():
                widget.destroy()
            self.sector_row_widgets = []
        except:
            pass

        if self.multiple_mode.get() == 'preset':
            self._setup_preset_mode()
        else:
            self._setup_custom_sectors_mode()

    def _setup_preset_mode(self):
        """Preset mode"""
        preset_frame = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        preset_frame.pack(fill=tk.X, pady=(5, 0), anchor=tk.W)

        tk.Label(preset_frame, text="Select Preset:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 10, 'bold')).pack(anchor=tk.W, pady=(0, 4))

        ttk.Combobox(preset_frame, textvariable=self.preset,
                    values=['quadrants', 'octants', 'hemispheres'],
                    width=28, state='readonly', font=('Comic Sans MS', 10)).pack(anchor=tk.W)

        preset_info = {
            'quadrants': "4 sectors: 0-90°, 90-180°, 180-270°, 270-360°",
            'octants': "8 sectors: Every 45° from 0° to 360°",
            'hemispheres': "2 sectors: 0-180° (Right), 180-360° (Left)"
        }

        info_text = preset_info.get(self.preset.get(), "Select a preset")
        tk.Label(preset_frame, text=f"🍓 {info_text}",
                bg=self.colors['card_bg'], fg=self.colors['text_light'],
                font=('Comic Sans MS', 9, 'italic')).pack(anchor=tk.W, pady=(8, 0))

    def _setup_custom_sectors_mode(self):
        """Custom sectors with BIN MODE support"""
        if not self.custom_sectors:
            try:
                self.custom_sectors = [
                    [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1"), tk.DoubleVar(value=10.0)],
                    [tk.DoubleVar(value=90.0), tk.DoubleVar(value=180.0), tk.StringVar(value="Sector_2"), tk.DoubleVar(value=10.0)],
                    [tk.DoubleVar(value=180.0), tk.DoubleVar(value=270.0), tk.StringVar(value="Sector_3"), tk.DoubleVar(value=10.0)],
                    [tk.DoubleVar(value=270.0), tk.DoubleVar(value=360.0), tk.StringVar(value="Sector_4"), tk.DoubleVar(value=10.0)]
                ]
            except Exception as e:
                print(f"Error initializing custom sectors: {e}")
                return

        # Main container
        self.custom_center_all = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        self.custom_center_all.pack(expand=True, anchor='center')

        # Bin mode toggle
        bin_toggle_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        bin_toggle_frame.pack(pady=(0, 10), anchor='center')

        tk.Checkbutton(bin_toggle_frame,
                       text="🍰 Enable Bin Mode (Each sector will be divided into bins)",
                       variable=self.multi_bin_mode,
                       bg=self.colors['card_bg'],
                       font=('Comic Sans MS', 10, 'bold'),
                       command=self._update_custom_sectors_display).pack()

        # Warning box
        instruction_frame = tk.Frame(self.custom_center_all, bg='#FFF4DC',
                                     relief='solid', borderwidth=1, padx=15, pady=8)
        instruction_frame.pack(pady=(0, 15), anchor='center')

        self.custom_instruction_label = tk.Label(instruction_frame,
                text="💡 Define custom azimuthal sectors. Add multiple rows for different angular ranges.",
                bg='#FFF4DC', fg=self.colors['text_dark'],
                font=('Comic Sans MS', 9))
        self.custom_instruction_label.pack()

        # Sectors container
        sectors_outer_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        sectors_outer_frame.pack(pady=(0, 15), anchor='center')

        self.sectors_spacer = tk.Frame(sectors_outer_frame, bg=self.colors['card_bg'],
                                       height=180, width=1)
        self.sectors_spacer.pack(side=tk.LEFT)

        self.sectors_container = tk.Frame(sectors_outer_frame, bg=self.colors['card_bg'])
        self.sectors_container.pack(side=tk.LEFT, anchor='center')

        # Buttons
        btn_frame = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        btn_frame.pack(anchor='center')

        tk.Button(btn_frame, text="🐾 Add Sector", command=self._add_sector,
                 bg='#D8A7D8', fg='white',
                 font=('Comic Sans MS', 10, 'bold'), relief='flat',
                 padx=6, pady=7, cursor='hand2').pack(side=tk.LEFT, padx=15)

        tk.Button(btn_frame, text="🍉 Clear All", command=self._clear_all_sectors,
                 bg='#FF9FB5', fg='white',
                 font=('Comic Sans MS', 10, 'bold'), relief='flat',
                 padx=6, pady=7, cursor='hand2').pack(side=tk.LEFT, padx=15)

        self._update_custom_sectors_display()

        for idx in range(len(self.custom_sectors)):
            self._create_sector_row(idx)

    def _update_custom_sectors_display(self):
        """Update instruction text and recreate sector rows when bin mode changes"""
        # Remove all existing traces before rebuilding
        self._remove_all_traces()

        if hasattr(self, 'custom_instruction_label'):
            try:
                if self.multi_bin_mode.get():
                    self.custom_instruction_label.config(
                        text="💡 Define sectors with bin size. Each sector will be divided into multiple bins.")
                else:
                    self.custom_instruction_label.config(
                        text="💡 Define custom azimuthal sectors. Add multiple rows for different angular ranges.")
            except:
                pass

        # Recreate all sector rows
        if hasattr(self, 'sectors_container'):
            try:
                # Process pending events to prevent flicker
                self.sectors_container.update_idletasks()

                for widget in self.sector_row_widgets:
                    widget.destroy()
                self.sector_row_widgets = []

                for idx in range(len(self.custom_sectors)):
                    self._create_sector_row(idx)
            except:
                pass

    def _create_sector_row(self, idx):
        """
        Create a single sector row with optional bin size display

        Args:
            idx: Index of the sector in the custom_sectors list

        Features:
            - Displays sector number, start/end angles, label
            - Shows bin size input when multi_bin_mode is enabled
            - Real-time calculation and display of number of bins to be generated
            - Automatic updates when any parameter changes
        """
        try:
            sector = self.custom_sectors[idx]

            # Ensure sector has bin_step variable (for backward compatibility)
            if len(sector) == 3:
                sector.append(tk.DoubleVar(value=10.0))
                self.custom_sectors[idx] = sector

            # Main row container
            row_frame = tk.Frame(self.sectors_container, bg=self.colors['card_bg'])
            row_frame.pack(pady=3, anchor='center')

            self.sector_row_widgets.append(row_frame)

            # Sector number label (e.g., "#1", "#2")
            num_label = tk.Label(row_frame, text=f"#{idx+1}", bg=self.colors['card_bg'],
                                font=('Comic Sans MS', 10, 'bold'), width=3)
            num_label.pack(side=tk.LEFT, padx=(0, 8))

            # Start angle input
            tk.Label(row_frame, text="Start:", bg=self.colors['card_bg'],
                    font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
            tk.Entry(row_frame, textvariable=sector[0], width=7,
                    font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 10))

            # End angle input
            tk.Label(row_frame, text="End:", bg=self.colors['card_bg'],
                    font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
            tk.Entry(row_frame, textvariable=sector[1], width=7,
                    font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 10))

            # Sector label input
            tk.Label(row_frame, text="Label:", bg=self.colors['card_bg'],
                    font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
            tk.Entry(row_frame, textvariable=sector[2],
                    font=('Comic Sans MS', 10), width=12).pack(side=tk.LEFT, padx=(0, 10))

            # Bin size input and bin count display (only shown when bin mode is enabled)
            if self.multi_bin_mode.get():
                # Bin size label and entry
                tk.Label(row_frame, text="Bin:", bg=self.colors['card_bg'],
                        font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
                tk.Entry(row_frame, textvariable=sector[3], width=5,
                        font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 10))

                # Function to calculate number of bins for this sector
                def calculate_sector_bins(s=sector):
                    """
                    Calculate how many bins will be generated for this sector

                    Returns:
                        str: Formatted string showing number of bins or warning symbol

                    Formula:
                        n_bins = ceil((end_angle - start_angle) / bin_size)
                    """
                    try:
                        start = s[0].get()
                        end = s[1].get()
                        step = s[3].get()
                        total_range = end - start

                        # Validate parameters
                        if step > 0 and total_range > 0:
                            n_bins = int(np.ceil(total_range / step))
                            return f"✨ {n_bins} bins"
                        return "⚠️"
                    except:
                        return "⚠️"

                # Create label to display bin count
                bin_count_label = tk.Label(row_frame,
                                           text=calculate_sector_bins(),
                                           bg=self.colors['card_bg'],
                                           fg=self.colors['primary'],
                                           font=('Comic Sans MS', 9, 'italic'),
                                           width=10)
                bin_count_label.pack(side=tk.LEFT, padx=(0, 10))

                # Callback function to update bin count display in real-time
                def update_bin_count(label=bin_count_label, s=sector):
                    def callback(*args):
                        """
                        Update the bin count label when any parameter changes
                        """
                        try:
                            if label.winfo_exists():
                                start = s[0].get()
                                end = s[1].get()
                                step = s[3].get()
                                total_range = end - start
                                if step > 0 and total_range > 0:
                                    n_bins = int(np.ceil(total_range / step))
                                    label.config(text=f"✨ {n_bins} bins")
                                else:
                                    label.config(text="⚠️")
                        except:
                            pass
                    return callback

                # Add trace callbacks with proper tracking
                callback = update_bin_count()
                self._add_trace(sector[0], callback)
                self._add_trace(sector[1], callback)
                self._add_trace(sector[3], callback)

            # Delete button to remove this sector
            tk.Button(row_frame, text="✖", command=lambda i=idx: self._delete_sector(i),
                     bg='#E88C8C', fg='white', font=('Comic Sans MS', 9, 'bold'),
                     relief='flat', width=3, cursor='hand2').pack(side=tk.LEFT)

        except Exception as e:
            print(f"Error creating sector row {idx}: {e}")

    def _add_sector(self):
        """
        Add a new sector row to custom sectors list

        Creates a new sector with default values:
        - Start: 0.0°
        - End: 90.0°
        - Label: "Sector_N" (where N is the new sector number)
        - Bin Size: 10.0° (default bin size)
        """
        try:
            new_sector = [
                tk.DoubleVar(value=0.0),
                tk.DoubleVar(value=90.0),
                tk.StringVar(value=f"Sector_{len(self.custom_sectors) + 1}"),
                tk.DoubleVar(value=10.0)  # Default bin size
            ]
            self.custom_sectors.append(new_sector)
            self._create_sector_row(len(self.custom_sectors) - 1)
        except Exception as e:
            self.log(f"Error adding sector: {e}")

    def _delete_sector(self, index):
        """
        Delete a sector from the custom sectors list

        Args:
            index: Index of the sector to delete

        Behavior:
            - Prevents deletion if only one sector remains
            - Updates sector numbers after deletion
            - Safely destroys widgets and removes from list

        Note:
            At least one sector must always be defined
        """
        if len(self.custom_sectors) <= 1:
            messagebox.showwarning("Warning", "At least one sector must be defined!")
            return

        try:
            # Force UI update before deletion
            self.sectors_container.update_idletasks()

            # Remove from data structure
            del self.custom_sectors[index]

            # Remove widget and update UI
            if index < len(self.sector_row_widgets):
                row_widget = self.sector_row_widgets[index]
                row_widget.pack_forget()
                del self.sector_row_widgets[index]

                # Update sector numbers
                self._renumber_sectors()

                # Force UI update and destroy widget
                self.sectors_container.update_idletasks()
                row_widget.destroy()

        except Exception as e:
            self.log(f"Error deleting sector: {e}")

    def _renumber_sectors(self):
        """
        Update sector number labels after deletion or reordering

        Iterates through all sector row widgets and updates their
        number labels to reflect current positions (e.g., #1, #2, #3)

        Note:
            This ensures sequential numbering after any sector is deleted
        """
        try:
            for idx, row_widget in enumerate(self.sector_row_widgets):
                children = row_widget.winfo_children()
                if children:
                    num_label = children[0]
                    if isinstance(num_label, tk.Label):
                        num_label.config(text=f"#{idx+1}")
        except Exception as e:
            print(f"Error renumbering sectors: {e}")

    def _clear_all_sectors(self):
        """
        Clear all sectors and reset to default single sector

        Behavior:
            - Shows confirmation dialog before clearing
            - Destroys all existing sector widgets
            - Creates a single default sector (0-90°)

        Default Sector:
            - Start: 0.0°
            - End: 90.0°
            - Label: "Sector_1"
            - Bin Size: 10.0°
        """
        result = messagebox.askyesno("Confirm", "Clear all sectors and reset to default?")
        if result:
            try:
                # Remove traces before clearing
                self._remove_all_traces()

                # Destroy all existing sector widgets
                for row_widget in self.sector_row_widgets:
                    row_widget.destroy()
                self.sector_row_widgets = []

                # Reset to single default sector
                self.custom_sectors = [
                    [tk.DoubleVar(value=0.0),
                     tk.DoubleVar(value=90.0),
                     tk.StringVar(value="Sector_1"),
                     tk.DoubleVar(value=10.0)]
                ]

                # Create the default sector row
                self._create_sector_row(0)

            except Exception as e:
                self.log(f"Error clearing sectors: {e}")

    def log(self, message):
        """Thread-safe logging"""
        def _log():
            with self._cleanup_lock:
                if self._is_destroyed:
                    return

            try:
                if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
            except (tk.TclError, RuntimeError):
                pass

        if threading.current_thread() is threading.main_thread():
            _log()
        else:
            try:
                self.root.after(0, _log)
            except (tk.TclError, RuntimeError):
                pass

    def run_integration(self):
        """Run integration with validation"""
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already running!")
            return

        try:
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

            if params['mode'] == 'single':
                if self.bin_mode.get():
                    # BIN MODE
                    bin_start = self.bin_start.get()
                    bin_end = self.bin_end.get()
                    bin_step = self.bin_step.get()

                    if bin_step <= 0:
                        messagebox.showerror("Error", "Bin size must be positive!")
                        return

                    if bin_start >= bin_end:
                        messagebox.showerror("Error", "Start angle must be less than end angle!")
                        return

                    sectors = []
                    current = bin_start
                    bin_idx = 1

                    while current < bin_end:
                        next_angle = min(current + bin_step, bin_end)
                        label = f"Bin{bin_idx:03d}_{current:.1f}-{next_angle:.1f}"
                        sectors.append((float(current), float(next_angle), label))
                        current = next_angle
                        bin_idx += 1

                    params['sectors'] = sectors
                    params['bin_mode'] = True
                else:
                    # NORMAL SINGLE SECTOR
                    params['sectors'] = [(
                        float(self.azimuth_start.get()),
                        float(self.azimuth_end.get()),
                        str(self.sector_label.get())
                    )]
                    params['bin_mode'] = False
            else:
                # MULTIPLE SECTORS
                if self.multiple_mode.get() == 'preset':
                    preset_name = self.preset.get()
                    params['sectors'] = self._get_preset_sectors(preset_name)
                    params['preset_name'] = preset_name
                    params['bin_mode'] = False
                else:
                    # CUSTOM SECTORS with optional BIN MODE
                    if self.multi_bin_mode.get():
                        all_sectors = []
                        for sector_data in self.custom_sectors:
                            start = sector_data[0].get()
                            end = sector_data[1].get()
                            base_label = sector_data[2].get()
                            bin_step = sector_data[3].get()

                            if bin_step <= 0:
                                messagebox.showerror("Error", f"Bin size for {base_label} must be positive!")
                                return

                            current = start
                            bin_idx = 1
                            while current < end:
                                next_angle = min(current + bin_step, end)
                                label = f"{base_label}_Bin{bin_idx:02d}_{current:.1f}-{next_angle:.1f}"
                                all_sectors.append((float(current), float(next_angle), label))
                                current = next_angle
                                bin_idx += 1

                        params['sectors'] = all_sectors
                        params['bin_mode'] = True
                    else:
                        sectors = []
                        for sector_data in self.custom_sectors:
                            start = sector_data[0].get()
                            end = sector_data[1].get()
                            label = sector_data[2].get()
                            sectors.append((float(start), float(end), str(label)))
                        params['sectors'] = sectors
                        params['bin_mode'] = False

            self.processing = True
            self.stop_processing = False

            threading.Thread(target=self._run_integration_thread,
                            args=(params,), daemon=True).start()
        except Exception as e:
            self.log(f"Error starting integration: {e}")
            messagebox.showerror("Error", f"Failed to start integration:\n{e}")

    def _run_integration_thread(self, params):
        """Thread-safe integration runner"""
        try:
            def safe_progress_start():
                with self._cleanup_lock:
                    if self._is_destroyed:
                        return
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.start()
                except (tk.TclError, RuntimeError):
                    pass

            self.root.after(0, safe_progress_start)

            if params['mode'] == 'single':
                if params.get('bin_mode', False):
                    self.log("🍰 Starting Bin Mode Azimuthal Integration")
                else:
                    self.log("🥝 Starting Single Sector Azimuthal Integration")
                self._run_single_sector(params)
            else:
                if params.get('bin_mode', False):
                    self.log("🍰 Starting Multiple Sectors with Bin Mode")
                else:
                    self.log("🍋 Starting Multiple Sectors Azimuthal Integration")
                self._run_multiple_sectors(params)

            if not self.stop_processing:
                self.log("🍇 Azimuthal integration completed!")

                def show_success():
                    with self._cleanup_lock:
                        if self._is_destroyed:
                            return
                    try:
                        messagebox.showinfo("Success", "Azimuthal integration completed successfully!")
                    except:
                        pass

                self.root.after(0, show_success)

        except Exception as e:
            if not self.stop_processing:
                import traceback
                error_details = traceback.format_exc()
                error_msg = str(e)  # Save error message first

                self.log(f"🐤 Error: {error_msg}")
                self.log(f"\nDetails:\n{error_details}")

                def show_error():
                    with self._cleanup_lock:
                        if self._is_destroyed:
                            return
                    try:
                        messagebox.showerror("Error", f"Azimuthal integration failed:\n{error_msg}")
                    except:
                        pass

                self.root.after(0, show_error)

        finally:
            def safe_progress_stop():
                with self._cleanup_lock:
                    if self._is_destroyed:
                        return
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.stop()
                except (tk.TclError, RuntimeError):
                    pass

            self.root.after(0, safe_progress_stop)
            self.processing = False

    def _run_single_sector(self, params):
        """Run single sector integration"""
        self.log(f"🍑 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🧁 Mask file: {os.path.basename(params['mask_path'])}")

        if params.get('bin_mode', False):
            self.log(f"🍰 Bin Mode: Processing {len(params['sectors'])} bins")
            all_output_files = []

            for idx, (azim_start, azim_end, sector_label) in enumerate(params['sectors']):
                if self.stop_processing:
                    self.log("☁️ Processing stopped by user")
                    break

                self.log(f"\n🌈 Bin {idx+1}/{len(params['sectors'])}: {azim_start}° to {azim_end}° ({sector_label})")
                output_files = self._integrate_sector(params, azim_start, azim_end, sector_label)
                all_output_files.extend(output_files)

            if not self.stop_processing:
                self.log(f"\n{'='*60}")
                self.log(f"✨ Bin integration complete!")
                self.log(f"🐨 Generated {len(all_output_files)} files from {len(params['sectors'])} bins")
                self.log(f"🐨 Output directory: {params['output_dir']}")
                self.log(f"{'='*60}\n")
        else:
            azim_start, azim_end, sector_label = params['sectors'][0]
            self.log(f"🌈 Azimuthal range: {azim_start}° to {azim_end}°")
            self.log(f"🌈 Sector label: {sector_label}")

            output_files = self._integrate_sector(params, azim_start, azim_end, sector_label)

            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"🐨 Generated {len(output_files)} files")
            self.log(f"🐨 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _run_multiple_sectors(self, params):
        """Run multiple sectors integration"""
        self.log(f"🍦 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🍦 Mask file: {os.path.basename(params['mask_path'])}")

        sector_list = params['sectors']

        if 'preset_name' in params:
            self.log(f"🌻 Using preset: {params['preset_name']}")
        else:
            if params.get('bin_mode', False):
                self.log(f"🦄 Using custom sectors with bin mode")
            else:
                self.log(f"🦄 Using custom sectors")

        self.log(f"🦄 Number of sectors/bins: {len(sector_list)}")

        for start, end, label in sector_list[:5]:
            self.log(f"   - {label}: {start}° to {end}°")
        if len(sector_list) > 5:
            self.log(f"   ... and {len(sector_list) - 5} more")

        all_output_files = []
        for idx, (start, end, label) in enumerate(sector_list):
            if self.stop_processing:
                self.log("☁️ Processing stopped by user")
                break
            self.log(f"\n☁️ [{idx+1}/{len(sector_list)}] Processing {label}...")
            output_files = self._integrate_sector(params, start, end, label)
            all_output_files.extend(output_files)

        if not self.stop_processing:
            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"🐧 Generated {len(all_output_files)} files total")
            self.log(f"🐧 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _integrate_sector(self, params, azim_start, azim_end, sector_label):
        """Integrate a single sector"""
        integrator = XRDAzimuthalIntegrator(
            params['poni_path'],
            params['mask_path']
        )

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

        if params['output_csv'] and csv_data and not self.stop_processing:
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            self._save_csv(csv_data, csv_path, sector_label, params['unit'])
            output_files.append(csv_path)
            self.log(f"   🥥 CSV saved: {csv_filename}")

        return output_files

    def _save_csv(self, csv_data, csv_path, sector_label, unit):
        """Save CSV file"""
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
# Standalone Entry Point
# ============================================================================

def main():
    """Main entry point with error handling"""
    root = tk.Tk()
    root.title("XRD Azimuthal Integration Tool")
    root.geometry("900x1000")
    root.configure(bg='#F8F3FF')

    main_frame = tk.Frame(root, bg='#F8F3FF')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

    app = AzimuthalIntegrationModule(main_frame, root)
    app.setup_ui()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    finally:
        try:
            root.quit()
        except:
            pass


if __name__ == "__main__":
    main()
