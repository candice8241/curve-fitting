#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XRD Azimuthal Integration Module with GUI
==========================================
Author: candicewang928@gmail.com
Created: Nov 15, 2025
Modified: Nov 16, 2025 - Ultimate Performance Fix (Incremental Updates)
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
# GUI Module with INCREMENTAL UPDATE STRATEGY
# ============================================================================

class AzimuthalIntegrationModule(GUIBase):
    """Azimuthal Integration module - INCREMENTAL UPDATE (NO REBUILD)"""

    def __init__(self, parent, root):
        super().__init__()
        self.parent = parent
        self.root = root
        self._init_variables()
        self.processing = False
        self.stop_processing = False
        self.custom_sectors = []
        self.sector_row_widgets = []  # Track row widgets for incremental updates

    def _init_variables(self):
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
        for widget in self.parent.winfo_children():
            widget.destroy()
        self._create_title_section()
        self._create_reference_section()
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
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        if self.mode.get() == 'single':
            self._setup_single_sector_ui()
        else:
            self._setup_multiple_sectors_ui()

    def _setup_single_sector_ui(self):
        """Single sector with simple Entry fields"""
        angle_frame = tk.Frame(self.dynamic_frame, bg=self.colors['card_bg'])
        angle_frame.pack(fill=tk.X, pady=(10, 0))

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
        for widget in self.submode_frame.winfo_children():
            widget.destroy()
        self.sector_row_widgets = []  # Clear row widgets tracking

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
        """Custom sectors - INCREMENTAL UPDATE STRATEGY (NO REBUILD)"""
        if not self.custom_sectors:
            self.custom_sectors = [
                [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1")],
                [tk.DoubleVar(value=90.0), tk.DoubleVar(value=180.0), tk.StringVar(value="Sector_2")],
                [tk.DoubleVar(value=180.0), tk.DoubleVar(value=270.0), tk.StringVar(value="Sector_3")],
                [tk.DoubleVar(value=270.0), tk.DoubleVar(value=360.0), tk.StringVar(value="Sector_4")]
            ]

        # Main container
        self.custom_center_all = tk.Frame(self.submode_frame, bg=self.colors['card_bg'])
        self.custom_center_all.pack(expand=True, anchor='center')

        # Warning box
        instruction_frame = tk.Frame(self.custom_center_all, bg='#FFF4DC',
                                     relief='solid', borderwidth=1, padx=15, pady=8)
        instruction_frame.pack(pady=(0, 15), anchor='center')

        tk.Label(instruction_frame,
                text="💡 Define custom azimuthal sectors. Add multiple rows for different angular ranges.",
                bg='#FFF4DC', fg=self.colors['text_dark'],
                font=('Comic Sans MS', 9)).pack()

        # Sectors container
        self.sectors_container = tk.Frame(self.custom_center_all, bg=self.colors['card_bg'])
        self.sectors_container.pack(pady=(0, 15), anchor='center')

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

        # Create initial rows
        for idx in range(len(self.custom_sectors)):
            self._create_sector_row(idx)

    def _create_sector_row(self, idx):
        """Create a single sector row - INCREMENTAL approach"""
        sector = self.custom_sectors[idx]

        row_frame = tk.Frame(self.sectors_container, bg=self.colors['card_bg'])
        row_frame.pack(pady=3, anchor='center')

        # Store reference
        self.sector_row_widgets.append(row_frame)

        # Number label
        num_label = tk.Label(row_frame, text=f"#{idx+1}", bg=self.colors['card_bg'],
                            font=('Comic Sans MS', 10, 'bold'), width=3)
        num_label.pack(side=tk.LEFT, padx=(0, 8))

        # Start
        tk.Label(row_frame, text="Start:", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row_frame, textvariable=sector[0], width=7,
                font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 10))

        # End
        tk.Label(row_frame, text="End:", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row_frame, textvariable=sector[1], width=7,
                font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 10))

        # Label
        tk.Label(row_frame, text="Label:", bg=self.colors['card_bg'],
                font=('Comic Sans MS', 10)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row_frame, textvariable=sector[2],
                font=('Comic Sans MS', 10), width=15).pack(side=tk.LEFT, padx=(0, 10))

        # Delete button
        tk.Button(row_frame, text="✖", command=lambda i=idx: self._delete_sector(i),
                 bg='#E88C8C', fg='white', font=('Comic Sans MS', 9, 'bold'),
                 relief='flat', width=3, cursor='hand2').pack(side=tk.LEFT)

    def _add_sector(self):
        """Add sector - INCREMENTAL (only create new row)"""
        new_sector = [
            tk.DoubleVar(value=0.0),
            tk.DoubleVar(value=90.0),
            tk.StringVar(value=f"Sector_{len(self.custom_sectors) + 1}")
        ]
        self.custom_sectors.append(new_sector)

        # Only create the new row, don't rebuild everything
        self._create_sector_row(len(self.custom_sectors) - 1)

    def _delete_sector(self, index):
        """Delete sector - INCREMENTAL (only remove one row)"""
        if len(self.custom_sectors) <= 1:
            messagebox.showwarning("Warning", "At least one sector must be defined!")
            return

        # Remove from data
        del self.custom_sectors[index]

        # Destroy the specific row widget
        if index < len(self.sector_row_widgets):
            self.sector_row_widgets[index].destroy()
            del self.sector_row_widgets[index]

        # Update numbering for remaining rows
        self._renumber_sectors()

    def _renumber_sectors(self):
        """Update sector numbers without rebuilding"""
        for idx, row_widget in enumerate(self.sector_row_widgets):
            # Find the number label (first child)
            children = row_widget.winfo_children()
            if children:
                num_label = children[0]
                if isinstance(num_label, tk.Label):
                    num_label.config(text=f"#{idx+1}")

    def _clear_all_sectors(self):
        """Clear all sectors - rebuild entire list"""
        result = messagebox.askyesno("Confirm", "Clear all sectors and reset to default?")
        if result:
            # Clear all
            for row_widget in self.sector_row_widgets:
                row_widget.destroy()
            self.sector_row_widgets = []

            # Reset data
            self.custom_sectors = [
                [tk.DoubleVar(value=0.0), tk.DoubleVar(value=90.0), tk.StringVar(value="Sector_1")]
            ]

            # Create single default row
            self._create_sector_row(0)

    def log(self, message):
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
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already running!")
            return

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

        self.processing = True
        self.stop_processing = False

        threading.Thread(target=self._run_integration_thread,
                        args=(params,), daemon=True).start()

    def _run_integration_thread(self, params):
        try:
            def safe_progress_start():
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.start()
                except tk.TclError:
                    pass

            self.root.after(0, safe_progress_start)

            if params['mode'] == 'single':
                self.log("🥝 Starting Single Sector Azimuthal Integration")
                self._run_single_sector(params)
            else:
                self.log("🍋 Starting Multiple Sectors Azimuthal Integration")
                self._run_multiple_sectors(params)

            if not self.stop_processing:
                self.log("🍇 Azimuthal integration completed!")
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    "Azimuthal integration completed successfully!"))

        except Exception as e:
            if not self.stop_processing:
                import traceback
                error_details = traceback.format_exc()
                self.log(f"🐤 Error: {str(e)}")
                self.log(f"\nDetails:\n{error_details}")
                self.root.after(0, lambda err=str(e): messagebox.showerror("Error",
                    f"Azimuthal integration failed:\n{err}"))

        finally:
            def safe_progress_stop():
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.stop()
                except tk.TclError:
                    pass

            self.root.after(0, safe_progress_stop)
            self.processing = False

    def _run_single_sector(self, params):
        self.log(f"🍑 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🧁 Mask file: {os.path.basename(params['mask_path'])}")

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
        self.log(f"🍦 PONI file: {os.path.basename(params['poni_path'])}")
        if params['mask_path']:
            self.log(f"🍦 Mask file: {os.path.basename(params['mask_path'])}")

        sector_list = params['sectors']

        if 'preset_name' in params:
            self.log(f"🌻 Using preset: {params['preset_name']}")
        else:
            self.log(f"🦄 Using custom sectors")

        self.log(f"🦄 Number of sectors: {len(sector_list)}")

        for start, end, label in sector_list:
            self.log(f"   - {label}: {start}° to {end}°")

        all_output_files = []
        for start, end, label in sector_list:
            if self.stop_processing:
                self.log("☁️ Processing stopped by user")
                break
            self.log(f"\n☁️ Processing {label}...")
            output_files = self._integrate_sector(params, start, end, label)
            all_output_files.extend(output_files)

        if not self.stop_processing:
            self.log(f"\n{'='*60}")
            self.log(f"✨ Integration complete!")
            self.log(f"🐧 Generated {len(all_output_files)} files total")
            self.log(f"🐧 Output directory: {params['output_dir']}")
            self.log(f"{'='*60}\n")

    def _integrate_sector(self, params, azim_start, azim_end, sector_label):
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
    root = tk.Tk()
    root.title("XRD Azimuthal Integration Tool")
    root.geometry("900x900")
    root.configure(bg='#F8F3FF')

    main_frame = tk.Frame(root, bg='#F8F3FF')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

    app = AzimuthalIntegrationModule(main_frame, root)
    app.setup_ui()

    root.mainloop()


if __name__ == "__main__":
    main()
