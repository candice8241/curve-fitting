# -*- coding: utf-8 -*-
"""
Radial XRD Module (Azimuthal Integration)
Contains azimuthal integration for radial diffraction analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
import glob
import pandas as pd
import numpy as np

from gui_base import GUIBase
from batch_appearance import ModernButton, CuteSheepProgressBar


class RadialXRDModule(GUIBase):
    """Radial XRD module for azimuthal integration"""

    def __init__(self, parent, root):
        """
        Initialize Radial XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root Tk window for dialogs
        """
        super().__init__()
        self.parent = parent
        self.root = root

        # Initialize variables
        self._init_variables()

        # Store custom sectors list
        self.custom_sectors = []

    def _init_variables(self):
        """Initialize all Tkinter variables"""
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
        self.radial_preset = tk.StringVar(value='quadrants')
        self.radial_mode = tk.StringVar(value='single')  # 'single' or 'multiple'
        self.radial_multiple_mode = tk.StringVar(value='custom')  # 'preset' or 'custom'
        self.radial_output_csv = tk.BooleanVar(value=True)  # Enable CSV output by default

    def setup_ui(self):
        """Setup the complete radial XRD UI"""
        main_frame = tk.Frame(self.parent, bg=self.colors['bg'])
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

        # Azimuthal angle reference diagram
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
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(unit_cont, textvariable=self.radial_unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # CSV Output option
        csv_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        csv_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Checkbutton(csv_cont, text="Output CSV", variable=self.radial_output_csv,
                      bg=self.colors['card_bg'], font=('Comic Sans MS', 9, 'bold'),
                      activebackground=self.colors['card_bg']).pack(anchor=tk.W, pady=(0, 2))
        tk.Label(csv_cont, text="(Save merged results)", bg=self.colors['card_bg'],
                fg=self.colors['text_light'], font=('Comic Sans MS', 7, 'italic')).pack(anchor=tk.W)

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

        # Mode selection
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

        tk.Radiobutton(mode_buttons, text="Multiple Sectors", variable=self.radial_mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_radial_mode).pack(side=tk.LEFT)

        # Container for dynamic content
        self.radial_dynamic_frame = tk.Frame(content2, bg=self.colors['card_bg'])
        self.radial_dynamic_frame.pack(fill=tk.BOTH, expand=True)

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
            self._setup_single_sector_ui()
        else:
            self._setup_multiple_sectors_ui()

    def _setup_single_sector_ui(self):
        """Setup UI for single sector mode"""
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

    def _setup_multiple_sectors_ui(self):
        """Setup UI for multiple sectors mode"""
        # Sub-mode selection
        submode_frame = tk.Frame(self.radial_dynamic_frame, bg=self.colors['card_bg'])
        submode_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Label(submode_frame, text="Multiple Sectors Mode:", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        submode_buttons = tk.Frame(submode_frame, bg=self.colors['card_bg'])
        submode_buttons.pack(anchor=tk.W)

        tk.Radiobutton(submode_buttons, text="Preset Templates", variable=self.radial_multiple_mode,
                      value='preset', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_multiple_submode).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(submode_buttons, text="Custom Sectors", variable=self.radial_multiple_mode,
                      value='custom', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_multiple_submode).pack(side=tk.LEFT)

        # Container for sub-mode content
        self.radial_submode_frame = tk.Frame(self.radial_dynamic_frame, bg=self.colors['card_bg'])
        self.radial_submode_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Initialize sub-mode
        self.update_multiple_submode()

    def update_multiple_submode(self):
        """Update UI based on multiple sectors sub-mode"""
        # Clear existing content
        for widget in self.radial_submode_frame.winfo_children():
            widget.destroy()

        if self.radial_multiple_mode.get() == 'preset':
            self._setup_preset_mode()
        else:
            self._setup_custom_sectors_mode()

    def _setup_preset_mode(self):
        """Setup preset templates mode"""
        preset_frame = tk.Frame(self.radial_submode_frame, bg=self.colors['card_bg'])
        preset_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(preset_frame, text="Select Preset:", bg=self.colors['card_bg'],
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

    def _setup_custom_sectors_mode(self):
        """Setup custom sectors mode with table for multiple entries"""
        # Instructions
        instruction_frame = tk.Frame(self.radial_submode_frame, bg='#FFF8DC',
                                     relief='solid', borderwidth=1, padx=10, pady=8)
        instruction_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(instruction_frame, text="💡 Define custom azimuthal sectors below. Add multiple rows to integrate different angular ranges.",
                bg='#FFF8DC', fg=self.colors['text_dark'],
                font=('Comic Sans MS', 8)).pack(anchor=tk.W)

        # Table frame with scrollbar
        table_container = tk.Frame(self.radial_submode_frame, bg=self.colors['card_bg'])
        table_container.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbar for table
        canvas = tk.Canvas(table_container, bg=self.colors['card_bg'], height=200, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=canvas.yview)

        self.sectors_table_frame = tk.Frame(canvas, bg=self.colors['card_bg'])

        self.sectors_table_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.sectors_table_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Header row
        header_frame = tk.Frame(self.sectors_table_frame, bg='#E8E4F3', relief='solid', borderwidth=1)
        header_frame.pack(fill=tk.X, padx=2, pady=2)

        tk.Label(header_frame, text="#", bg='#E8E4F3', width=3,
                font=('Comic Sans MS', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Label(header_frame, text="Start Angle (°)", bg='#E8E4F3', width=15,
                font=('Comic Sans MS', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Label(header_frame, text="End Angle (°)", bg='#E8E4F3', width=15,
                font=('Comic Sans MS', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Label(header_frame, text="Label", bg='#E8E4F3', width=20,
                font=('Comic Sans MS', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Label(header_frame, text="Action", bg='#E8E4F3', width=8,
                font=('Comic Sans MS', 8, 'bold')).pack(side=tk.LEFT, padx=2)

        # Initialize with default 4 sectors
        if not self.custom_sectors:
            self.custom_sectors = [
                [0.0, 90.0, "Sector_1"],
                [90.0, 180.0, "Sector_2"],
                [180.0, 270.0, "Sector_3"],
                [270.0, 360.0, "Sector_4"]
            ]

        self._refresh_sectors_table()

        # Add/Clear buttons
        btn_frame = tk.Frame(self.radial_submode_frame, bg=self.colors['card_bg'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(btn_frame, text="➕ Add Sector", command=self._add_sector,
                 bg=self.colors['success'], fg='white',
                 font=('Comic Sans MS', 8, 'bold'), relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(btn_frame, text="🗑️ Clear All", command=self._clear_all_sectors,
                 bg=self.colors['error'], fg='white',
                 font=('Comic Sans MS', 8, 'bold'), relief='flat',
                 padx=10, pady=5).pack(side=tk.LEFT)

    def _refresh_sectors_table(self):
        """Refresh the sectors table display"""
        # Clear all rows except header
        for widget in self.sectors_table_frame.winfo_children()[1:]:
            widget.destroy()

        # Add rows for each sector
        for idx, sector_data in enumerate(self.custom_sectors):
            row_frame = tk.Frame(self.sectors_table_frame, bg='white', relief='solid', borderwidth=1)
            row_frame.pack(fill=tk.X, padx=2, pady=1)

            # Row number
            tk.Label(row_frame, text=str(idx + 1), bg='white', width=3,
                    font=('Comic Sans MS', 8)).pack(side=tk.LEFT, padx=2)

            # Start angle entry
            start_var = tk.DoubleVar(value=sector_data[0])
            start_entry = ttk.Spinbox(row_frame, from_=0, to=360, textvariable=start_var,
                                     width=13, font=('Comic Sans MS', 8))
            start_entry.pack(side=tk.LEFT, padx=2)
            sector_data[0] = start_var

            # End angle entry
            end_var = tk.DoubleVar(value=sector_data[1])
            end_entry = ttk.Spinbox(row_frame, from_=0, to=360, textvariable=end_var,
                                   width=13, font=('Comic Sans MS', 8))
            end_entry.pack(side=tk.LEFT, padx=2)
            sector_data[1] = end_var

            # Label entry
            label_var = tk.StringVar(value=sector_data[2])
            label_entry = tk.Entry(row_frame, textvariable=label_var,
                                  font=('Comic Sans MS', 8), width=18)
            label_entry.pack(side=tk.LEFT, padx=2)
            sector_data[2] = label_var

            # Delete button
            del_btn = tk.Button(row_frame, text="✖", command=lambda i=idx: self._delete_sector(i),
                               bg='#FF6B6B', fg='white', font=('Comic Sans MS', 8, 'bold'),
                               relief='flat', width=6, padx=2)
            del_btn.pack(side=tk.LEFT, padx=2)

    def _add_sector(self):
        """Add a new sector row"""
        new_sector = [0.0, 90.0, f"Sector_{len(self.custom_sectors) + 1}"]
        self.custom_sectors.append(new_sector)
        self._refresh_sectors_table()

    def _delete_sector(self, index):
        """Delete a sector by index"""
        if len(self.custom_sectors) > 1:
            del self.custom_sectors[index]
            self._refresh_sectors_table()
        else:
            messagebox.showwarning("Warning", "At least one sector must be defined!")

    def _clear_all_sectors(self):
        """Clear all sectors and reset to default"""
        result = messagebox.askyesno("Confirm Clear", "Clear all sectors and reset to default?")
        if result:
            self.custom_sectors = [[0.0, 90.0, "Sector_1"]]
            self._refresh_sectors_table()

    def _get_sectors_list(self):
        """Get list of sectors from custom table"""
        sectors = []
        for sector_data in self.custom_sectors:
            start = sector_data[0].get() if hasattr(sector_data[0], 'get') else sector_data[0]
            end = sector_data[1].get() if hasattr(sector_data[1], 'get') else sector_data[1]
            label = sector_data[2].get() if hasattr(sector_data[2], 'get') else sector_data[2]
            sectors.append((start, end, label))
        return sectors

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
            self.show_success(self.root, "Azimuthal integration completed successfully!")

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

        azim_start = self.radial_azimuth_start.get()
        azim_end = self.radial_azimuth_end.get()
        sector_label = self.radial_sector_label.get()

        self.radial_log(f"📐 Azimuthal range: {azim_start}° to {azim_end}°")
        self.radial_log(f"🏷️  Sector label: {sector_label}")

        # Call integration with CSV output
        output_files = self._integrate_sector(azim_start, azim_end, sector_label)

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

        # Get sectors based on mode
        if self.radial_multiple_mode.get() == 'preset':
            preset_name = self.radial_preset.get()
            sector_list = self._get_preset_sectors(preset_name)
            self.radial_log(f"📐 Using preset: {preset_name}")
        else:
            sector_list = self._get_sectors_list()
            self.radial_log(f"📐 Using custom sectors")

        self.radial_log(f"📊 Number of sectors: {len(sector_list)}")

        for start, end, label in sector_list:
            self.radial_log(f"   - {label}: {start}° to {end}°")

        # Run integration for all sectors
        all_output_files = []
        for start, end, label in sector_list:
            self.radial_log(f"\n🔄 Processing {label}...")
            output_files = self._integrate_sector(start, end, label)
            all_output_files.extend(output_files)

        self.radial_log(f"\n{'='*60}")
        self.radial_log(f"✨ Integration complete!")
        self.radial_log(f"📊 Generated {len(all_output_files)} files total")
        self.radial_log(f"📁 Output directory: {self.radial_output_dir.get()}")
        self.radial_log(f"{'='*60}\n")

    def _integrate_sector(self, azim_start, azim_end, sector_label):
        """
        Perform integration for a single sector using pyFAI

        Args:
            azim_start: Start azimuth angle in degrees
            azim_end: End azimuth angle in degrees
            sector_label: Label for this sector

        Returns:
            List of output file paths
        """
        try:
            import pyFAI
            import h5py
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as pyFAI_AI
        except ImportError:
            raise ImportError("pyFAI is required for azimuthal integration. Install it with: pip install pyFAI")

        # Load calibration
        ai = pyFAI_AI()
        ai.load(self.radial_poni_path.get())

        # Load mask if provided
        mask = None
        if self.radial_mask_path.get():
            mask_path = self.radial_mask_path.get()
            if mask_path.endswith('.edf'):
                import fabio
                mask = fabio.open(mask_path).data
            elif mask_path.endswith('.npy'):
                mask = np.load(mask_path)

        # Get input files
        input_files = sorted(glob.glob(self.radial_input_pattern.get()))
        if not input_files:
            raise ValueError(f"No files found matching pattern: {self.radial_input_pattern.get()}")

        self.radial_log(f"   Found {len(input_files)} input files")

        # Prepare output directory
        output_dir = self.radial_output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        # Storage for CSV output
        csv_data = {}
        output_files = []

        # Process each file
        for idx, h5_file in enumerate(input_files):
            filename = os.path.basename(h5_file)
            self.radial_log(f"   [{idx+1}/{len(input_files)}] {filename}")

            # Read data from H5
            with h5py.File(h5_file, 'r') as f:
                data = f[self.radial_dataset_path.get()][()]

            # Perform azimuthal integration with sector
            result = ai.integrate1d(
                data,
                npt=self.radial_npt.get(),
                unit=self.radial_unit.get(),
                mask=mask,
                azimuth_range=(azim_start, azim_end)
            )

            # Save individual .xy file
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{sector_label}.xy"
            output_path = os.path.join(output_dir, output_filename)

            np.savetxt(output_path, np.column_stack([result.radial, result.intensity]),
                      header=f"{self.radial_unit.get()}  Intensity", comments='#')
            output_files.append(output_path)

            # Store for CSV
            csv_data[filename] = {
                'x': result.radial,
                'y': result.intensity
            }

        # Generate CSV if enabled
        if self.radial_output_csv.get():
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            self._save_csv(csv_data, csv_path, sector_label)
            output_files.append(csv_path)
            self.radial_log(f"   💾 CSV saved: {csv_filename}")

        return output_files

    def _save_csv(self, csv_data, csv_path, sector_label):
        """
        Save integrated data to CSV format

        Args:
            csv_data: Dictionary with filename -> {x, y} data
            csv_path: Output CSV file path
            sector_label: Label for this sector
        """
        if not csv_data:
            return

        # Get reference x-axis (assuming all have same x)
        first_key = list(csv_data.keys())[0]
        x_values = csv_data[first_key]['x']

        # Build dataframe
        df_dict = {self.radial_unit.get(): x_values}

        for filename, data in csv_data.items():
            # Use filename as column header
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
            ],
            'horizontal_vertical': [
                (0, 90, "Right"),
                (90, 180, "Top"),
                (180, 270, "Left"),
                (270, 360, "Bottom")
            ]
        }
        return presets.get(preset_name, [])


def main():
    """Main function to run the Radial XRD Module standalone"""
    # Create root window
    root = tk.Tk()
    root.title("Radial XRD - Azimuthal Integration")
    root.geometry("900x900")

    # Set theme colors
    root.configure(bg='#F5F3FF')

    # Create main container frame
    main_container = tk.Frame(root, bg='#F5F3FF')
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Initialize the Radial XRD module
    try:
        radial_module = RadialXRDModule(main_container, root)
        radial_module.setup_ui()

        # Log startup message
        radial_module.radial_log("="*60)
        radial_module.radial_log("🎯 Radial XRD Module - Azimuthal Integration")
        radial_module.radial_log("="*60)
        radial_module.radial_log("✨ Module loaded successfully!")
        radial_module.radial_log("📖 Configure settings and click 'Run Azimuthal Integration' to start")
        radial_module.radial_log("")

    except Exception as e:
        messagebox.showerror("Initialization Error",
                           f"Failed to initialize Radial XRD Module:\n{str(e)}\n\n"
                           f"Make sure gui_base.py and batch_appearance.py are available.")
        root.destroy()
        return

    # Start the main event loop
    root.mainloop()


if __name__ == "__main__":
    main()
