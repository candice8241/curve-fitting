# -*- coding: utf-8 -*-
"""
Radial XRD Module (Azimuthal Integration)
Contains azimuthal integration for radial diffraction analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os

from gui_base import GUIBase
from batch_appearance import ModernButton, CuteSheepProgressBar
from batch_azimuthal_integration import AzimuthalIntegrator, get_preset_sectors


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
        self.radial_preset = tk.StringVar(value='custom')
        self.radial_mode = tk.StringVar(value='single')  # 'single' or 'multiple'

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

        tk.Radiobutton(mode_buttons, text="Multiple Sectors (Preset)", variable=self.radial_mode,
                      value='multiple', bg=self.colors['card_bg'],
                      font=('Comic Sans MS', 9),
                      command=self.update_radial_mode).pack(side=tk.LEFT)

        # Container for dynamic content
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
