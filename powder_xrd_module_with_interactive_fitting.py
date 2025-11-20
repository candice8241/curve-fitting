# -*- coding: utf-8 -*-
"""
Powder XRD Module - OPTIMIZED VERSION WITH INTERACTIVE PEAK FITTING
Contains integration, peak fitting, phase analysis, and Birch-Murnaghan fitting
Optimized for smooth module switching without lag
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from batch_integration import BatchIntegrator
#from peak_fitting import BatchFitter
from half_auto_fitting import DataProcessor
from batch_cal_volume import XRayDiffractionAnalyzer as XRDAnalyzer
from birch_murnaghan_batch import BirchMurnaghanFitter
from theme_module import GUIBase, CuteSheepProgressBar, ModernTab, ModernButton

# Import the enhanced peak fitting GUI
from peak_fitting_gui_enhanced import PeakFittingGUI

# Import icon utility
try:
    from icon_utils import set_window_icon
except ImportError:
    # Fallback if icon_utils not available
    def set_window_icon(window, icon_path=None, use_default=True):
        try:
            if icon_path and os.path.exists(icon_path):
                if icon_path.endswith('.ico'):
                    window.iconbitmap(icon_path)
                else:
                    icon_image = tk.PhotoImage(file=icon_path)
                    window.iconphoto(True, icon_image)
                    window._icon_image = icon_image
        except:
            pass


class SpinboxStyleButton(tk.Frame):
    """Spinbox-style button widget matching the reference image"""

    def __init__(self, parent, text, command, width=80, **kwargs):
        super().__init__(parent, bg='#E8D5F0', **kwargs)

        self.command = command
        self.text = text

        # Configure frame with rounded appearance
        self.configure(relief='solid', borderwidth=1, highlightbackground='#C8B5D0',
                      highlightthickness=1)

        # Create button
        self.button = tk.Button(
            self,
            text=text,
            command=command,
            bg='#E8D5F0',
            fg='#6B4C7A',
            font=('Comic Sans MS', 9),
            relief='flat',
            borderwidth=0,
            activebackground='#D5C0E0',
            activeforeground='#6B4C7A',
            cursor='hand2',
            padx=15,
            pady=5
        )
        self.button.pack(fill=tk.BOTH, expand=True)

        # Hover effects
        self.button.bind('<Enter>', self._on_enter)
        self.button.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        self.button.config(bg='#D5C0E0')
        self.configure(bg='#D5C0E0')

    def _on_leave(self, event):
        self.button.config(bg='#E8D5F0')
        self.configure(bg='#E8D5F0')


class CustomSpinbox(tk.Frame):
    """Custom spinbox with left/right arrow buttons for value adjustment"""

    def __init__(self, parent, from_=0, to=100, textvariable=None, increment=1,
                 width=80, is_float=False, **kwargs):
        super().__init__(parent, bg='#F0E6FA', **kwargs)

        self.from_ = from_
        self.to = to
        self.increment = increment
        self.is_float = is_float
        self.textvariable = textvariable

        # Left decrease button
        self.left_btn = tk.Button(
            self,
            text="<",
            command=self.decrease,
            bg='#E8D5F0',
            fg='#6B4C7A',
            font=('Arial', 10, 'bold'),
            relief='flat',
            borderwidth=1,
            activebackground='#D5C0E0',
            cursor='hand2',
            width=2,
            padx=2,
            pady=2
        )
        self.left_btn.pack(side=tk.LEFT, padx=2)

        # Value display
        self.value_frame = tk.Frame(self, bg='white', relief='solid', borderwidth=1)
        self.value_frame.pack(side=tk.LEFT, padx=2)

        self.entry = tk.Entry(
            self.value_frame,
            textvariable=textvariable,
            font=('Arial', 10),
            bg='white',
            fg='#333333',
            justify='center',
            relief='flat',
            borderwidth=0,
            width=8
        )
        self.entry.pack(padx=3, pady=2)

        # Right increase button
        self.right_btn = tk.Button(
            self,
            text=">",
            command=self.increase,
            bg='#E8D5F0',
            fg='#6B4C7A',
            font=('Arial', 10, 'bold'),
            relief='flat',
            borderwidth=1,
            activebackground='#D5C0E0',
            cursor='hand2',
            width=2,
            padx=2,
            pady=2
        )
        self.right_btn.pack(side=tk.LEFT, padx=2)

        # Bind hover effects
        self.left_btn.bind('<Enter>', lambda e: self.left_btn.config(bg='#D5C0E0'))
        self.left_btn.bind('<Leave>', lambda e: self.left_btn.config(bg='#E8D5F0'))
        self.right_btn.bind('<Enter>', lambda e: self.right_btn.config(bg='#D5C0E0'))
        self.right_btn.bind('<Leave>', lambda e: self.right_btn.config(bg='#E8D5F0'))

        # Bind entry validation
        self.entry.bind('<Return>', self.validate_entry)
        self.entry.bind('<FocusOut>', self.validate_entry)

    def get_current_value(self):
        """Get current value from textvariable"""
        if self.textvariable:
            try:
                if self.is_float:
                    return float(self.textvariable.get())
                else:
                    return int(self.textvariable.get())
            except (ValueError, tk.TclError):
                return self.from_
        return self.from_

    def set_value(self, value):
        """Set value to textvariable"""
        # Clamp value to bounds
        value = max(self.from_, min(self.to, value))

        if self.textvariable:
            if self.is_float:
                self.textvariable.set(float(value))
            else:
                self.textvariable.set(int(value))

    def increase(self):
        """Increase value"""
        current = self.get_current_value()
        new_value = current + self.increment
        self.set_value(new_value)

    def decrease(self):
        """Decrease value"""
        current = self.get_current_value()
        new_value = current - self.increment
        self.set_value(new_value)

    def validate_entry(self, event=None):
        """Validate manual entry"""
        try:
            current = self.get_current_value()
            self.set_value(current)  # This will clamp to bounds
        except:
            self.set_value(self.from_)


class PowderXRDModule(GUIBase):
    """Powder XRD processing module with integration and analysis capabilities - OPTIMIZED"""

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

        # Pre-create module frames (OPTIMIZATION: create once, switch visibility)
        self.integration_frame = None
        self.analysis_frame = None

        # Track interactive fitting window
        self.interactive_fitting_window = None

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
        # 先不要 pack main_frame，等所有组件构建完再 pack

        # Module selector buttons with improved styling
        module_frame = tk.Frame(main_frame, bg=self.colors['bg'], height=60)
        module_frame.pack(fill=tk.X, pady=(5, 15))

        btn_container = tk.Frame(module_frame, bg=self.colors['bg'])

        # Create module buttons with distinct colors
        self.integration_module_btn = tk.Button(
            btn_container,
            text="1D Integration & Peak Fitting",
            font=('Comic Sans MS', 10),
            bg='#C8A2D9',  # Light purple when active
            fg='#4A2C5F',
            activebackground='#B794F6',
            relief='solid',
            borderwidth=2,
            padx=20,
            pady=12,
            cursor='hand2',
            command=lambda: self.show_module("integration")
        )
        self.integration_module_btn.pack(side=tk.LEFT, padx=8)

        self.analysis_module_btn = tk.Button(
            btn_container,
            text="Cal_Volume & BM_Fitting",
            font=('Comic Sans MS', 10),
            bg='#E8D5F0',  # Light pink when inactive
            fg='#4A2C5F',
            activebackground='#FFB6D9',
            relief='solid',
            borderwidth=2,
            padx=20,
            pady=12,
            cursor='hand2',
            command=lambda: self.show_module("analysis")
        )
        self.analysis_module_btn.pack(side=tk.LEFT, padx=8)

        btn_container.pack()

        # Container for dynamic content
        self.dynamic_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True)

        # OPTIMIZATION: Pre-create both module frames
        self._create_module_frames()

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

        # 所有组件构建完成后，再显示主框架（避免中间过程的闪烁）
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 强制立即完成所有布局计算
        self.root.update_idletasks()

    def _create_module_frames(self):
        """OPTIMIZATION: Pre-create both module frames to avoid recreation lag"""
        # Create integration module frame
        self.integration_frame = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        self.setup_integration_module(self.integration_frame)

        # Create analysis module frame
        self.analysis_frame = tk.Frame(self.dynamic_frame, bg=self.colors['bg'])
        self.setup_analysis_module(self.analysis_frame)

        # Both frames are created but not packed yet

    def show_module(self, module_type):
        """
        OPTIMIZED: Switch between integration and analysis modules
        Uses pack/pack_forget instead of destroy/recreate for smooth transitions
        """
        self.current_module = module_type

        # Update button styles immediately (visual feedback)
        if module_type == "integration":
            self.integration_module_btn.config(bg='#C8A2D9', fg='#4A2C5F', relief='sunken')
            self.analysis_module_btn.config(bg='#E8D5F0', fg='#6B4C7A', relief='solid')
        else:
            self.integration_module_btn.config(bg='#E8D5F0', fg='#6B4C7A', relief='solid')
            self.analysis_module_btn.config(bg='#FFB6D9', fg='#4A2C5F', relief='sunken')

        # Force immediate UI update for button state
        self.root.update_idletasks()

        # OPTIMIZATION: Use pack_forget/pack instead of destroy/recreate
        # This is MUCH faster and smoother
        if module_type == "integration":
            # Hide analysis frame
            if self.analysis_frame.winfo_ismapped():
                self.analysis_frame.pack_forget()
            # Show integration frame
            if not self.integration_frame.winfo_ismapped():
                self.integration_frame.pack(fill=tk.BOTH, expand=True)
        else:
            # Hide integration frame
            if self.integration_frame.winfo_ismapped():
                self.integration_frame.pack_forget()
            # Show analysis frame
            if not self.analysis_frame.winfo_ismapped():
                self.analysis_frame.pack(fill=tk.BOTH, expand=True)

        # Final UI update
        self.root.update_idletasks()

    def create_file_picker_with_spinbox_btn(self, parent, label_text, var, filetypes, pattern=False):
        """Create file picker with spinbox-style button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(container, text=label_text, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        tk.Entry(input_frame, textvariable=var, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if pattern:
            btn = SpinboxStyleButton(input_frame, "Browse",
                                    lambda: self.browse_file(var, filetypes, pattern=True),
                                    width=75)
        else:
            btn = SpinboxStyleButton(input_frame, "Browse",
                                    lambda: self.browse_file(var, filetypes),
                                    width=75)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_folder_picker_with_spinbox_btn(self, parent, label_text, var):
        """Create folder picker with spinbox-style button"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(container, text=label_text, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        tk.Entry(input_frame, textvariable=var, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        btn = SpinboxStyleButton(input_frame, "Browse",
                                lambda: self.browse_folder(var),
                                width=75)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def setup_integration_module(self, parent_frame):
        """
        Setup integration and peak fitting module UI
        MODIFIED: Now accepts parent_frame parameter for pre-creation
        """
        # Integration Settings Card
        integration_card = self.create_card_frame(parent_frame)
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

        # Use new spinbox-style buttons for all file/folder pickers
        self.create_file_picker_with_spinbox_btn(content1, "PONI File", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(content1, "Mask File", self.mask_path,
                               [("EDF files", "*.edf"), ("All files", "*.*")])
        self.create_file_picker_with_spinbox_btn(content1, "Input Pattern", self.input_pattern,
                               [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)
        self.create_folder_picker_with_spinbox_btn(content1, "Output Directory", self.output_dir)

        # Dataset Path - WITH BROWSE BUTTON
        dataset_container = tk.Frame(content1, bg=self.colors['card_bg'])
        dataset_container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(dataset_container, text="Dataset Path", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        dataset_input_frame = tk.Frame(dataset_container, bg=self.colors['card_bg'])
        dataset_input_frame.pack(fill=tk.X)

        tk.Entry(dataset_input_frame, textvariable=self.dataset_path, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        # Browse button for dataset path
        dataset_browse_btn = SpinboxStyleButton(
            dataset_input_frame,
            "Browse",
            lambda: self.browse_dataset_path(),
            width=75
        )
        dataset_browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Parameters with CUSTOM SPINBOXES
        param_frame = tk.Frame(content1, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(10, 0))

        # Number of Points - with custom spinbox
        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npt_cont, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        CustomSpinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                     increment=100, is_float=False).pack(anchor=tk.W)

        # Unit - keep as combobox
        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(unit_cont, text="Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Combobox(unit_cont, textvariable=self.unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Fitting Settings Card
        fitting_card = self.create_card_frame(parent_frame)
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
        btn_frame = tk.Frame(parent_frame, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        btn_cont = tk.Frame(btn_frame, bg=self.colors['bg'])
        btn_cont.pack(expand=True)

        btns = tk.Frame(btn_cont, bg=self.colors['bg'])
        btns.pack()

        SpinboxStyleButton(btns, "🐿️ Run Integration", self.run_integration,
                          width=180).pack(side=tk.LEFT, padx=6)

        SpinboxStyleButton(btns, "🐻 Run Fitting", self.run_fitting,
                          width=180).pack(side=tk.LEFT, padx=6)

        SpinboxStyleButton(btns, "🦔 Full Pipeline", self.run_full_pipeline,
                          width=180).pack(side=tk.LEFT, padx=6)

        # NEW: Interactive Peak Fitting Button
        SpinboxStyleButton(btns, "✨ Interactive Fitting", self.open_interactive_fitting,
                          width=180).pack(side=tk.LEFT, padx=6)

    def browse_dataset_path(self):
        """Browse for dataset path (allows manual text entry or file selection)"""
        result = messagebox.askquestion(
            "Dataset Path",
            "Dataset path is typically an HDF5 internal path like 'entry/data/data'.\n\n" +
            "Do you want to manually enter the path?\n\n" +
            "Click 'No' to keep the current value.",
            icon='question'
        )

        if result == 'yes':
            # Simple dialog for text input
            dialog = tk.Toplevel(self.root)
            dialog.title("Enter Dataset Path")
            dialog.geometry("400x150")
            dialog.configure(bg='#F0E6FA')
            dialog.transient(self.root)
            dialog.grab_set()

            tk.Label(dialog, text="Enter HDF5 Dataset Path:",
                    bg='#F0E6FA', font=('Comic Sans MS', 10)).pack(pady=10)

            entry = tk.Entry(dialog, width=40, font=('Comic Sans MS', 10))
            entry.insert(0, self.dataset_path.get())
            entry.pack(pady=5)
            entry.focus()

            def confirm():
                self.dataset_path.set(entry.get())
                dialog.destroy()

            SpinboxStyleButton(dialog, "Confirm", confirm, width=100).pack(pady=10)

            dialog.bind('<Return>', lambda e: confirm())

    def open_interactive_fitting(self):
        """
        Open the interactive peak fitting GUI in a new window
        """
        # Check if window already exists and is open
        if self.interactive_fitting_window is not None:
            try:
                # Check if window still exists
                if self.interactive_fitting_window.winfo_exists():
                    # Bring window to front
                    self.interactive_fitting_window.lift()
                    self.interactive_fitting_window.focus_force()
                    self.log("📊 Interactive fitting window brought to front")
                    return
            except:
                # Window was closed, create new one
                pass

        # Create new toplevel window
        self.interactive_fitting_window = tk.Toplevel(self.root)
        self.interactive_fitting_window.title("Interactive Peak Fitting - Enhanced")

        # Set window size and position
        window_width = 1400
        window_height = 850
        screen_width = self.interactive_fitting_window.winfo_screenwidth()
        screen_height = self.interactive_fitting_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.interactive_fitting_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Set window icon using utility function
        # Tries multiple icon locations and formats
        set_window_icon(self.interactive_fitting_window, use_default=True)

        # Create the peak fitting GUI inside this window
        fitting_app = PeakFittingGUI(self.interactive_fitting_window, icon_path=None)
        fitting_app.setup_ui()

        # Log the action
        self.log("✨ Interactive peak fitting GUI opened in new window")

        # Handle window close event
        def on_closing():
            if messagebox.askokcancel("Close Interactive Fitting",
                                     "Are you sure you want to close the interactive fitting window?"):
                self.interactive_fitting_window.destroy()
                self.interactive_fitting_window = None
                self.log("📊 Interactive fitting window closed")

        self.interactive_fitting_window.protocol("WM_DELETE_WINDOW", on_closing)

    def setup_analysis_module(self, parent_frame):
        """
        Setup phase analysis and Birch-Murnaghan fitting module UI
        MODIFIED: Now accepts parent_frame parameter for pre-creation
        """
        # Phase Analysis Section
        phase_card = self.create_card_frame(parent_frame)
        phase_card.pack(fill=tk.X, pady=(0, 15))

        content3 = tk.Frame(phase_card, bg=self.colors['card_bg'], padx=20, pady=15)
        content3.pack(fill=tk.BOTH, expand=True)

        header3 = tk.Frame(content3, bg=self.colors['card_bg'])
        header3.pack(anchor=tk.W, pady=(0, 15))

        tk.Label(header3, text="🐶", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header3, text="Phase Transition Analysis & Volume Calculation",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # Main content area - 2 columns layout
        main_content = tk.Frame(content3, bg=self.colors['card_bg'])
        main_content.pack(fill=tk.BOTH, expand=True)

        # Left column - Input files
        left_col = tk.Frame(main_content, bg=self.colors['card_bg'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # Peak CSV input
        tk.Label(left_col, text="Input CSV (Peak Data)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 3))

        peak_input_frame = tk.Frame(left_col, bg=self.colors['card_bg'])
        peak_input_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Entry(peak_input_frame, textvariable=self.phase_peak_csv, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        SpinboxStyleButton(peak_input_frame, "Browse",
                          lambda: self.browse_file(self.phase_peak_csv, [("CSV files", "*.csv")]),
                          width=75).pack(side=tk.LEFT, padx=(5, 0))

        # Separate peaks button
        SpinboxStyleButton(left_col, "🐶 Separate Original & New Peaks",
                          self.separate_peaks,
                          width=280).pack(pady=(0, 15))

        # Volume CSV input
        tk.Label(left_col, text="Input CSV (Volume Calculation)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 3))

        volume_input_frame = tk.Frame(left_col, bg=self.colors['card_bg'])
        volume_input_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Entry(volume_input_frame, textvariable=self.phase_volume_csv, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        SpinboxStyleButton(volume_input_frame, "Browse",
                          lambda: self.browse_file(self.phase_volume_csv, [("CSV files", "*.csv")]),
                          width=75).pack(side=tk.LEFT, padx=(5, 0))

        # Crystal system selection
        system_frame = tk.Frame(left_col, bg=self.colors['card_bg'])
        system_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(system_frame, text="Crystal System", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Combobox(system_frame, textvariable=self.phase_volume_system,
                    values=['FCC', 'BCC', 'SC', 'Hexagonal', 'Tetragonal',
                           'Orthorhombic', 'Monoclinic', 'Triclinic'],
                    width=15, state='readonly', font=('Comic Sans MS', 9)).pack(side=tk.LEFT)

        # Output directory
        tk.Label(left_col, text="Output Directory", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 3))

        output_frame = tk.Frame(left_col, bg=self.colors['card_bg'])
        output_frame.pack(fill=tk.X)

        tk.Entry(output_frame, textvariable=self.phase_volume_output, font=('Comic Sans MS', 9),
                bg='white', relief='solid', borderwidth=1).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        SpinboxStyleButton(output_frame, "Browse",
                          lambda: self.browse_folder(self.phase_volume_output),
                          width=75).pack(side=tk.LEFT, padx=(5, 0))

        # Calculate Volume button
        SpinboxStyleButton(left_col, "🦊 Calculate Volume & Fit Lattice Parameters",
                          self.run_phase_analysis,
                          width=300).pack(pady=(15, 0))

        # Right column - Parameters
        right_col = tk.Frame(main_content, bg='#F0E6FA', relief='solid', borderwidth=2, padx=20, pady=20)
        right_col.pack(side=tk.LEFT, fill=tk.Y)

        # Header with emoji
        param_header = tk.Frame(right_col, bg='#F0E6FA')
        param_header.pack(pady=(0, 15))

        tk.Label(param_header, text="🎀", bg='#F0E6FA',
                font=('Segoe UI Emoji', 12)).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(param_header, text="Analysis Parameters", bg='#F0E6FA',
                fg='#9966CC', font=('Comic Sans MS', 10, 'bold')).pack(side=tk.LEFT)

        # Wavelength section
        wl_container = tk.Frame(right_col, bg='#F0E6FA')
        wl_container.pack(fill=tk.X, pady=(0, 8))

        wl_label_frame = tk.Frame(wl_container, bg='#F0E6FA')
        wl_label_frame.pack(pady=(0, 3))

        tk.Label(wl_label_frame, text="🌸", bg='#F0E6FA',
                font=('Segoe UI Emoji', 10)).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(wl_label_frame, text="Wavelength (Å)", bg='#F0E6FA',
                fg='#4A4A4A', font=('Comic Sans MS', 9,'bold')).pack(side=tk.LEFT)

        wl_entry = tk.Entry(wl_container, textvariable=self.phase_wavelength,
                           font=('Arial', 10), width=12, justify='center',
                           bg='white', relief='solid', borderwidth=1)
        wl_entry.pack()

        # Separator
        tk.Frame(right_col, bg='#FFC1CC', height=2).pack(fill=tk.X, pady=12)

        # Peak Tolerances section
        tol_header = tk.Frame(right_col, bg='#F0E6FA')
        tol_header.pack(pady=(0, 10))

        tk.Label(tol_header, text="🎀", bg='#F0E6FA',
                font=('Segoe UI Emoji', 10)).pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(tol_header, text="Peak Tolerances", bg='#F0E6FA',
                fg='#4A4A4A', font=('Comic Sans MS', 9, 'bold')).pack(side=tk.LEFT)

        # Tolerance 1
        tol1_row = tk.Frame(right_col, bg='#F0E6FA')
        tol1_row.pack(fill=tk.X, pady=3)
        tk.Label(tol1_row, text="Tolerance 1:", bg='#F0E6FA',
                font=('Comic Sans MS', 8), anchor='w').pack(side=tk.LEFT)
        tk.Entry(tol1_row, textvariable=self.phase_tolerance_1,
                font=('Arial', 9), width=12, justify='center',
                bg='white', relief='solid', borderwidth=1).pack(side=tk.RIGHT, padx=(0, 0))

        # Tolerance 2
        tol2_row = tk.Frame(right_col, bg='#F0E6FA')
        tol2_row.pack(fill=tk.X, pady=3)
        tk.Label(tol2_row, text="Tolerance 2:", bg='#F0E6FA',
                font=('Comic Sans MS', 8), anchor='w').pack(side=tk.LEFT)
        tk.Entry(tol2_row, textvariable=self.phase_tolerance_2,
                font=('Arial', 9), width=12, justify='center',
                bg='white', relief='solid', borderwidth=1).pack(side=tk.RIGHT, padx=(0, 0))

        # Tolerance 3
        tol3_row = tk.Frame(right_col, bg='#F0E6FA')
        tol3_row.pack(fill=tk.X, pady=3)
        tk.Label(tol3_row, text="Tolerance 3:", bg='#F0E6FA',
                font=('Comic Sans MS', 8), anchor='w').pack(side=tk.LEFT)
        tk.Entry(tol3_row, textvariable=self.phase_tolerance_3,
                font=('Arial', 9), width=12, justify='center',
                bg='white', relief='solid', borderwidth=1).pack(side=tk.RIGHT, padx=(0, 0))

        # Separator
        tk.Frame(right_col, bg='#FFC1CC', height=2).pack(fill=tk.X, pady=12)

        # N Pressure Points
        n_row = tk.Frame(right_col, bg='#F0E6FA')
        n_row.pack(fill=tk.X)

        tk.Label(n_row, text="N Pressure Points:", bg='#F0E6FA',
                font=('Comic Sans MS', 8), anchor='w').pack(side=tk.LEFT)

        ttk.Spinbox(n_row, from_=1, to=20, textvariable=self.phase_n_points,
                   width=8, font=('Arial', 9)).pack(side=tk.RIGHT, padx=(10, 0))

        # Birch-Murnaghan Section
        bm_card = self.create_card_frame(parent_frame)
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

        # Use new spinbox-style buttons
        self.create_file_picker_with_spinbox_btn(content4, "Input CSV (P-V Data)",
                               self.bm_input_file, [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_folder_picker_with_spinbox_btn(content4, "Output Directory", self.bm_output_dir)

        order_cont = tk.Frame(content4, bg=self.colors['card_bg'])
        order_cont.pack(fill=tk.X, pady=(5, 0))
        tk.Label(order_cont, text="BM Order", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(order_cont, textvariable=self.bm_order,
                    values=['2', '3'], width=18, state='readonly',
                    font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # BM button
        btn_frame3 = tk.Frame(parent_frame, bg=self.colors['bg'])
        btn_frame3.pack(fill=tk.X, pady=(10, 0))

        btn_cont3 = tk.Frame(btn_frame3, bg=self.colors['bg'])
        btn_cont3.pack(expand=True)

        SpinboxStyleButton(btn_cont3, "⚗️ Birch-Murnaghan Fit",
                          self.run_birch_murnaghan,
                          width=250).pack()

    # ==================== Processing Functions ====================
    # (All processing functions remain the same)

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

            analyzer = XRDAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get()
            )

            self.log(f"📄 Reading data from: {os.path.basename(csv_path)}")
            pressure_data = analyzer.read_pressure_peak_data(csv_path)
            self.log(f"✓ Loaded {len(pressure_data)} pressure points")

            self.log("🔍 Identifying phase transition...")
            transition_pressure, before_pressures, after_pressures = analyzer.find_phase_transition_point()

            if transition_pressure is None:
                self.log("⚠️ No phase transition detected")
                messagebox.showwarning("Warning", "No phase transition detected in the data")
                return

            self.log(f"✓ Phase transition detected at {transition_pressure:.2f} GPa")

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

            base_filename = csv_path.replace('.csv', '')
            new_peaks_dataset_csv = f"{base_filename}_new_peaks_dataset.csv"
            original_peaks_dataset_csv = f"{base_filename}_original_peaks_dataset.csv"

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

            self.log("📊 Building original peaks dataset...")
            original_peak_dataset = analyzer.build_original_peak_dataset(
                pressure_data,
                tracked_new_peaks,
                analyzer.peak_tolerance_3,
                output_csv=original_peaks_dataset_csv
            )

            self.log(f"✓ Original peaks dataset constructed for {len(original_peak_dataset)} pressure points")
            self.log(f"💾 Original peaks dataset saved to: {os.path.basename(original_peaks_dataset_csv)}")

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
            self.log("🐶 Starting Volume Calculation & Lattice Parameter Fitting")

            csv_path = self.phase_volume_csv.get()
            output_dir = self.phase_volume_output.get()

            os.makedirs(output_dir, exist_ok=True)

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

            input_dir = os.path.dirname(csv_path)
            base_filename = os.path.splitext(os.path.basename(csv_path))[0]

            generated_files = []

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

            os.makedirs(output_directory, exist_ok=True)

            self.log(f"📄 Reading data from: {os.path.basename(input_file_path)}")
            df = pd.read_csv(input_file_path)

            if 'V_atomic' not in df.columns or 'Pressure (GPa)' not in df.columns:
                raise ValueError("CSV must contain 'V_atomic' and 'Pressure (GPa)' columns")

            V_data = df['V_atomic'].dropna().values
            P_data = df['Pressure (GPa)'].dropna().values

            min_len = min(len(V_data), len(P_data))
            V_data = V_data[:min_len]
            P_data = P_data[:min_len]

            self.log(f"✓ Loaded {len(V_data)} data points")
            self.log(f"   Volume range: {V_data.min():.4f} - {V_data.max():.4f} Å³/atom")
            self.log(f"   Pressure range: {P_data.min():.2f} - {P_data.max():.2f} GPa")

            fitter = BirchMurnaghanFitter(
                V0_bounds=(0.8, 1.3),
                B0_bounds=(50, 500),
                B0_prime_bounds=(2.5, 6.5),
                max_iterations=10000
            )

            self.log(f"\n🔧 Fitting {order_str} Birch-Murnaghan equation...")

            results = fitter.fit_single_phase(V_data, P_data, phase_name="Single Phase")

            if order == 2:
                if results['2nd_order'] is None:
                    raise ValueError("2nd order fitting failed")
                fit_results = results['2nd_order']
            else:
                if results['3rd_order'] is None:
                    raise ValueError("3rd order fitting failed")
                fit_results = results['3rd_order']

            self.log(f"\n{'='*60}")
            self.log(f"✅ {order_str} BM Fitting Results:")
            self.log(f"{'='*60}")
            self.log(f"📊 V₀ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom")
            self.log(f"📊 B₀ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa")
            self.log(f"📊 B₀' = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}")
            self.log(f"📈 R² = {fit_results['R_squared']:.6f}")
            self.log(f"📉 RMSE = {fit_results['RMSE']:.4f} GPa")
            self.log(f"{'='*60}\n")

            self.log("📈 Generating plots...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.scatter(V_data, P_data, s=80, c='blue', marker='o',
                       label='Experimental Data', alpha=0.7, edgecolors='black', linewidths=1.5)

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

            textstr = f"$V_0$ = {fit_results['V0']:.4f} ± {fit_results['V0_err']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {fit_results['B0']:.2f} ± {fit_results['B0_err']:.2f} GPa\n"
            textstr += f"$B_0'$ = {fit_results['B0_prime']:.3f} ± {fit_results['B0_prime_err']:.3f}\n"
            textstr += f"$R^2$ = {fit_results['R_squared']:.6f}\n"
            textstr += f"RMSE = {fit_results['RMSE']:.4f} GPa"

            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.8, edgecolor='black'))

            residuals = P_data - fit_results['fitted_P']
            ax2.scatter(V_data, residuals, s=60, c='blue', marker='o',
                       alpha=0.7, edgecolors='black', linewidths=1.5)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
            ax2.set_xlabel('Volume V (Å³/atom)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals (GPa)', fontsize=12, fontweight='bold')
            ax2.set_title('Fitting Residuals Analysis', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=10)

            rmse_text = f"RMSE = {fit_results['RMSE']:.4f} GPa"
            ax2.text(0.05, 0.95, rmse_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightblue', alpha=0.8, edgecolor='black'))

            plt.tight_layout()

            fig_path = os.path.join(output_directory, f'BM_{order}rd_order_single_phase_fit.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"💾 Plot saved: {os.path.basename(fig_path)}")

            self.log(f"\n{'='*60}")
            self.log("✨ All tasks completed successfully!")
            self.log(f"{'='*60}")
            self.log(f"📁 Output directory: {output_directory}")
            self.log(f"   - {os.path.basename(fig_path)} : P-V curve and residuals")
            self.log(f"{'='*60}\n")

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
