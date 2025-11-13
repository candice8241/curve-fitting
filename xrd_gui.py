# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:11:31 2025

@author: 16961
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from batch_integration import BatchIntegrator
from peak_fitting import BatchFitter
from birch_murnaghan_batch import BirchMurnaghanFitter
from batch_cal_volume import XRayDiffractionAnalyzer
import math
import random


class ModernButton(tk.Canvas):
    """Custom button with hover effects and hand cursor"""
    def __init__(self, parent, text, command, icon="", bg_color="#9D4EDD",
                 hover_color="#C77DFF", text_color="white", width=200, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        self.icon = icon

        self.rect = self.create_rounded_rectangle(0, 0, width, height, radius=10,
                                                   fill=bg_color, outline="")

        display_text = f"{icon}  {text}" if icon else text
        self.text_id = self.create_text(width//2, height//2, text=display_text,
                                       fill=text_color, font=('Comic Sans MS', 11, 'bold'))

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.config(cursor="hand2")

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, e):
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, e):
        self.itemconfig(self.rect, fill=self.bg_color)

    def on_click(self, e):
        if self.command:
            self.command()


class ModernTab(tk.Frame):
    """Custom tab button"""
    def __init__(self, parent, text, command, is_active=False, **kwargs):
        super().__init__(parent, bg=parent['bg'], **kwargs)
        self.command = command
        self.is_active = is_active
        self.parent_widget = parent

        self.active_color = "#9D4EDD"
        self.inactive_color = "#8B8BA7"
        self.hover_color = "#C77DFF"

        self.label = tk.Label(self, text=text,
                             fg=self.active_color if is_active else self.inactive_color,
                             bg=parent['bg'], font=('Comic Sans MS', 11, 'bold'),
                             cursor="hand2", padx=20, pady=10)
        self.label.pack()

        self.underline = tk.Frame(self, bg=self.active_color if is_active else parent['bg'],
                                 height=3)
        self.underline.pack(fill=tk.X)

        self.label.bind("<Enter>", self.on_enter)
        self.label.bind("<Leave>", self.on_leave)
        self.label.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        if not self.is_active:
            self.label.config(fg=self.hover_color)

    def on_leave(self, e):
        if not self.is_active:
            self.label.config(fg=self.inactive_color)

    def on_click(self, e):
        if self.command:
            self.command()

    def set_active(self, active):
        self.is_active = active
        self.label.config(fg=self.active_color if active else self.inactive_color)
        self.underline.config(bg=self.active_color if active else self.parent_widget['bg'])


class CuteSheepProgressBar(tk.Canvas):
    """Ultra adorable animated sheep progress bar"""
    def __init__(self, parent, width=700, height=80, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.width = width
        self.height = height
        self.sheep = []
        self.is_animating = False
        self.frame_count = 0

    def draw_adorable_sheep(self, x, y, jump_phase):
        """Draw the most adorable sheep ever"""
        # Calculate jump
        jump = -abs(math.sin(jump_phase) * 20)
        y = y + jump

        # Cute shadow
        shadow = self.create_oval(x-15, y+25, x+15, y+28, fill="#E8E4F3", outline="")

        # Fluffy body
        body = self.create_oval(x-20, y-15, x+20, y+15, fill="#FFFFFF", outline="#FFB6D9", width=3)

        # Extra fluff
        fluff1 = self.create_oval(x-18, y-10, x-10, y-2, fill="#FFF5FF", outline="")
        fluff2 = self.create_oval(x+10, y-8, x+18, y, fill="#FFF5FF", outline="")
        fluff3 = self.create_oval(x-5, y+8, x+5, y+15, fill="#FFF5FF", outline="")

        # Adorable head
        head = self.create_oval(x+15, y-12, x+35, y+8, fill="#FFE4F0", outline="#FFB6D9", width=3)

        # Super cute ears
        ear1 = self.create_polygon(x+17, y-10, x+20, y-18, x+23, y-10,
                                   fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)
        ear2 = self.create_polygon(x+27, y-10, x+30, y-18, x+33, y-10,
                                   fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)

        # Inner ears
        canvas.create_polygon(x+17, y-10, x+20, y-15, x+23, y-10,
                            fill="#FFD4E5", outline="", smooth=True)
        canvas.create_polygon(x+27, y-10, x+30, y-15, x+33, y-10,
                            fill="#FFD4E5", outline="", smooth=True)

        # Big sparkly eyes
        eye1_white = self.create_oval(x+19, y-6, x+24, y-1, fill="#FFFFFF")
        eye1 = self.create_oval(x+20, y-5, x+23, y-2, fill="#2B2D42")
        sparkle1 = self.create_oval(x+21, y-4, x+22, y-3, fill="#FFFFFF")

        eye2_white = self.create_oval(x+26, y-6, x+31, y-1, fill="#FFFFFF")
        eye2 = self.create_oval(x+27, y-5, x+30, y-2, fill="#2B2D42")
        sparkle2 = self.create_oval(x+28, y-4, x+29, y-3, fill="#FFFFFF")

        # Tiny pink nose
        nose = self.create_oval(x+23, y+2, x+27, y+6, fill="#FFB6D9", outline="#FF6B9D", width=2)

        # Sweet smile
        smile = self.create_arc(x+20, y+3, x+30, y+9, start=0, extent=-180,
                               outline="#FF6B9D", width=3, style="arc")

        # Rosy cheeks
        cheek1 = self.create_oval(x+16, y+1, x+19, y+4, fill="#FFD4E5", outline="")
        cheek2 = self.create_oval(x+31, y+1, x+34, y+4, fill="#FFD4E5", outline="")

        # Cute legs
        leg_offset = abs(math.sin(jump_phase) * 3)
        leg1 = self.create_line(x-12, y+15, x-12, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        leg2 = self.create_line(x-4, y+15, x-4, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        leg3 = self.create_line(x+6, y+15, x+6, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        leg4 = self.create_line(x+14, y+15, x+14, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")

        # Tiny hooves
        hoof1 = self.create_oval(x-14, y+22-leg_offset, x-10, y+25-leg_offset, fill="#D4BBFF")
        hoof2 = self.create_oval(x-6, y+22+leg_offset, x-2, y+25+leg_offset, fill="#D4BBFF")
        hoof3 = self.create_oval(x+4, y+22-leg_offset, x+8, y+25-leg_offset, fill="#D4BBFF")
        hoof4 = self.create_oval(x+12, y+22+leg_offset, x+16, y+25+leg_offset, fill="#D4BBFF")

        # Fluffy tail
        tail = self.create_oval(x-22, y+5, x-16, y+11, fill="#FFFFFF", outline="#FFB6D9", width=2)

    def start(self):
        self.is_animating = True
        self.frame_count = 0
        self.sheep = []
        self._animate()

    def stop(self):
        self.is_animating = False
        self.delete("all")
        self.sheep = []
        self.frame_count = 0

    def _animate(self):
        if not self.is_animating:
            return

        self.delete("all")

        # Add new sheep
        if self.frame_count % 35 == 0:
            self.sheep.append({'x': -40, 'phase': 0})

        # Update and draw sheep
        new_sheep = []
        for sheep_data in self.sheep:
            sheep_data['x'] += 3.5
            sheep_data['phase'] += 0.25

            if sheep_data['x'] < self.width + 50:
                self.draw_adorable_sheep(sheep_data['x'], self.height // 2, sheep_data['phase'])
                new_sheep.append(sheep_data)

        self.sheep = new_sheep
        self.frame_count += 1

        self.after(35, self._animate)


class XRDProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XRD Batch Processing Suite")
        self.root.geometry("1100x900")
        self.root.resizable(True, True)

        # Modern color scheme
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
            'error': '#EF476F'
        }

        self.root.configure(bg=self.colors['bg'])

        # Variables for existing features
        self.poni_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.input_pattern = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.dataset_path = tk.StringVar(value="entry/data/data")
        self.npt = tk.IntVar(value=4000)
        self.unit = tk.StringVar(value='2th_deg')
        self.fit_method = tk.StringVar(value='pseudo')

        # New variables for volume calculation
        self.volume_input_file = tk.StringVar()  # Changed from folder to file
        self.volume_output_file = tk.StringVar()
        self.lattice_type = tk.StringVar(value='FCC')
        self.wavelength = tk.DoubleVar(value=0.4133)

        # New variables for Birch-Murnaghan fitting
        self.bm_input_file = tk.StringVar()
        self.bm_output_dir = tk.StringVar()
        self.bm_order = tk.StringVar(value='3')  # 2 or 3 order
        self.v0_guess = tk.DoubleVar(value=10.0)
        self.k0_guess = tk.DoubleVar(value=180.0)
        self.k0_prime_guess = tk.DoubleVar(value=4.0)

        self.current_tab = "powder"

        self.setup_ui()

    def setup_ui(self):
        """Create the modern UI layout"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        # Center the title
        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title with cat icon
        icon_label = tk.Label(header_content, text="🐱", bg=self.colors['card_bg'],
                             font=('Segoe UI Emoji', 32))
        icon_label.pack(side=tk.LEFT, padx=(0, 12))

        title = tk.Label(header_content, text="XRD Batch Processing Suite",
                        bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                        font=('Comic Sans MS', 20, 'bold'))
        title.pack(side=tk.LEFT)

        # Tab Navigation
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        tab_frame.pack(fill=tk.X, padx=30, pady=(10, 0))

        tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        tabs_container.pack(side=tk.LEFT)

        self.powder_tab = ModernTab(tabs_container, "Powder XRD",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 5))

        self.single_tab = ModernTab(tabs_container, "Single Crystal XRD",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=5)

        self.radial_tab = ModernTab(tabs_container, "Radial XRD",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT, padx=5)

        # Main scrollable container
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

        # Enhanced mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas

        # Content
        self.setup_powder_content()

    def setup_powder_content(self):
        """Setup powder XRD content with all features"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== Integration Settings Card =====
        integration_card = self.create_card_frame(main_frame)
        integration_card.pack(fill=tk.X, pady=(0, 15))

        card_content = tk.Frame(integration_card, bg=self.colors['card_bg'], padx=20, pady=12)
        card_content.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(card_content, bg=self.colors['card_bg'])
        header_frame.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header_frame, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header_frame, text="Integration Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # File inputs
        self.create_tiny_file_picker(card_content, "PONI File", self.poni_path,
                                    [("PONI files", "*.poni"), ("All files", "*.*")])

        self.create_tiny_file_picker(card_content, "Mask File", self.mask_path,
                                    [("EDF files", "*.edf"), ("All files", "*.*")])

        self.create_tiny_file_picker(card_content, "Input Pattern", self.input_pattern,
                                    [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)

        self.create_tiny_folder_picker(card_content, "Output Directory", self.output_dir)

        self.create_tiny_entry(card_content, "Dataset Path", self.dataset_path)

        # Parameters
        param_frame = tk.Frame(card_content, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(5, 0))

        # NPT
        npt_container = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        tk.Label(npt_container, text="Number of Points", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        npt_spinbox = ttk.Spinbox(npt_container, from_=500, to=10000, textvariable=self.npt,
                                 width=18, font=('Comic Sans MS', 9))
        npt_spinbox.pack(anchor=tk.W)
        npt_spinbox.config(cursor="hand2")

        # Unit
        unit_container = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(unit_container, text="Integration Unit", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        unit_combo = ttk.Combobox(unit_container, textvariable=self.unit,
                                 values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                                 width=16, state='readonly', font=('Comic Sans MS', 9))
        unit_combo.pack(anchor=tk.W)
        unit_combo.config(cursor="hand2")

        # ===== Fitting Settings Card =====
        fitting_card = self.create_card_frame(main_frame)
        fitting_card.pack(fill=tk.X, pady=(0, 15))

        card_content2 = tk.Frame(fitting_card, bg=self.colors['card_bg'], padx=20, pady=12)
        card_content2.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame2 = tk.Frame(card_content2, bg=self.colors['card_bg'])
        header_frame2.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header_frame2, text="🐹", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header_frame2, text="Peak Fitting Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        fit_container = tk.Frame(card_content2, bg=self.colors['card_bg'])
        fit_container.pack(fill=tk.X)

        tk.Label(fit_container, text="Fitting Method", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        fit_combo = ttk.Combobox(fit_container, textvariable=self.fit_method,
                                values=['pseudo', 'voigt'], width=22, state='readonly',
                                font=('Comic Sans MS', 9))
        fit_combo.pack(anchor=tk.W)
        fit_combo.config(cursor="hand2")

        # ===== MOVED: Action Buttons (Integration, Fitting, Full Pipeline) =====
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        button_frame.pack(fill=tk.X, pady=(0, 15))

        button_container = tk.Frame(button_frame, bg=self.colors['bg'])
        button_container.pack(expand=True)

        # Buttons row
        buttons_row = tk.Frame(button_container, bg=self.colors['bg'])
        buttons_row.pack()

        ModernButton(buttons_row, "Run Integration", self.run_integration, icon="🐿️",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(buttons_row, "Run Fitting", self.run_fitting, icon="🐻",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(buttons_row, "Full Pipeline", self.run_full_pipeline, icon="🦔",
                    bg_color=self.colors['primary'], hover_color=self.colors['accent'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        # ===== Volume Calculation Card (MODIFIED) =====
        volume_card = self.create_card_frame(main_frame)
        volume_card.pack(fill=tk.X, pady=(0, 15))

        vol_content = tk.Frame(volume_card, bg=self.colors['card_bg'], padx=20, pady=12)
        vol_content.pack(fill=tk.BOTH, expand=True)

        # Header
        vol_header = tk.Frame(vol_content, bg=self.colors['card_bg'])
        vol_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(vol_header, text="📐", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(vol_header, text="Volume Calculation Settings",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        # Changed from folder to CSV file input
        self.create_tiny_file_picker(vol_content, "Input CSV File (Peak Data)", self.volume_input_file,
                                    [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_tiny_file_picker(vol_content, "Output CSV File", self.volume_output_file,
                                    [("CSV files", "*.csv"), ("All files", "*.*")], save=True)

        # Parameters
        vol_param_frame = tk.Frame(vol_content, bg=self.colors['card_bg'])
        vol_param_frame.pack(fill=tk.X, pady=(5, 0))

        # Lattice Type - Updated options
        lattice_container = tk.Frame(vol_param_frame, bg=self.colors['card_bg'])
        lattice_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        tk.Label(lattice_container, text="Lattice Type", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        lattice_combo = ttk.Combobox(lattice_container, textvariable=self.lattice_type,
                                    values=['FCC', 'BCC', 'SC', 'Hexagonal', 'Tetragonal',
                                           'Orthorhombic', 'Monoclinic', 'Triclinic'],
                                    width=18, state='readonly', font=('Comic Sans MS', 9))
        lattice_combo.pack(anchor=tk.W)
        lattice_combo.config(cursor="hand2")

        # Wavelength
        wave_container = tk.Frame(vol_param_frame, bg=self.colors['card_bg'])
        wave_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(wave_container, text="Wavelength (Å)", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        wave_entry = tk.Entry(wave_container, textvariable=self.wavelength,
                             font=('Comic Sans MS', 9), width=20)
        wave_entry.pack(anchor=tk.W)

        # Button for volume calculation
        vol_button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        vol_button_frame.pack(fill=tk.X, pady=(0, 15))

        vol_button_container = tk.Frame(vol_button_frame, bg=self.colors['bg'])
        vol_button_container.pack(expand=True)

        ModernButton(vol_button_container, "Calculate Volumes", self.run_volume_calculation, icon="📊",
                    bg_color="#06D6A0", hover_color="#05B88A",
                    width=220, height=45).pack()

        # ===== Birch-Murnaghan Fitting Card (MODIFIED) =====
        bm_card = self.create_card_frame(main_frame)
        bm_card.pack(fill=tk.X, pady=(0, 15))

        bm_content = tk.Frame(bm_card, bg=self.colors['card_bg'], padx=20, pady=12)
        bm_content.pack(fill=tk.BOTH, expand=True)

        # Header
        bm_header = tk.Frame(bm_content, bg=self.colors['card_bg'])
        bm_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(bm_header, text="⚗️", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(bm_header, text="Birch-Murnaghan Equation of State",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_tiny_file_picker(bm_content, "Input CSV File (P-V Data)", self.bm_input_file,
                                    [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_tiny_folder_picker(bm_content, "Output Directory", self.bm_output_dir)

        # Order selection
        order_container = tk.Frame(bm_content, bg=self.colors['card_bg'])
        order_container.pack(fill=tk.X, pady=(5, 0))

        tk.Label(order_container, text="BM Equation Order", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        order_combo = ttk.Combobox(order_container, textvariable=self.bm_order,
                                  values=['2', '3'], width=18, state='readonly',
                                  font=('Comic Sans MS', 9))
        order_combo.pack(anchor=tk.W)
        order_combo.config(cursor="hand2")
        order_combo.bind('<<ComboboxSelected>>', self.on_bm_order_change)

        # Initial guess parameters
        bm_param_frame = tk.Frame(bm_content, bg=self.colors['card_bg'])
        bm_param_frame.pack(fill=tk.X, pady=(5, 0))

        # V0
        v0_container = tk.Frame(bm_param_frame, bg=self.colors['card_bg'])
        v0_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        tk.Label(v0_container, text="V₀ Initial Guess", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        v0_entry = tk.Entry(v0_container, textvariable=self.v0_guess,
                           font=('Comic Sans MS', 9), width=15)
        v0_entry.pack(anchor=tk.W)

        # K0
        k0_container = tk.Frame(bm_param_frame, bg=self.colors['card_bg'])
        k0_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        tk.Label(k0_container, text="K₀ Initial Guess", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        k0_entry = tk.Entry(k0_container, textvariable=self.k0_guess,
                           font=('Comic Sans MS', 9), width=15)
        k0_entry.pack(anchor=tk.W)

        # K0' - with conditional disable
        k0p_container = tk.Frame(bm_param_frame, bg=self.colors['card_bg'])
        k0p_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(k0p_container, text="K₀' Initial Guess", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        self.k0p_entry = tk.Entry(k0p_container, textvariable=self.k0_prime_guess,
                                   font=('Comic Sans MS', 9), width=15)
        self.k0p_entry.pack(anchor=tk.W)

        # Button for BM fitting
        bm_button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        bm_button_frame.pack(fill=tk.X, pady=(0, 15))

        bm_button_container = tk.Frame(bm_button_frame, bg=self.colors['bg'])
        bm_button_container.pack(expand=True)

        ModernButton(bm_button_container, "Birch-Murnaghan Fit", self.run_birch_murnaghan, icon="⚗️",
                    bg_color="#FF6B9D", hover_color="#FF8FB3",
                    width=220, height=45).pack()

        # ===== Sheep progress bar =====
        progress_container = tk.Frame(main_frame, bg=self.colors['bg'])
        progress_container.pack(fill=tk.X, pady=(0, 15))

        progress_inner = tk.Frame(progress_container, bg=self.colors['bg'])
        progress_inner.pack(expand=True)

        self.progress = CuteSheepProgressBar(progress_inner, width=780, height=80)
        self.progress.pack()

        # ===== Log Card =====
        log_card = self.create_card_frame(main_frame)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_content = tk.Frame(log_card, bg=self.colors['card_bg'], padx=20, pady=12)
        log_content.pack(fill=tk.BOTH, expand=True)

        # Header
        log_header_frame = tk.Frame(log_content, bg=self.colors['card_bg'])
        log_header_frame.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(log_header_frame, text="🐰", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(log_header_frame, text="Process Log",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(log_content, height=10, wrap=tk.WORD,
                                                  font=('Comic Sans MS', 10),
                                                  bg='#FAFAFA', fg='#B794F6',
                                                  relief='flat', borderwidth=0, padx=10, pady=10,
                                                  spacing1=3, spacing2=2, spacing3=3)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_bm_order_change(self, event=None):
        """Handle BM order change - disable K0' entry if order is 2"""
        if self.bm_order.get() == '2':
            self.k0p_entry.config(state='disabled')
            self.k0_prime_guess.set(4.0)  # Fixed value for 2nd order
        else:
            self.k0p_entry.config(state='normal')

    def switch_tab(self, tab_name):
        """Switch between tabs"""
        self.current_tab = tab_name

        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if tab_name == "powder":
            self.setup_powder_content()
        elif tab_name == "single":
            self.setup_placeholder_content("Single Crystal XRD",
                                          "Single crystal analysis features coming soon...")
        else:
            self.setup_placeholder_content("Radial XRD",
                                          "Radial integration features coming soon...")

    def setup_placeholder_content(self, title, message):
        """Setup placeholder content"""
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

    def create_card_frame(self, parent, **kwargs):
        """Create a modern card-style frame"""
        card = tk.Frame(parent, bg=self.colors['card_bg'],
                       relief='flat', borderwidth=0, **kwargs)
        card.configure(highlightbackground=self.colors['border'],
                      highlightthickness=1)
        return card

    def create_tiny_file_picker(self, parent, label, variable, filetypes, pattern=False, save=False):
        """Create ultra compact file picker"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if save:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_save_file(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        elif pattern:
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

    def create_tiny_folder_picker(self, parent, label, variable):
        """Create ultra compact folder picker"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        btn = ModernButton(input_frame, "Browse",
                         lambda: self.browse_folder(variable),
                         bg_color=self.colors['secondary'],
                         hover_color=self.colors['primary'],
                         width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_tiny_entry(self, parent, label, variable):
        """Create ultra compact entry"""
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        entry = tk.Entry(container, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(fill=tk.X, ipady=3)

    def browse_file(self, variable, filetypes):
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            variable.set(filename)

    def browse_save_file(self, variable, filetypes):
        filename = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".csv")
        if filename:
            variable.set(filename)

    def browse_pattern(self, variable, filetypes):
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            folder = os.path.dirname(filename)
            ext = os.path.splitext(filename)[1]
            pattern = os.path.join(folder, f"*{ext}")
            variable.set(pattern)

    def browse_folder(self, variable):
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def log(self, message):
        """Add message to log with compact double-ribbon separator"""
        self.log_text.config(state='normal')

        # Double pink ribbon rows, closer together
        if "🎀" in message and "🎀 " * 15 in message:
            self.log_text.insert(tk.END, "🎀 " * 10 + "\n")
            self.log_text.insert(tk.END, "🎀 " * 10 + "\n")
        else:
            self.log_text.insert(tk.END, message + "\n")

        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def draw_cute_cat_with_cookies(self, canvas, x, y):
        """Draw super adorable pink-purple cat munching cookies"""
        # Adorable cat body
        body = canvas.create_oval(x-45, y-35, x+45, y+35,
                                 fill="#E0AAFF", outline="#B794F6", width=3)

        # Cute pointy ears
        ear1 = canvas.create_polygon(x-40, y-35, x-30, y-60, x-20, y-35,
                                    fill="#FFB6D9", outline="#B794F6", width=2, smooth=True)
        ear2 = canvas.create_polygon(x+20, y-35, x+30, y-60, x+40, y-35,
                                    fill="#FFB6D9", outline="#B794F6", width=2, smooth=True)

        # Inner ears
        canvas.create_polygon(x-35, y-35, x-30, y-50, x-25, y-35,
                            fill="#FFD4E5", outline="", smooth=True)
        canvas.create_polygon(x+25, y-35, x+30, y-50, x+35, y-35,
                            fill="#FFD4E5", outline="", smooth=True)

        # Big happy eyes
        eye1_w = canvas.create_oval(x-25, y-15, x-10, y, fill="#FFFFFF")
        eye1 = canvas.create_oval(x-22, y-12, x-13, y-3, fill="#2B2D42")
        sparkle1 = canvas.create_oval(x-20, y-10, x-17, y-7, fill="#FFFFFF")

        eye2_w = canvas.create_oval(x+10, y-15, x+25, y, fill="#FFFFFF")
        eye2 = canvas.create_oval(x+13, y-12, x+22, y-3, fill="#2B2D42")
        sparkle2 = canvas.create_oval(x+17, y-10, x+20, y-7, fill="#FFFFFF")

        # Cookie in mouth
        canvas.create_oval(x-8, y+5, x+8, y+21, fill="#D4A574", outline="#8B6F47", width=2)
        canvas.create_oval(x-3, y+9, x, y+12, fill="#4A2C1B")
        canvas.create_oval(x+2, y+11, x+5, y+14, fill="#4A2C1B")

        # Big smile
        smile = canvas.create_arc(x-18, y+8, x+18, y+25, start=0, extent=-180,
                                 outline="#FF6B9D", width=3, style="arc")

        # Pink nose
        nose = canvas.create_polygon(x-4, y+5, x, y+2, x+4, y+5, x, y+8,
                                    fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)

        # Rosy cheeks
        canvas.create_oval(x-42, y+5, x-32, y+15, fill="#FFD4E5", outline="")
        canvas.create_oval(x+32, y+5, x+42, y+15, fill="#FFD4E5", outline="")

        # Cute whiskers
        canvas.create_line(x-45, y+5, x-65, y, fill="#B794F6", width=2)
        canvas.create_line(x-45, y+10, x-65, y+10, fill="#B794F6", width=2)
        canvas.create_line(x-45, y+15, x-65, y+20, fill="#B794F6", width=2)
        canvas.create_line(x+45, y+5, x+65, y, fill="#B794F6", width=2)
        canvas.create_line(x+45, y+10, x+65, y+10, fill="#B794F6", width=2)
        canvas.create_line(x+45, y+15, x+65, y+20, fill="#B794F6", width=2)

        # Random cookies scattered around
        cookie_spots = [
            (x-90, y-40), (x-70, y+50), (x+75, y-35), (x+85, y+45),
            (x-95, y+10), (x+95, y+5), (x-60, y+70), (x+65, y+65),
            (x-110, y-10), (x+105, y-15), (x-80, y+30), (x+90, y+25),
            (x-100, y-25), (x+100, y-30)
        ]

        for cx, cy in cookie_spots:
            # Randomize size slightly
            size = random.randint(10, 14)
            canvas.create_oval(cx-size, cy-size, cx+size, cy+size,
                             fill="#D4A574", outline="#8B6F47", width=2)
            # Random chocolate chips
            for _ in range(random.randint(2, 4)):
                chip_x = cx + random.randint(-size+3, size-3)
                chip_y = cy + random.randint(-size+3, size-3)
                chip_size = random.randint(2, 4)
                canvas.create_oval(chip_x-chip_size, chip_y-chip_size,
                                 chip_x+chip_size, chip_y+chip_size, fill="#4A2C1B")

    def show_cute_success(self, message):
        """Show super cute cat eating cookies"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Success")
        dialog.geometry("450x380")
        dialog.configure(bg=self.colors['card_bg'])
        dialog.resizable(False, False)

        dialog.transient(self.root)
        dialog.grab_set()

        # Canvas for cat and cookies
        cat_canvas = tk.Canvas(dialog, width=430, height=250,
                              bg=self.colors['card_bg'], highlightthickness=0)
        cat_canvas.pack(pady=(20, 10))

        self.draw_cute_cat_with_cookies(cat_canvas, 215, 125)

        # Message
        tk.Label(dialog, text=message, bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Comic Sans MS', 13, 'bold'),
                wraplength=400).pack(pady=(10, 20))

        # OK button
        ok_btn = ModernButton(dialog, "OK", dialog.destroy,
                            bg_color=self.colors['primary'],
                            hover_color=self.colors['primary_hover'],
                            width=120, height=40)
        ok_btn.pack()

    # ===== Validation Methods =====

    def validate_integration_inputs(self):
        if not self.poni_path.get():
            messagebox.showerror("Error", "Please select a PONI file")
            return False
        if not self.mask_path.get():
            messagebox.showerror("Error", "Please select a mask file")
            return False
        if not self.input_pattern.get():
            messagebox.showerror("Error", "Please select input files")
            return False
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select output directory")
            return False
        return True

    def validate_fitting_inputs(self):
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select output directory containing .xy files")
            return False
        return True

    def validate_volume_inputs(self):
        if not self.volume_input_file.get():
            messagebox.showerror("Error", "Please select input CSV file containing peak data")
            return False
        if not self.volume_output_file.get():
            messagebox.showerror("Error", "Please specify output CSV file")
            return False
        return True

    def validate_birch_murnaghan_inputs(self):
        if not self.bm_input_file.get():
            messagebox.showerror("Error", "Please select input CSV file with P-V data")
            return False
        if not self.bm_output_dir.get():
            messagebox.showerror("Error", "Please select output directory")
            return False
        return True

    # ===== Existing Feature Methods =====

    def run_integration(self):
        if not self.validate_integration_inputs():
            return
        thread = threading.Thread(target=self._run_integration_thread)
        thread.daemon = True
        thread.start()

    def _run_integration_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("🔁 Starting Batch Integration (HDF5 ➜ .xy)")

            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            dataset = self.dataset_path.get() if self.dataset_path.get() else None

            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=dataset,
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr',
                safe=True,
                normalization_factor=1.0
            )

            self.log("✅ Integration completed successfully!")
            self.show_cute_success("Integration completed successfully!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Integration failed:\n{str(e)}")
        finally:
            self.progress.stop()

    def run_fitting(self):
        if not self.validate_fitting_inputs():
            return
        thread = threading.Thread(target=self._run_fitting_thread)
        thread.daemon = True
        thread.start()

    def _run_fitting_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("📈 Starting Batch Fitting (.xy ➜ fitted peaks)")

            fitter = BatchFitter(
                folder=self.output_dir.get(),
                fit_method=self.fit_method.get()
            )
            fitter.run_batch_fitting()

            self.log("✅ Fitting completed successfully!")
            self.show_cute_success("Fitting completed successfully!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Fitting failed:\n{str(e)}")
        finally:
            self.progress.stop()

    def run_full_pipeline(self):
        if not self.validate_integration_inputs():
            return
        thread = threading.Thread(target=self._run_full_pipeline_thread)
        thread.daemon = True
        thread.start()

    def _run_full_pipeline_thread(self):
        try:
            self.progress.start()

            self.log("🎀 " * 15)
            self.log("🔁 Step 1/2: Running Batch Integration")

            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            dataset = self.dataset_path.get() if self.dataset_path.get() else None

            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=dataset,
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr',
                safe=True,
                normalization_factor=1.0
            )

            self.log("✅ Integration completed!")

            self.log("🎀 " * 15)
            self.log("📈 Step 2/2: Running Batch Fitting")

            fitter = BatchFitter(
                folder=self.output_dir.get(),
                fit_method=self.fit_method.get()
            )
            fitter.run_batch_fitting()

            self.log("✅ Fitting completed!")
            self.log("🎀 " * 15)
            self.log("🎉 Full pipeline completed successfully!")

            self.show_cute_success("Full pipeline completed successfully!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Pipeline failed:\n{str(e)}")
        finally:
            self.progress.stop()

    # ===== New Feature Methods =====

    def run_volume_calculation(self):
        """Run XRayDiffractionAnalyzer to calculate volumes"""
        if not self.validate_volume_inputs():
            return
        thread = threading.Thread(target=self._run_volume_calculation_thread)
        thread.daemon = True
        thread.start()

    def _run_volume_calculation_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("📐 Starting Volume Calculation from Peak Positions")

            analyzer = XRayDiffractionAnalyzer(
                csv_file=self.volume_input_file.get(),
                lattice_type=self.lattice_type.get(),
                wavelength=self.wavelength.get()
            )

            analyzer.process_all_files()
            analyzer.save_results(self.volume_output_file.get())

            self.log(f"✅ Volume calculation completed!")
            self.log(f"📊 Results saved to: {self.volume_output_file.get()}")
            self.show_cute_success("Volume calculation completed successfully!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Volume calculation failed:\n{str(e)}")
        finally:
            self.progress.stop()

    def run_birch_murnaghan(self):
        """Run Birch-Murnaghan equation of state fitting"""
        if not self.validate_birch_murnaghan_inputs():
            return
        thread = threading.Thread(target=self._run_birch_murnaghan_thread)
        thread.daemon = True
        thread.start()

    def _run_birch_murnaghan_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            order_str = "2nd" if self.bm_order.get() == '2' else "3rd"
            self.log(f"⚗️ Starting {order_str} Order Birch-Murnaghan Equation of State Fitting")

            fitter = BirchMurnaghanFitter(
                data_file=self.bm_input_file.get(),
                output_dir=self.bm_output_dir.get(),
                order=int(self.bm_order.get())
            )

            # Set initial guesses
            if self.bm_order.get() == '2':
                # For 2nd order, K0' is fixed at 4
                initial_guess = [
                    self.v0_guess.get(),
                    self.k0_guess.get()
                ]
            else:
                # For 3rd order, include K0'
                initial_guess = [
                    self.v0_guess.get(),
                    self.k0_guess.get(),
                    self.k0_prime_guess.get()
                ]

            results = fitter.fit(initial_guess=initial_guess)

            self.log("✅ Birch-Murnaghan fitting completed!")
            self.log(f"📊 V₀ = {results['V0']:.4f} ± {results['V0_err']:.4f}")
            self.log(f"📊 K₀ = {results['K0']:.4f} ± {results['K0_err']:.4f} GPa")

            if self.bm_order.get() == '3':
                self.log(f"📊 K₀' = {results['K0_prime']:.4f} ± {results['K0_prime_err']:.4f}")
            else:
                self.log(f"📊 K₀' = 4.0 (fixed for 2nd order)")

            self.log(f"📈 R² = {results['r_squared']:.6f}")

            self.show_cute_success(f"{order_str} Order Birch-Murnaghan fitting completed successfully!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Birch-Murnaghan fitting failed:\n{str(e)}")
        finally:
            self.progress.stop()


def main():
    root = tk.Tk()
    app = XRDProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
