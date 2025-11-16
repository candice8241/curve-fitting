#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Crystal XRD Module
Handles single crystal X-ray diffraction data processing
"""

import tkinter as tk
from tkinter import ttk
from theme_module import GUIBase, ModernButton


class SingleCrystalModule(GUIBase):
    """Module for Single Crystal XRD data processing"""

    def __init__(self, parent, root):
        """
        Initialize Single Crystal XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root window
        """
        super().__init__()
        self.parent = parent
        self.root = root

    def setup_ui(self):
        """Setup the user interface for Single Crystal XRD module"""
        # Title
        title_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            title_frame,
            text="💎 Single Crystal XRD Analysis",
            font=('Comic Sans MS', 16, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text_dark']
        ).pack(anchor=tk.W)

        # Content card
        card = tk.Frame(self.parent, bg=self.colors['card_bg'], relief=tk.FLAT)
        card.pack(fill=tk.BOTH, expand=True, pady=10)

        # Padding inside card
        content = tk.Frame(card, bg=self.colors['card_bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # Description
        tk.Label(
            content,
            text="Process and analyze single crystal X-ray diffraction data",
            font=('Segoe UI', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_light'],
            wraplength=600,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 20))

        # Frame file selection
        file_frame = tk.LabelFrame(
            content,
            text="Diffraction Frames",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dark'],
            padx=15,
            pady=15
        )
        file_frame.pack(fill=tk.X, pady=10)

        tk.Entry(
            file_frame,
            font=('Segoe UI', 10),
            width=50
        ).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(file_frame, "Browse Frames", command=self._browse_frames).pack(side=tk.LEFT)

        # Crystal parameters
        crystal_frame = tk.LabelFrame(
            content,
            text="Crystal Parameters",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dark'],
            padx=15,
            pady=15
        )
        crystal_frame.pack(fill=tk.X, pady=10)

        # Space group
        sg_row = tk.Frame(crystal_frame, bg=self.colors['card_bg'])
        sg_row.pack(fill=tk.X, pady=5)
        tk.Label(sg_row, text="Space Group:", bg=self.colors['card_bg'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Entry(sg_row, font=('Segoe UI', 10), width=20).pack(side=tk.LEFT)

        # Unit cell
        cell_row = tk.Frame(crystal_frame, bg=self.colors['card_bg'])
        cell_row.pack(fill=tk.X, pady=5)
        tk.Label(cell_row, text="Unit Cell (a, b, c):", bg=self.colors['card_bg'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Entry(cell_row, font=('Segoe UI', 10), width=30).pack(side=tk.LEFT)

        # Analysis options
        analysis_frame = tk.LabelFrame(
            content,
            text="Analysis Options",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dark'],
            padx=15,
            pady=15
        )
        analysis_frame.pack(fill=tk.X, pady=10)

        tk.Checkbutton(
            analysis_frame,
            text="Peak indexing",
            bg=self.colors['card_bg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=5)

        tk.Checkbutton(
            analysis_frame,
            text="Structure refinement",
            bg=self.colors['card_bg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=5)

        tk.Checkbutton(
            analysis_frame,
            text="Generate CIF file",
            bg=self.colors['card_bg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=5)

        # Action buttons
        button_frame = tk.Frame(content, bg=self.colors['card_bg'])
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ModernButton(button_frame, "Process Crystal", command=self._process_crystal).pack(side=tk.LEFT, padx=(0, 10))
        ModernButton(button_frame, "Export CIF", command=self._export_cif).pack(side=tk.LEFT)

    def _browse_frames(self):
        """Handle frame browsing"""
        print("Browse frames clicked - Single Crystal XRD")

    def _process_crystal(self):
        """Handle crystal processing"""
        print("Process crystal clicked - Single Crystal XRD")

    def _export_cif(self):
        """Handle CIF export"""
        print("Export CIF clicked - Single Crystal XRD")
