#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Powder XRD Module
Handles powder X-ray diffraction data processing
"""

import tkinter as tk
from tkinter import ttk
from theme_module import GUIBase, ModernButton


class PowderXRDModule(GUIBase):
    """Module for Powder XRD data processing"""

    def __init__(self, parent, root):
        """
        Initialize Powder XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root window
        """
        super().__init__()
        self.parent = parent
        self.root = root

    def setup_ui(self):
        """Setup the user interface for Powder XRD module"""
        # Title
        title_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            title_frame,
            text="⚛️ Powder XRD Analysis",
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
            text="Process and analyze powder X-ray diffraction patterns",
            font=('Segoe UI', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_light'],
            wraplength=600,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 20))

        # File selection section
        file_frame = tk.LabelFrame(
            content,
            text="Data Files",
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

        ModernButton(file_frame, "Browse Files", command=self._browse_files).pack(side=tk.LEFT)

        # Processing options
        options_frame = tk.LabelFrame(
            content,
            text="Processing Options",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dark'],
            padx=15,
            pady=15
        )
        options_frame.pack(fill=tk.X, pady=10)

        tk.Checkbutton(
            options_frame,
            text="Background subtraction",
            bg=self.colors['card_bg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=5)

        tk.Checkbutton(
            options_frame,
            text="Peak fitting",
            bg=self.colors['card_bg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W, pady=5)

        # Action buttons
        button_frame = tk.Frame(content, bg=self.colors['card_bg'])
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ModernButton(button_frame, "Process Data", command=self._process_data).pack(side=tk.LEFT, padx=(0, 10))
        ModernButton(button_frame, "Export Results", command=self._export_results).pack(side=tk.LEFT)

    def _browse_files(self):
        """Handle file browsing"""
        print("Browse files clicked - Powder XRD")

    def _process_data(self):
        """Handle data processing"""
        print("Process data clicked - Powder XRD")

    def _export_results(self):
        """Handle results export"""
        print("Export results clicked - Powder XRD")
