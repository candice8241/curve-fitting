#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Azimuthal Integration (Radial XRD) Module
Handles radial integration of 2D XRD patterns
"""

import tkinter as tk
from tkinter import ttk
from theme_module import GUIBase, ModernButton


class AzimuthalIntegrationModule(GUIBase):
    """Module for Azimuthal Integration / Radial XRD processing"""

    def __init__(self, parent, root):
        """
        Initialize Radial XRD module

        Args:
            parent: Parent frame to contain this module
            root: Root window
        """
        super().__init__()
        self.parent = parent
        self.root = root

    def setup_ui(self):
        """Setup the user interface for Radial XRD module"""
        # Title
        title_frame = tk.Frame(self.parent, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            title_frame,
            text="🔄 Radial XRD Integration",
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
            text="Perform azimuthal integration on 2D XRD detector images",
            font=('Segoe UI', 11),
            bg=self.colors['card_bg'],
            fg=self.colors['text_light'],
            wraplength=600,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 20))

        # Image file selection
        file_frame = tk.LabelFrame(
            content,
            text="2D Detector Images",
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

        ModernButton(file_frame, "Browse Images", command=self._browse_images).pack(side=tk.LEFT)

        # Integration parameters
        params_frame = tk.LabelFrame(
            content,
            text="Integration Parameters",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_dark'],
            padx=15,
            pady=15
        )
        params_frame.pack(fill=tk.X, pady=10)

        # Wavelength
        wavelength_row = tk.Frame(params_frame, bg=self.colors['card_bg'])
        wavelength_row.pack(fill=tk.X, pady=5)
        tk.Label(wavelength_row, text="Wavelength (Å):", bg=self.colors['card_bg'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Entry(wavelength_row, font=('Segoe UI', 10), width=15).pack(side=tk.LEFT)

        # Distance
        distance_row = tk.Frame(params_frame, bg=self.colors['card_bg'])
        distance_row.pack(fill=tk.X, pady=5)
        tk.Label(distance_row, text="Sample Distance (mm):", bg=self.colors['card_bg'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Entry(distance_row, font=('Segoe UI', 10), width=15).pack(side=tk.LEFT)

        # Radial bins
        bins_row = tk.Frame(params_frame, bg=self.colors['card_bg'])
        bins_row.pack(fill=tk.X, pady=5)
        tk.Label(bins_row, text="Number of Radial Bins:", bg=self.colors['card_bg'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Entry(bins_row, font=('Segoe UI', 10), width=15).pack(side=tk.LEFT)

        # Action buttons
        button_frame = tk.Frame(content, bg=self.colors['card_bg'])
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ModernButton(button_frame, "Integrate", command=self._integrate).pack(side=tk.LEFT, padx=(0, 10))
        ModernButton(button_frame, "Save Pattern", command=self._save_pattern).pack(side=tk.LEFT)

    def _browse_images(self):
        """Handle image browsing"""
        print("Browse images clicked - Radial XRD")

    def _integrate(self):
        """Handle integration"""
        print("Integrate clicked - Radial XRD")

    def _save_pattern(self):
        """Handle pattern saving"""
        print("Save pattern clicked - Radial XRD")
