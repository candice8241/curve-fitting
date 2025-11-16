#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Theme Module
Provides base classes and themed widgets for the XRD GUI
"""

import tkinter as tk
from tkinter import ttk


class GUIBase:
    """Base class providing color scheme and common styling"""

    def __init__(self):
        self.colors = {
            'bg': '#F5F5F5',
            'card_bg': '#FFFFFF',
            'primary': '#6200EA',
            'primary_hover': '#7C4DFF',
            'text_dark': '#212121',
            'text_light': '#757575',
            'border': '#E0E0E0',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336'
        }


class ModernTab(tk.Frame):
    """Modern styled tab button"""

    def __init__(self, parent, text, command, is_active=False):
        super().__init__(parent, bg='#F5F5F5')

        self.text = text
        self.command = command
        self.is_active = is_active

        self.button = tk.Button(
            self,
            text=text,
            command=command,
            relief=tk.FLAT,
            borderwidth=0,
            padx=20,
            pady=10,
            font=('Segoe UI', 10, 'bold' if is_active else 'normal'),
            bg='#6200EA' if is_active else '#FFFFFF',
            fg='#FFFFFF' if is_active else '#212121',
            cursor='hand2'
        )
        self.button.pack()

        # Hover effects
        self.button.bind('<Enter>', self._on_enter)
        self.button.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        if not self.is_active:
            self.button.config(bg='#E8EAF6')

    def _on_leave(self, event):
        if not self.is_active:
            self.button.config(bg='#FFFFFF')

    def set_active(self, active):
        """Set the active state of the tab"""
        self.is_active = active
        self.button.config(
            font=('Segoe UI', 10, 'bold' if active else 'normal'),
            bg='#6200EA' if active else '#FFFFFF',
            fg='#FFFFFF' if active else '#212121'
        )


class ModernButton(tk.Button):
    """Modern styled button"""

    def __init__(self, parent, text, command=None, **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            relief=tk.FLAT,
            borderwidth=0,
            padx=20,
            pady=10,
            font=('Segoe UI', 10),
            bg='#6200EA',
            fg='#FFFFFF',
            cursor='hand2',
            **kwargs
        )

        self.bind('<Enter>', lambda e: self.config(bg='#7C4DFF'))
        self.bind('<Leave>', lambda e: self.config(bg='#6200EA'))


class CuteSheepProgressBar(tk.Canvas):
    """Cute animated progress bar"""

    def __init__(self, parent, width=400, height=30, **kwargs):
        super().__init__(parent, width=width, height=height, bg='white',
                        highlightthickness=1, highlightbackground='#E0E0E0', **kwargs)

        self.width = width
        self.height = height
        self.progress = 0

        # Create progress bar background
        self.create_rectangle(0, 0, width, height, fill='#F5F5F5', outline='')

        # Create progress indicator
        self.progress_rect = self.create_rectangle(
            0, 0, 0, height, fill='#6200EA', outline=''
        )

    def set_progress(self, value):
        """Set progress value (0-100)"""
        self.progress = max(0, min(100, value))
        progress_width = (self.width * self.progress) / 100
        self.coords(self.progress_rect, 0, 0, progress_width, self.height)
        self.update_idletasks()

    def reset(self):
        """Reset progress to 0"""
        self.set_progress(0)
