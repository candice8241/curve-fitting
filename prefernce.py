# -*- coding: utf-8 -*-
"""
GUI Base Classes and Shared Components
XRD Data Post-Processing Suite - Common UI elements
"""

import tkinter as tk
from tkinter import ttk


class GUIBase:
    """Base class for GUI components with shared styling"""

    def __init__(self):
        """Initialize base GUI with color scheme"""
        self.colors = {
            'bg': '#F8F3FF',
            'card_bg': '#FFFFFF',
            'primary': '#B794F6',
            'accent': '#DDA0DD',
            'text_dark': '#4A4A4A',
            'text_light': '#888888',
            'success': '#90EE90',
            'error': '#FF6B6B'
        }

    def create_card_frame(self, parent):
        """Create a card-style frame"""
        frame = tk.Frame(parent, bg='#FFFFFF', relief='flat', borderwidth=0)
        frame.configure(highlightbackground='#E8E4F3', highlightthickness=1)
        return frame


class ModernButton(tk.Button):
    """Modern styled button with hover effects"""

    def __init__(self, parent, text, command, **kwargs):
        """
        Initialize modern button

        Args:
            parent: Parent widget
            text: Button text
            command: Command to execute on click
            **kwargs: Additional button options
        """
        bg_color = kwargs.pop('bg', '#B794F6')
        fg_color = kwargs.pop('fg', 'white')

        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2',
            **kwargs
        )

        self.default_bg = bg_color
        self.hover_bg = '#DDA0DD'

        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        """Handle mouse enter event"""
        self['bg'] = self.hover_bg

    def on_leave(self, e):
        """Handle mouse leave event"""
        self['bg'] = self.default_bg


class ModernTab(tk.Frame):
    """Modern tab button with active state"""

    def __init__(self, parent, text, command, is_active=False):
        """
        Initialize modern tab

        Args:
            parent: Parent widget
            text: Tab text
            command: Command to execute on click
            is_active: Whether tab is currently active
        """
        super().__init__(parent, bg='#F8F3FF')

        self.text = text
        self.command = command
        self.is_active = is_active

        self.active_color = '#B794F6'
        self.inactive_color = '#E8E4F3'

        self.button = tk.Button(
            self,
            text=text,
            command=self.on_click,
            bg=self.active_color if is_active else self.inactive_color,
            fg='white' if is_active else '#888888',
            font=('Arial', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.button.pack()

        if not is_active:
            self.button.bind('<Enter>', self.on_hover)
            self.button.bind('<Leave>', self.on_leave)

    def on_click(self):
        """Handle tab click"""
        if self.command:
            self.command()

    def on_hover(self, e):
        """Handle mouse hover"""
        if not self.is_active:
            self.button['bg'] = '#DDA0DD'

    def on_leave(self, e):
        """Handle mouse leave"""
        if not self.is_active:
            self.button['bg'] = self.inactive_color

    def set_active(self, active):
        """
        Set tab active state

        Args:
            active: Whether tab should be active
        """
        self.is_active = active
        self.button['bg'] = self.active_color if active else self.inactive_color
        self.button['fg'] = 'white' if active else '#888888'

        if active:
            self.button.unbind('<Enter>')
            self.button.unbind('<Leave>')
        else:
            self.button.bind('<Enter>', self.on_hover)
            self.button.bind('<Leave>', self.on_leave)


class CuteSheepProgressBar(tk.Canvas):
    """Cute sheep animation progress bar"""

    def __init__(self, parent, width=780, height=80):
        """
        Initialize progress bar

        Args:
            parent: Parent widget
            width: Canvas width
            height: Canvas height
        """
        super().__init__(parent, width=width, height=height, bg='#FFF5F7', highlightthickness=0)
        self.width = width
        self.height = height
        self.sheep = None
        self.is_running = False
        self.position = 0

    def start(self):
        """Start the animation"""
        if not self.is_running:
            self.is_running = True
            self.position = 0
            self._draw_sheep()
            self._animate()

    def stop(self):
        """Stop the animation"""
        self.is_running = False
        self.delete("all")

    def _draw_sheep(self):
        """Draw a cute sheep"""
        x = self.position
        y = self.height // 2

        # Body
        self.create_oval(x, y-15, x+40, y+15, fill='#F0E6FA', outline='#D8BFD8', width=2, tags='sheep')

        # Head
        self.create_oval(x+30, y-10, x+50, y+10, fill='#F0E6FA', outline='#D8BFD8', width=2, tags='sheep')

        # Eyes
        self.create_oval(x+38, y-5, x+42, y-1, fill='black', tags='sheep')
        self.create_oval(x+42, y-5, x+46, y-1, fill='black', tags='sheep')

        # Legs
        for leg_x in [x+8, x+16, x+24, x+32]:
            self.create_line(leg_x, y+15, leg_x, y+25, fill='#D8BFD8', width=3, tags='sheep')

    def _animate(self):
        """Animate the sheep"""
        if not self.is_running:
            return

        self.delete('sheep')
        self.position = (self.position + 5) % (self.width - 50)
        self._draw_sheep()

        self.after(50, self._animate)
