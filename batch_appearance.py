# -*- coding: utf-8 -*-
"""
GUI Appearance Components for XRD Processing Suite
"""

import tkinter as tk
from tkinter import ttk


class ModernButton(tk.Frame):
    """Modern styled button with hover effects"""

    def __init__(self, parent, text, command, icon="", bg_color="#B794F6",
                 hover_color="#D4BBFF", width=120, height=40):
        super().__init__(parent, bg=parent['bg'])

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color

        # Create button
        self.button = tk.Button(
            self,
            text=f"{icon} {text}" if icon else text,
            command=command,
            font=('Comic Sans MS', 10, 'bold'),
            bg=bg_color,
            fg='white',
            activebackground=hover_color,
            relief='flat',
            borderwidth=0,
            width=width // 8,  # Approximate character width
            height=height // 20,  # Approximate line height
            cursor='hand2'
        )
        self.button.pack(fill=tk.BOTH, expand=True)

        # Bind hover events
        self.button.bind('<Enter>', self._on_enter)
        self.button.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        """Mouse hover in"""
        self.button.config(bg=self.hover_color)

    def _on_leave(self, event):
        """Mouse hover out"""
        self.button.config(bg=self.bg_color)


class ModernTab(tk.Frame):
    """Modern tab button"""

    def __init__(self, parent, text, command, is_active=False):
        super().__init__(parent, bg=parent['bg'])

        self.text = text
        self.command = command
        self.is_active = is_active

        colors = {
            'bg': '#F8F7FF',
            'primary': '#B794F6',
            'border': '#E8E4F3',
            'text_dark': '#2B2D42',
            'text_light': '#8B8BA7'
        }

        self.colors = colors

        self.button = tk.Button(
            self,
            text=text,
            command=self._on_click,
            font=('Comic Sans MS', 11, 'bold' if is_active else 'normal'),
            bg=colors['primary'] if is_active else colors['bg'],
            fg='white' if is_active else colors['text_light'],
            activebackground=colors['primary'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.button.pack()

    def _on_click(self):
        """Handle tab click"""
        self.command()

    def set_active(self, active):
        """Set tab active state"""
        self.is_active = active
        if active:
            self.button.config(
                bg=self.colors['primary'],
                fg='white',
                font=('Comic Sans MS', 11, 'bold')
            )
        else:
            self.button.config(
                bg=self.colors['bg'],
                fg=self.colors['text_light'],
                font=('Comic Sans MS', 11, 'normal')
            )


class CuteSheepProgressBar(tk.Canvas):
    """Cute animated progress bar"""

    def __init__(self, parent, width=600, height=60):
        super().__init__(parent, width=width, height=height,
                         bg='#F8F7FF', highlightthickness=0)

        self.width = width
        self.height = height
        self.is_running = False
        self.position = 0

        # Draw progress bar background
        self.bar_height = 20
        self.bar_y = height // 2
        self.bg_rect = self.create_rectangle(
            50, self.bar_y - 10, width - 50, self.bar_y + 10,
            fill='#E8E4F3', outline='#B794F6', width=2
        )

        # Progress fill
        self.progress_rect = self.create_rectangle(
            50, self.bar_y - 10, 50, self.bar_y + 10,
            fill='#B794F6', outline=''
        )

        # Cute sheep emoji
        self.sheep = self.create_text(
            50, self.bar_y,
            text='🐑',
            font=('Segoe UI Emoji', 24),
            anchor=tk.W
        )

        self.hide()

    def start(self):
        """Start progress animation"""
        self.is_running = True
        self.position = 0
        self.pack()
        self._animate()

    def stop(self):
        """Stop progress animation"""
        self.is_running = False
        self.position = 0
        self.coords(self.progress_rect, 50, self.bar_y - 10, 50, self.bar_y + 10)
        self.coords(self.sheep, 50, self.bar_y)
        self.hide()

    def hide(self):
        """Hide progress bar"""
        self.pack_forget()

    def _animate(self):
        """Animate progress"""
        if not self.is_running:
            return

        self.position += 2
        if self.position > self.width - 100:
            self.position = 0

        # Update progress rectangle
        x1 = 50
        x2 = 50 + self.position
        self.coords(self.progress_rect, x1, self.bar_y - 10, x2, self.bar_y + 10)

        # Update sheep position
        self.coords(self.sheep, x2, self.bar_y)

        # Continue animation
        self.after(30, self._animate)
