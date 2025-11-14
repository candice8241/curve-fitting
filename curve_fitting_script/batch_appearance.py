# -*- coding: utf-8 -*-
"""
Custom GUI Components for XRD Processing
Provides modern-styled buttons, tabs, and progress bars
"""

import tkinter as tk
from tkinter import ttk


class ModernButton:
    """Modern styled button with hover effects"""

    def __init__(self, parent, text, command, icon="", bg_color="#B794F6",
                 hover_color="#D4BBFF", width=150, height=40):
        self.frame = tk.Frame(parent, bg=parent.cget('bg'))
        self.bg_color = bg_color
        self.hover_color = hover_color

        self.button = tk.Button(
            self.frame,
            text=f"{icon} {text}" if icon else text,
            command=command,
            font=('Comic Sans MS', 10, 'bold'),
            bg=bg_color,
            fg='white',
            activebackground=hover_color,
            relief='flat',
            borderwidth=0,
            cursor='hand2'
        )
        self.button.pack(fill=tk.BOTH, expand=True)
        self.frame.configure(width=width, height=height)

        # Bind hover effects
        self.button.bind("<Enter>", self._on_enter)
        self.button.bind("<Leave>", self._on_leave)

    def _on_enter(self, e):
        self.button.config(bg=self.hover_color)

    def _on_leave(self, e):
        self.button.config(bg=self.bg_color)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)


class ModernTab:
    """Modern styled tab button"""

    def __init__(self, parent, text, command, is_active=False):
        self.active_color = '#B794F6'
        self.inactive_color = '#E8E4F3'
        self.is_active = is_active

        self.button = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Comic Sans MS', 11, 'bold'),
            bg=self.active_color if is_active else self.inactive_color,
            fg='white' if is_active else '#8B8BA7',
            activebackground=self.active_color,
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=8,
            cursor='hand2'
        )

    def set_active(self, active):
        self.is_active = active
        if active:
            self.button.config(bg=self.active_color, fg='white')
        else:
            self.button.config(bg=self.inactive_color, fg='#8B8BA7')

    def pack(self, **kwargs):
        self.button.pack(**kwargs)


class CuteSheepProgressBar:
    """Animated progress bar with cute sheep animation"""

    def __init__(self, parent, width=600, height=60):
        self.canvas = tk.Canvas(parent, width=width, height=height,
                               bg='#F8F7FF', highlightthickness=0)
        self.width = width
        self.height = height
        self.running = False
        self.position = 0

        # Draw progress bar background
        self.bar_bg = self.canvas.create_rectangle(
            10, height//2 - 10, width - 10, height//2 + 10,
            fill='#E8E4F3', outline=''
        )

        # Progress fill
        self.bar_fill = self.canvas.create_rectangle(
            10, height//2 - 10, 10, height//2 + 10,
            fill='#B794F6', outline=''
        )

        # Sheep emoji
        self.sheep = self.canvas.create_text(
            20, height//2, text='🐑', font=('Segoe UI Emoji', 24)
        )

        # Status text
        self.status_text = self.canvas.create_text(
            width//2, height - 10, text='Ready',
            font=('Comic Sans MS', 9), fill='#8B8BA7'
        )

    def start(self):
        """Start the progress animation"""
        self.running = True
        self.position = 0
        self._animate()

    def stop(self):
        """Stop the progress animation"""
        self.running = False
        self.canvas.coords(self.bar_fill, 10, self.height//2 - 10,
                          10, self.height//2 + 10)
        self.canvas.coords(self.sheep, 20, self.height//2)
        self.canvas.itemconfig(self.status_text, text='Ready')

    def _animate(self):
        """Animate the progress bar"""
        if not self.running:
            return

        # Update position
        self.position = (self.position + 2) % (self.width - 20)

        # Update bar fill
        bar_width = min(self.position, self.width - 20)
        self.canvas.coords(self.bar_fill, 10, self.height//2 - 10,
                          10 + bar_width, self.height//2 + 10)

        # Update sheep position
        sheep_x = 20 + self.position
        self.canvas.coords(self.sheep, sheep_x, self.height//2)

        # Update status text
        status = f"Processing... {int((self.position / (self.width - 20)) * 100)}%"
        self.canvas.itemconfig(self.status_text, text=status)

        # Schedule next frame
        self.canvas.after(20, self._animate)

    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)
