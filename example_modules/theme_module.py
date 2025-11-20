# -*- coding: utf-8 -*-
"""
示例主题模块 - 用于测试打包
如果您已有完整的theme_module.py，请使用您自己的文件
"""

import tkinter as tk


class GUIBase:
    """GUI基类，提供颜色主题"""

    def __init__(self):
        self.colors = {
            'bg': '#F5F5F5',
            'card_bg': '#FFFFFF',
            'text_dark': '#333333',
            'text_light': '#666666',
            'primary': '#8E24AA',
            'secondary': '#EDE9F3',
            'accent': '#F9EBF2'
        }


class ModernButton(tk.Button):
    """现代化按钮组件"""

    def __init__(self, parent, text, command=None, **kwargs):
        default_style = {
            'font': ('Arial', 10),
            'bg': '#8E24AA',
            'fg': 'white',
            'relief': 'flat',
            'padx': 20,
            'pady': 8,
            'cursor': 'hand2'
        }
        default_style.update(kwargs)
        super().__init__(parent, text=text, command=command, **default_style)

        # 悬停效果
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, e):
        self['bg'] = '#9C27B0'

    def _on_leave(self, e):
        self['bg'] = '#8E24AA'


class ModernTab(tk.Frame):
    """现代化标签页组件"""

    def __init__(self, parent, text, command, is_active=False):
        super().__init__(parent, bg='#F5F5F5')

        self.is_active = is_active
        self.command = command
        self.text = text

        # 创建标签按钮
        self.button = tk.Button(
            self,
            text=text,
            command=self._on_click,
            font=('Arial', 11, 'bold' if is_active else 'normal'),
            bg='#8E24AA' if is_active else '#E0E0E0',
            fg='white' if is_active else '#666666',
            relief='flat',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        self.button.pack()

        # 活动指示器
        if is_active:
            indicator = tk.Frame(self, bg='#8E24AA', height=3)
            indicator.pack(fill=tk.X)

    def _on_click(self):
        if self.command:
            self.command()

    def set_active(self, active):
        """设置标签页活动状态"""
        self.is_active = active

        if active:
            self.button.config(
                font=('Arial', 11, 'bold'),
                bg='#8E24AA',
                fg='white'
            )
        else:
            self.button.config(
                font=('Arial', 11),
                bg='#E0E0E0',
                fg='#666666'
            )


class CuteSheepProgressBar(tk.Frame):
    """可爱的进度条组件"""

    def __init__(self, parent):
        super().__init__(parent, bg='#F5F5F5')

        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)

        # 标签
        self.label = tk.Label(
            self,
            text="🐑 Processing...",
            font=('Arial', 10),
            bg='#F5F5F5',
            fg='#8E24AA'
        )
        self.label.pack(pady=5)

        # 进度条
        self.canvas = tk.Canvas(self, height=20, bg='#E0E0E0', highlightthickness=0)
        self.canvas.pack(fill=tk.X, padx=10)

        self.progress_rect = self.canvas.create_rectangle(
            0, 0, 0, 20,
            fill='#8E24AA',
            outline=''
        )

    def set_progress(self, value):
        """设置进度（0-100）"""
        self.progress_var.set(value)
        width = self.canvas.winfo_width()
        progress_width = (value / 100) * width
        self.canvas.coords(self.progress_rect, 0, 0, progress_width, 20)
        self.update_idletasks()

    def set_text(self, text):
        """设置进度条文本"""
        self.label.config(text=text)
