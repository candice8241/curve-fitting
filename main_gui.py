# -*- coding: utf-8 -*-
"""
Main GUI Application
XRD Data Post-Processing Suite - Entry point and main window
"""

import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
import ctypes
import os

from gui_base import GUIBase
from batch_appearance import ModernTab
from powder_module import PowderXRDModule
from radial_module import RadialXRDModule
from single_crystal_module import SingleCrystalModule


class XRDProcessingGUI(GUIBase):
    """Main GUI application for XRD data processing"""

    def __init__(self, root):
        """
        Initialize main GUI

        Args:
            root: Tk root window
        """
        super().__init__()
        self.root = root
        self.root.title("XRD Data Post-Processing")
        self.root.geometry("1100x950")
        self.root.resizable(True, True)

        # Try to set icon
        try:
            icon_path = r'D:\HEPS\ID31\dioptas_data\github_felicity\batch\ChatGPT Image.ico'
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"无法加载图标: {e}")

        self.root.configure(bg=self.colors['bg'])

        # Initialize modules
        self.powder_module = None
        self.radial_module = None
        self.single_crystal_module = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup main user interface"""
        # Header section
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(header_content, text="🐱", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Complete Suite",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 20, 'bold')).pack(side=tk.LEFT)

        # Tab bar
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        tab_frame.pack(fill=tk.X, padx=30, pady=(15, 0))

        tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        tabs_container.pack(side=tk.LEFT)

        self.powder_tab = ModernTab(tabs_container, "Powder XRD",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 15))

        self.single_tab = ModernTab(tabs_container, "Single Crystal",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=15)

        self.radial_tab = ModernTab(tabs_container, "Radial",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT, padx=15)

        # Scrollable container setup
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

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas

        # Show powder tab by default
        self.switch_tab("powder")

    def switch_tab(self, tab_name):
        """
        Switch between main tabs

        Args:
            tab_name: Name of tab to switch to ('powder', 'single', 'radial')
        """
        # Update tab active states
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Load appropriate module
        if tab_name == "powder":
            if self.powder_module is None:
                self.powder_module = PowderXRDModule(self.scrollable_frame, self.root)
            self.powder_module.setup_ui()

        elif tab_name == "radial":
            if self.radial_module is None:
                self.radial_module = RadialXRDModule(self.scrollable_frame, self.root)
            self.radial_module.setup_ui()

        elif tab_name == "single":
            if self.single_crystal_module is None:
                self.single_crystal_module = SingleCrystalModule(self.scrollable_frame, self.root)
            self.single_crystal_module.setup_ui()


def launch_main_app():
    """Launch the main application window"""
    # Set AppUserModelID (important for taskbar icon on Windows)
    try:
        app_id = u"mycompany.myapp.xrdpostprocessor"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except:
        pass  # Not on Windows or other error

    # Create main window
    root = tk.Tk()

    # Set window icon
    icon_path = r"D:\HEPS\ID31\dioptas_data\github_felicity\batch\ChatGPT Image.ico"
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Failed to set icon: {e}")

    # Set window title
    root.title("XRD Data Post-Processing")

    # Set window size and allow resizing
    root.geometry("700x400")
    root.resizable(True, True)

    # Set background color
    root.configure(bg="#EDE9F3")

    # Define cute font style
    cute_font = tkFont.Font(family="Comic Sans MS", size=14, weight="bold")

    # Create welcome label
    welcome_text = ("💜 Hey there, crystal cutie! Ready to sparkle your XRD data? 🌈\n"
                    "\n"
                    "\n"
                    "📧 Contact: candicewang928@egmail.com")
    label = tk.Label(
        root,
        text=welcome_text,
        font=cute_font,
        bg="#F9EBF2",
        fg="#8E24AA",
        pady=90
    )
    label.pack(pady=40)

    root.mainloop()


def show_startup_window():
    """Show startup splash screen before launching main app"""
    splash = tk.Tk()
    splash.title("Loading...")

    # Set startup window size and position
    splash.geometry("300x100")
    tk.Label(splash, text="Starting up, please wait...").pack(pady=20)

    # Automatically close startup window and open the main app
    splash.after(100, lambda: [splash.destroy(), launch_main_app()])
    splash.mainloop()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
