#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main GUI Application
XRD Data Post-Processing Suite - Entry point and main window
"""

import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
import os
import sys
import ctypes
from theme_module import GUIBase, ModernButton, ModernTab, CuteSheepProgressBar
from powder_module import PowderXRDModule
from radial_module import AzimuthalIntegrationModule
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

        # Try to set icon (Windows only)
        try:
            icon_path = r'D:\HEPS\ID31\dioptas_data\github_felicity\batch\\HP_full_package\ChatGPT Image.ico'
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not load icon: {e}")

        self.root.configure(bg=self.colors['bg'])

        # Initialize module containers
        self.module_frames = {}
        self.modules = {}
        self.current_tab = None

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

        tk.Label(header_content, text="✨", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Data Post_Processing",
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

        self.single_tab = ModernTab(tabs_container, "Single Crystal XRD",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=15)

        self.radial_tab = ModernTab(tabs_container, "Radial XRD",
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

        # Pre-create all module frames (hidden initially)
        self._initialize_all_modules()

        # Show powder tab by default
        self.switch_tab("powder")

    def _initialize_all_modules(self):
        """Pre-create all module frames to avoid recreation flickering"""
        # Create frames for each module
        for module_name in ["powder", "radial", "single"]:
            frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
            self.module_frames[module_name] = frame

    def _get_or_create_module(self, tab_name):
        """
        Get existing module or create new one if needed

        Args:
            tab_name: Name of the module ('powder', 'single', 'radial')

        Returns:
            Module instance
        """
        if tab_name in self.modules:
            return self.modules[tab_name]

        # Create module based on tab name
        parent_frame = self.module_frames[tab_name]

        if tab_name == "powder":
            module = PowderXRDModule(parent_frame, self.root)
        elif tab_name == "radial":
            module = AzimuthalIntegrationModule(parent_frame, self.root)
        elif tab_name == "single":
            module = SingleCrystalModule(parent_frame, self.root)
        else:
            return None

        # Store the module
        self.modules[tab_name] = module

        # Setup UI only once
        module.setup_ui()

        return module

    def switch_tab(self, tab_name):
        """
        Switch between main tabs (optimized to reduce flickering)

        Args:
            tab_name: Name of tab to switch to ('powder', 'single', 'radial')
        """
        # If already on this tab, do nothing
        if self.current_tab == tab_name:
            return

        # Temporarily disable canvas updates to reduce flickering
        self.root.update_idletasks()

        # Update tab active states
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        # Hide current module frame
        if self.current_tab and self.current_tab in self.module_frames:
            self.module_frames[self.current_tab].pack_forget()

        # Get or create the target module
        self._get_or_create_module(tab_name)

        # Show the new module frame
        if tab_name in self.module_frames:
            self.module_frames[tab_name].pack(fill=tk.BOTH, expand=True)

        # Update current tab
        self.current_tab = tab_name

        # Reset scroll position to top
        self.canvas.yview_moveto(0)

        # Force update to ensure smooth transition
        self.root.update_idletasks()


def launch_main_app():
    """Launch the main application window"""
    # Set AppUserModelID (important for taskbar icon on Windows)
    try:
        app_id = u"mycompany.myapp.xrdpostprocessor"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Not on Windows or failed

    # Create main window
    root = tk.Tk()

    # Set window icon (taskbar + title bar)
    icon_path = r"D:\HEPS\ID31\dioptas_data\github_felicity\batch\\HP_full_package\ChatGPT Image.ico"
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Failed to set icon: {e}")
    else:
        print("Icon file not found!")

    # Set window title
    root.title("XRD Data Post-Processing")

    # Set window size and allow resizing
    root.geometry("700x400")
    root.resizable(True, True)  # width & height resizable

    # Set background color to purple-pink
    root.configure(bg="#EDE9F3")  # light purple-pink (Thistle)

    # Define cute font style
    cute_font = tkFont.Font(family="Comic Sans MS", size=14, weight="bold")

    # Create adorable welcome label
    welcome_text = ("💜 Hey there, crystal cutie! Ready to sparkle your XRD data? 🌈\n"
                    "\n"
                    #"💜 Beam me up, XRD Commander – it's fitting time! ~ 💖✨\n"
                    "\n"
                    "📧 Contact: lixd@ihep.ac.cn\n\n fzhang@ihep.ac.cn\n\n candicewang928@gmail.com")
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
    """Show a brief startup splash screen"""
    splash = tk.Tk()
    splash.title("Loading...")
    splash.overrideredirect(True)  # Remove window decorations for splash

    # Center the splash screen
    window_width = 300
    window_height = 100
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    splash.geometry(f"{window_width}x{window_height}+{x}+{y}")
    splash.configure(bg="#EDE9F3")

    # Loading label
    tk.Label(
        splash,
        text="Starting up, please wait...",
        bg="#EDE9F3",
        fg="#8E24AA",
        font=("Comic Sans MS", 12)
    ).pack(pady=20)

    # Progress indicator
    progress = ttk.Progressbar(splash, mode='indeterminate', length=200)
    progress.pack(pady=10)
    progress.start(10)

    # Automatically close startup window and open the main app
    splash.after(1000, lambda: [splash.destroy(), launch_main_app()])
    splash.mainloop()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # Use startup window for better UX
    show_startup_window()
    # Or directly launch main app
    # main()
