#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the integrated powder XRD module with interactive peak fitting
"""

import tkinter as tk
from powder_xrd_module_with_interactive_fitting import PowderXRDModule


def main():
    """Main function to test the integrated module"""

    # Create root window
    root = tk.Tk()
    root.title("Powder XRD Analysis - Test")
    root.geometry("1200x900")
    root.configure(bg='#F0E6FA')

    # Create main container frame
    main_container = tk.Frame(root, bg='#F0E6FA')
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create the powder XRD module
    powder_module = PowderXRDModule(main_container, root)
    powder_module.setup_ui()

    # Add a header
    header = tk.Label(
        root,
        text="🔬 Powder XRD Analysis Suite - With Interactive Fitting 🔬",
        font=('Comic Sans MS', 16, 'bold'),
        bg='#C8A2D9',
        fg='#FFFFFF',
        pady=10
    )
    header.pack(side=tk.TOP, fill=tk.X, before=main_container)

    # Instructions label
    instructions = tk.Label(
        root,
        text="Click '✨ Interactive Fitting' button in the Integration module to open the enhanced peak fitting GUI",
        font=('Comic Sans MS', 9),
        bg='#F0E6FA',
        fg='#6B4C7A',
        pady=5
    )
    instructions.pack(side=tk.BOTTOM, fill=tk.X)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Start the application
    print("="*60)
    print("Powder XRD Analysis Suite - Test Application")
    print("="*60)
    print("Features:")
    print("  • 1D Integration & Peak Fitting")
    print("  • ✨ NEW: Interactive Peak Fitting GUI")
    print("  • Phase Transition Analysis")
    print("  • Volume Calculation")
    print("  • Birch-Murnaghan EOS Fitting")
    print("="*60)
    print("\nClick the '✨ Interactive Fitting' button to open")
    print("the enhanced peak fitting GUI in a new window!")
    print("="*60)

    root.mainloop()


if __name__ == "__main__":
    main()
