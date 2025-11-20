#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for XRD Peak Fitting GUI Application

This script launches the interactive peak fitting GUI.

Usage:
    python run_peak_fitting_gui.py

Or on Unix-like systems with executable permission:
    ./run_peak_fitting_gui.py

@author: candicewang928@gmail.com
"""

import sys
import os

# Add the parent directory to the path to import curve_fitting_script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curve_fitting_script.gui import main

if __name__ == "__main__":
    print("=" * 60)
    print("XRD Peak Fitting Tool - Interactive GUI")
    print("Version 2.0.0")
    print("Author: candicewang928@gmail.com")
    print("=" * 60)
    print("\nLaunching GUI application...")
    print("Please use the 'Load File' button to begin.\n")

    main()
