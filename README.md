# curve-fitting
A simple Python utility for detecting and fitting diffraction peaks using Voigt and pseudo‑Voigt profiles. Reads two‑column .xy files, automatically finds peaks, fits each peak with a chosen profile, and outputs high‑resolution plots and CSV summaries. Suitable for materials science and XRD data analysis.
# 🔬 Peak Fitting Tool for Dioptas `.xy` Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![SciPy](https://img.shields.io/badge/powered%20by-SciPy-orange)](https://scipy.org/)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4-pink)]()

This script is designed for automatic **peak detection and fitting** of 1D X-ray diffraction (XRD) data exported from **Dioptas** in `.xy` format.  
It supports both **Voigt** and **Pseudo-Voigt** profiles for flexible fitting, and outputs individual peak plots and `.csv` results for each dataset.

- ✅ Batch process all `.xy` files in a folder  
- ✅ Customize peak detection sensitivity and fitting window  
- ✅ Save high-quality plots + `.csv` per file  
- ✅ Final `all_results.csv` summarizes everything for you

---

> **Author:** [candicewang928@gmail.com](mailto:candicewang928@gmail.com)  
> **Created on:** Nov 6, 2025  
