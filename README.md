# curve-fitting
A Python toolkit for X-ray diffraction (XRD) data analysis including peak fitting and azimuthal integration. Supports Voigt and pseudo‑Voigt profiles for peak fitting, and batch processing of diffraction data with pyFAI integration. Suitable for materials science and XRD data analysis.

# 🔬 XRD Data Analysis Toolkit

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![SciPy](https://img.shields.io/badge/powered%20by-SciPy-orange)](https://scipy.org/)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4-pink)]()

## 📦 Tools Included

### 1. Peak Fitting Tool (`curve_fitting.py`)
Automatic **peak detection and fitting** of 1D X-ray diffraction (XRD) data exported from **Dioptas** in `.xy` format.
Supports both **Voigt** and **Pseudo-Voigt** profiles for flexible fitting.

- ✅ Batch process all `.xy` files in a folder
- ✅ Customize peak detection sensitivity and fitting window
- ✅ Save high-quality plots + `.csv` per file
- ✅ Final `all_results.csv` summarizes everything for you

### 2. Azimuthal Integration CLI (`azimuthal_integration_cli.py`)
Batch processing of 2D XRD data using **pyFAI 2025.3.0** for azimuthal integration.
Process multiple HDF5 files with sector-based integration.

- ✅ Multi-sector azimuthal integration (quadrants, octants, custom)
- ✅ Batch process HDF5 files with calibration (PONI files)
- ✅ Support for detector masks (.edf, .npy)
- ✅ Export to .xy and merged .csv formats
- ✅ Comprehensive error handling and validation
- ✅ See [AZIMUTHAL_INTEGRATION_USAGE.md](AZIMUTHAL_INTEGRATION_USAGE.md) for detailed usage

---

> **Author:** [candicewang928@gmail.com](mailto:candicewang928@gmail.com)
> **Created on:** Nov 6, 2025
