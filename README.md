# curve-fitting
A comprehensive Python toolkit for X-ray diffraction (XRD) data analysis, including peak fitting, phase transition detection, and crystal system identification.

# 🔬 XRD Peak Fitting & Phase Transition Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![SciPy](https://img.shields.io/badge/powered%20by-SciPy-orange)](https://scipy.org/)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4-pink)]()

## 📦 Two Powerful Tools in One Package

### 1. Peak Fitting Tool (`curve_fitting.py`)
Automatic **peak detection and fitting** of 1D X-ray diffraction (XRD) data exported from **Dioptas** in `.xy` format.
Supports both **Voigt** and **Pseudo-Voigt** profiles for flexible fitting.

- ✅ Batch process all `.xy` files in a folder
- ✅ Customize peak detection sensitivity and fitting window
- ✅ Save high-quality plots + `.csv` per file
- ✅ Final `all_results.csv` summarizes everything for you

### 2. Phase Transition Analysis Tool (`phase_transition_analysis.py`) 🆕
Intelligent **phase transition detection** and **crystal system identification** from high-pressure XRD data.

- ✅ Automatic phase transition point detection
- ✅ New peak identification and tracking
- ✅ Crystal system determination (Cubic, Hexagonal, Tetragonal, Orthorhombic, Monoclinic, Triclinic)
- ✅ Unit cell parameter calculation
- ✅ JSON output with detailed analysis results

---

> **Author:** [candicewang928@gmail.com](mailto:candicewang928@gmail.com)
> **Created on:** Nov 6, 2025
> **Last Updated:** Nov 12, 2025

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/curve-fitting.git
cd curve-fitting/curve_fitting_script
pip install -r requirements.txt
```

### Tool 1: Peak Fitting

```bash
# Edit the folder path in curve_fitting.py
python curve_fitting.py
```

### Tool 2: Phase Transition Analysis

```bash
# Run with your CSV file
python phase_transition_analysis.py your_data.csv

# Or try the quick start example
python quick_start.py
```

---

## 📖 Documentation

### Peak Fitting Tool

See the built-in comments in `curve_fitting.py` for detailed configuration options.

**Key Features:**
- Reads `.xy` files from Dioptas
- Automatic peak detection with customizable sensitivity
- Voigt and Pseudo-Voigt profile fitting
- Background subtraction (linear interpolation)
- High-quality matplotlib plots
- CSV output for each file and combined results

### Phase Transition Analysis Tool

See **[PHASE_ANALYSIS_README.md](curve_fitting_script/PHASE_ANALYSIS_README.md)** for comprehensive documentation.

**Input Format:**
- CSV file with `File` column (pressure in GPa) and `Center` column (2θ peak positions)
- Different pressure points separated by blank rows

**Key Parameters (configurable in code):**
```python
WAVELENGTH = 0.6199           # X-ray wavelength (Å)
PEAK_TOLERANCE_1 = 0.3        # Phase transition detection tolerance
PEAK_TOLERANCE_2 = 0.2        # New peak counting tolerance
PEAK_TOLERANCE_3 = 0.15       # Peak tracking tolerance
N_PRESSURE_POINTS = 4         # Points for stability determination
```

**Output:**
- Detailed console output with analysis steps
- JSON file with complete analysis results
- Crystal system identification with unit cell parameters
- Phase transition pressure point

**Example Output:**
```
>>> 发现相变点：15.80 GPa

相变前晶系：Face-Centered Cubic (FCC)
晶胞参数：{'a': 4.0521}

压力 18.20 GPa:
  新相晶系：Hexagonal (HCP)
  晶胞参数：{'a': 2.9345, 'c': 4.6823, 'c/a': 1.5952}
```

---

## 📊 Supported Crystal Systems

The phase transition analysis tool supports **7 crystal systems** with automatic hkl indexing:

| Crystal System | Min. Peaks Required | Parameters |
|----------------|---------------------|------------|
| Cubic (FCC/BCC/SC) | 1 | a |
| Hexagonal (HCP) | 2 | a, c |
| Tetragonal | 2 | a, c |
| Orthorhombic | 3 | a, b, c |
| Monoclinic | 4 | a, b, c, β |
| Triclinic | 6 | a, b, c, α, β, γ |

---

## 🔬 Scientific Background

### Bragg's Law
```
nλ = 2d sinθ
```

### d-spacing Formulas

**Cubic:**
```
1/d² = (h² + k² + l²) / a²
```

**Hexagonal:**
```
1/d² = 4/3 · (h² + hk + k²) / a² + l² / c²
```

**Tetragonal:**
```
1/d² = (h² + k²) / a² + l² / c²
```

**Orthorhombic:**
```
1/d² = h²/a² + k²/b² + l²/c²
```

---

## 📁 Repository Structure

```
curve-fitting/
├── curve_fitting_script/
│   ├── curve_fitting.py              # Peak fitting tool
│   ├── phase_transition_analysis.py  # Phase transition analysis tool
│   ├── quick_start.py                # Quick start example
│   ├── example_peaks.csv             # Example data
│   ├── requirements.txt              # Python dependencies
│   ├── PHASE_ANALYSIS_README.md      # Detailed documentation
├── README.md                         # This file
└── LICENSE                           # Apache 2.0 License
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 📮 Contact

For questions, suggestions, or collaboration opportunities:
- Email: candicewang928@gmail.com
- GitHub Issues: [Submit an issue](https://github.com/yourusername/curve-fitting/issues)

---

## 🙏 Acknowledgments

This project uses the following open-source libraries:
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing and optimization
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Matplotlib](https://matplotlib.org/) - Data visualization

References:
- Cullity, B. D., & Stock, S. R. (2001). *Elements of X-ray Diffraction* (3rd ed.). Prentice Hall.
- Warren, B. E. (1990). *X-ray Diffraction*. Dover Publications.

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
