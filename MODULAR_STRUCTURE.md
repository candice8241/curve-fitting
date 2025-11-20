# Modular Structure Documentation

## Overview

The XRD Peak Fitting Tool has been refactored into a well-organized modular structure for better maintainability, reusability, and extensibility.

## Directory Structure

```
curve-fitting/
├── curve_fitting_script/
│   ├── __init__.py              # Package initialization with exports
│   ├── smoothing.py             # Data smoothing algorithms
│   ├── clustering.py            # DBSCAN peak clustering
│   ├── background.py            # Background fitting methods
│   ├── peak_fitting.py          # Peak profile functions (Voigt, Pseudo-Voigt)
│   ├── gui.py                   # Main GUI application class
│   ├── gui_fitting.py           # Peak fitting algorithms for GUI
│   └── curve_fitting.py         # Original batch processing script (preserved)
├── run_peak_fitting_gui.py      # Main entry point for GUI application
├── README.md                    # Project README
├── MODULAR_STRUCTURE.md         # This file
└── LICENSE                      # Apache 2.0 License
```

## Module Descriptions

### 1. `smoothing.py`
**Purpose:** Data preprocessing and noise reduction

**Functions:**
- `apply_gaussian_smoothing(y, sigma=2)` - Apply Gaussian smoothing
- `apply_savgol_smoothing(y, window_length=11, polyorder=3)` - Apply Savitzky-Golay smoothing
- `apply_smoothing(y, method='gaussian', **kwargs)` - Unified smoothing interface

**Usage Example:**
```python
from curve_fitting_script.smoothing import apply_smoothing

y_smooth = apply_smoothing(y_data, method='gaussian', sigma=3)
```

### 2. `clustering.py`
**Purpose:** Group nearby peaks using DBSCAN density clustering

**Functions:**
- `cluster_peaks_dbscan(peak_positions, eps=None, min_samples=1)` - Group peaks by proximity

**Features:**
- Automatic eps estimation based on median peak spacing
- Noise point handling (assigns to nearest cluster)
- Ideal for overlapping peak detection

**Usage Example:**
```python
from curve_fitting_script.clustering import cluster_peaks_dbscan

labels, n_clusters = cluster_peaks_dbscan(peak_positions, eps=1.5)
```

### 3. `background.py`
**Purpose:** Background estimation and subtraction

**Functions:**
- `fit_global_background(x, y, peak_indices, method='spline', ...)` - Fit smooth background
- `find_background_points_auto(x, y, n_points=10, window_size=50)` - Auto-find BG points
- `find_group_minima(x, y, peak_indices)` - Find local minima between peaks
- `create_piecewise_background(x_data, minima_points)` - Create piecewise linear background

**Methods:**
- `'spline'`: Smooth spline interpolation
- `'piecewise'`: Piecewise linear (adjacent points connected)
- `'polynomial'`: Polynomial fit with bounded curvature (order 2-5)

**Usage Example:**
```python
from curve_fitting_script.background import fit_global_background

background, bg_points = fit_global_background(
    x, y, peak_indices, method='polynomial', poly_order=3
)
```

### 4. `peak_fitting.py`
**Purpose:** Peak profile functions and parameter calculations

**Functions:**
- `pseudo_voigt(x, amplitude, center, sigma, gamma, eta)` - Pseudo-Voigt profile
- `voigt(x, amplitude, center, sigma, gamma)` - True Voigt profile (Faddeeva function)
- `calculate_fwhm(sigma, gamma, eta)` - Calculate FWHM from parameters
- `calculate_area(amplitude, sigma, gamma, eta)` - Calculate integrated area
- `estimate_fwhm_robust(x, y, peak_idx, smooth=True)` - Robust FWHM estimation

**Peak Profiles:**
- **Pseudo-Voigt**: Weighted sum of Gaussian and Lorentzian (faster, approximate)
- **Voigt**: True convolution of Gaussian and Lorentzian (slower, more accurate)

**Usage Example:**
```python
from curve_fitting_script.peak_fitting import pseudo_voigt, calculate_fwhm

y_peak = pseudo_voigt(x, amplitude=100, center=25.5, sigma=0.1, gamma=0.08, eta=0.5)
fwhm = calculate_fwhm(sigma=0.1, gamma=0.08, eta=0.5)
```

### 5. `gui.py`
**Purpose:** Main interactive GUI application

**Class:**
- `PeakFittingGUI` - Complete Tkinter-based GUI application

**Features:**
- Interactive peak selection (click to add, right-click to remove)
- Manual and automatic background selection
- Data smoothing controls
- Batch file navigation
- Real-time visualization
- Results export (CSV + PNG)

**Usage:**
```python
from curve_fitting_script.gui import main

if __name__ == "__main__":
    main()
```

### 6. `gui_fitting.py`
**Purpose:** Core fitting algorithms for the GUI

**Functions:**
- `fit_peaks_method(gui_instance)` - Main peak fitting routine with DBSCAN grouping

**Algorithm:**
1. Fit global background (manual or automatic)
2. Subtract background from data
3. Estimate FWHM for each peak
4. Group peaks using DBSCAN clustering
5. Fit each group separately with shared background
6. Extract and display results

### 7. `curve_fitting.py` (Original)
**Purpose:** Batch processing script (preserved for backwards compatibility)

**Functions:**
- `process_file(file_path, save_dir)` - Process single XY file
- `main()` - Batch process all files in a folder

## Running the Application

### GUI Application (Recommended)
```bash
python run_peak_fitting_gui.py
```

### Batch Processing (Legacy)
Edit the folder path in `curve_fitting_script/curve_fitting.py` and run:
```bash
python curve_fitting_script/curve_fitting.py
```

### As a Library
```python
# Import specific modules
from curve_fitting_script import (
    apply_smoothing,
    cluster_peaks_dbscan,
    fit_global_background,
    pseudo_voigt,
    voigt
)

# Use functions in your own script
y_smooth = apply_smoothing(y, method='gaussian', sigma=2)
background, bg_points = fit_global_background(x, y, peaks)
```

## Key Improvements in Version 2.0

1. **Modular Design**: Each functional area is in a separate module
2. **Reusability**: Functions can be imported and used independently
3. **Documentation**: Comprehensive docstrings with parameter descriptions
4. **DBSCAN Clustering**: Intelligent peak grouping for overlapping peaks
5. **Multiple Background Methods**: Spline, piecewise linear, and polynomial
6. **Enhanced GUI**: File navigation, smoothing controls, overlap mode
7. **Better Peak Detection**: Adaptive thresholds and filtering

## Dependencies

- `numpy` - Numerical operations
- `scipy` - Signal processing, optimization, special functions
- `matplotlib` - Plotting and visualization
- `pandas` - Data handling and export
- `scikit-learn` - DBSCAN clustering
- `tkinter` - GUI framework (usually included with Python)

## Future Enhancements

Potential areas for expansion:
- Support for additional file formats (CSV, Excel, HDF5)
- Peak deconvolution for highly overlapping peaks
- Automated peak identification by phase matching
- Batch processing GUI
- Export to other formats (JSON, Excel with formatting)
- Undo/redo for all operations
- Real-time fitting preview

## Contributing

To add new features:
1. Create a new module in `curve_fitting_script/` if it's a distinct functional area
2. Add exports to `__init__.py`
3. Document all functions with NumPy-style docstrings
4. Include usage examples in docstrings
5. Test with various XRD datasets

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact

**Author:** candicewang928@gmail.com

For questions, bug reports, or feature requests, please contact the author.
