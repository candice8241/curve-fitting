# XRD Data Post-Processing GUI Application

A comprehensive graphical user interface for X-Ray Diffraction (XRD) data analysis, including integration, peak fitting, volume calculation, and equation of state fitting.

## Features

### 1. **1D Integration & Peak Fitting**
- Batch integration of 2D diffraction patterns to 1D profiles
- Automatic peak detection and fitting
- Support for Voigt and Pseudo-Voigt fitting methods
- Background subtraction with linear interpolation
- Batch processing of multiple files

### 2. **Volume Calculation & Phase Analysis**
- Calculate unit cell volumes from peak positions
- Support for multiple crystal systems (FCC, BCC, SC, Hexagonal, etc.)
- Automatic phase transition detection
- Separate original and new peaks
- Generate pressure-volume curves

### 3. **Birch-Murnaghan Equation of State**
- Fit P-V data to 2nd or 3rd order Birch-Murnaghan EOS
- Calculate bulk modulus (K₀) and its pressure derivative (K₀')
- Generate fitted curves and residual plots
- Comprehensive uncertainty analysis

## Installation

### Prerequisites

1. **Python 3.7 or higher**

2. **Install Required Packages:**
```bash
cd curve_fitting_script
pip install -r requirements.txt
```

3. **Install Tkinter (for GUI):**

   On Ubuntu/Debian:
   ```bash
   sudo apt-get install python3-tk
   ```

   On macOS (usually included):
   ```bash
   # Should be already installed with Python
   ```

   On Windows:
   ```bash
   # Tkinter is included with Python installation
   ```

4. **Optional Dependencies (for full functionality):**
```bash
# For 2D integration
pip install pyFAI fabio

# For HDF5 support
pip install h5py
```

## Usage

### Starting the Application

```bash
cd curve_fitting_script
python xrd_gui.py
```

### Module 1: 1D Integration & Peak Fitting

#### Integration Settings:
1. **PONI File**: Calibration file from pyFAI
2. **Mask File**: Optional mask file (.edf format)
3. **Input Pattern**: File pattern for batch processing (e.g., `data/*.h5`)
4. **Output Directory**: Where to save integrated data
5. **Dataset Path**: HDF5 dataset path (default: `entry/data/data`)
6. **Number of Points**: Resolution of integrated pattern (default: 4000)
7. **Unit**: Integration unit (2θ, q, etc.)

#### Peak Fitting Settings:
1. **Fitting Method**: Choose between `pseudo` (Pseudo-Voigt) or `voigt`
2. Click **Run Fitting** after integration to fit peaks

#### Buttons:
- **Run Integration**: Process 2D patterns to 1D
- **Run Fitting**: Fit peaks in integrated data
- **Full Pipeline**: Run both integration and fitting sequentially

### Module 2: Cal_Volume & BM_Fitting

#### Volume Calculation:
1. **Input CSV (Peak Data)**: CSV file with peak positions at different pressures
2. **Crystal System**: Select your crystal system (FCC, BCC, etc.)
3. **Wavelength**: X-ray wavelength in Angstroms (default: 0.4133 Å)
4. **Output Directory**: Where to save results

**Peak Separation:**
- Click **Separate Original & New Peaks** to identify phase transitions
- Adjust tolerances (Tol-1, Tol-2, Tol-3) for peak matching
- Set N (minimum pressure points for tracking)

#### Birch-Murnaghan Fitting:
1. **Input CSV**: Pressure-volume data (2 columns: P, V)
2. **Output Directory**: Where to save results
3. **BM Order**: Choose 2nd or 3rd order equation
4. Click **Birch-Murnaghan Fit**

## File Formats

### Input Files

#### Peak Data CSV Format:
```csv
File, Pressure, Peak Index, Peak #, Center, Amplitude, Sigma, Gamma, Eta
file1.xy, 0.0, 0, 1, 12.345, 1000, 0.1, 0.1, 0.5
file1.xy, 0.0, 1, 2, 14.567, 800, 0.12, 0.11, 0.48
...
```

#### Pressure-Volume CSV Format:
```csv
Pressure (GPa), Volume (Å³)
0.0, 100.5
5.0, 95.3
10.0, 90.8
...
```

### Output Files

#### Integration Output:
- `.xy` files: 2-column text files (2θ, Intensity)

#### Fitting Output:
- `*_fit.png`: Peak fitting plots
- `*_results.csv`: Fitted parameters for each peak
- `all_results.csv`: Combined results for all files

#### Volume Calculation Output:
- `pressure_volume_results.csv`: P-V data
- `pressure_volume_plot.png`: P-V curve
- `original_peaks.csv`: Original phase peaks
- `new_peaks.csv`: New phase peaks (if detected)

#### BM Fitting Output:
- `bm_parameters.csv`: Fitted parameters (V₀, K₀, K₀')
- `bm_fitted_curve.csv`: Fitted P-V curve
- `bm_fit_plot.png`: Fit and residuals plot

## Module Descriptions

### 1. `batch_appearance.py`
Custom GUI components with modern styling:
- `ModernButton`: Styled buttons with hover effects
- `ModernTab`: Tab navigation buttons
- `CuteSheepProgressBar`: Animated progress indicator

### 2. `batch_integration.py`
2D to 1D integration using pyFAI:
- `BatchIntegrator`: Handles batch integration of diffraction patterns
- Supports HDF5, EDF, and other image formats
- Configurable integration parameters

### 3. `peak_fitting.py`
Peak fitting with Voigt profiles:
- `BatchFitter`: Batch peak detection and fitting
- Automatic peak detection with filtering
- Background subtraction
- Multiple fitting methods

### 4. `batch_cal_volume.py`
Volume calculation and phase analysis:
- `XRayDiffractionAnalyzer`: Calculate unit cell volumes
- Phase transition detection
- Peak tracking across pressure range
- Support for multiple crystal systems

### 5. `birch_murnaghan_batch.py`
Equation of state fitting:
- `BirchMurnaghanFitter`: 2nd and 3rd order BM EOS
- Automatic initial guess generation
- Uncertainty quantification
- Comprehensive output

## Example Workflow

### Complete Analysis Pipeline:

1. **Integrate 2D Patterns:**
   - Load PONI and mask files
   - Select input pattern
   - Run integration

2. **Fit Peaks:**
   - Choose fitting method
   - Run peak fitting
   - Review results in `fit_output/all_results.csv`

3. **Calculate Volumes:**
   - Load peak fitting results CSV
   - Select crystal system
   - Run volume calculation

4. **Fit Equation of State:**
   - Load pressure-volume CSV
   - Choose BM order
   - Run fitting
   - Review parameters in `bm_parameters.csv`

## Troubleshooting

### Common Issues:

1. **"No module named 'tkinter'"**
   - Install python3-tk package (see Installation section)

2. **"pyFAI not installed"**
   - Integration will run in simulation mode
   - Install pyFAI for real integration: `pip install pyFAI`

3. **"No files found"**
   - Check input pattern syntax
   - Ensure files exist in specified directory

4. **Fitting fails**
   - Adjust peak detection parameters
   - Check data quality
   - Try different fitting method

5. **BM fitting gives unrealistic values**
   - Check input data units (GPa for pressure, Å³ for volume)
   - Ensure data is sorted by pressure
   - Try different initial guess

## Tips for Best Results

1. **Peak Fitting:**
   - Use larger window size for broader peaks
   - Adjust distance parameter if peaks are missed
   - Pseudo-Voigt generally works better for most XRD data

2. **Volume Calculation:**
   - Ensure correct wavelength value
   - Choose appropriate crystal system
   - Verify peak assignments

3. **BM Fitting:**
   - Include data points across wide pressure range
   - 3rd order BM is more flexible but may overfit with few points
   - Check residuals plot for systematic deviations

## Citation

If you use this software in your research, please cite:

```
XRD Data Post-Processing Suite
Author: [Your Name]
Year: 2025
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please contact: candicewang928@gmail.com

## Acknowledgments

- pyFAI for 2D integration capabilities
- SciPy for optimization algorithms
- Matplotlib for visualization
