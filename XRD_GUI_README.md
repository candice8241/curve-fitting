# XRD Data Post-Processing Suite

A comprehensive GUI application for X-ray diffraction data processing, featuring azimuthal integration and other XRD analysis tools.

## Features

### 🎯 Azimuthal Integration Module (Radial)
- **Single Sector Integration**: Process diffraction data over a specified azimuthal angle range
- **Multiple Sectors Integration**:
  - Preset templates (Quadrants, Octants, Hemispheres)
  - Custom sector definitions
- **Flexible Output**: Export to XY, CHI, DAT formats with optional CSV compilation
- **Batch Processing**: Process multiple HDF5 files automatically
- **Visual Progress**: Cute sheep animation progress indicator

### ⚗️ Powder XRD Module
- Coming soon!

### 💎 Single Crystal Module
- Coming soon!

## Installation

### Requirements
```bash
pip install numpy pandas h5py pyFAI hdf5plugin
```

### Optional Dependencies
```bash
pip install fabio  # For reading EDF and TIFF mask files
```

## Usage

### Running the Application

**Main Application (with all modules):**
```bash
python xrd_gui_main.py
```

**Standalone Azimuthal Integration:**
```bash
python radial_module.py
```

### Azimuthal Integration Workflow

1. **Select Files**:
   - PONI calibration file (from pyFAI calibration)
   - Mask file (optional, supports .npy, .edf, .tif, .h5)
   - Input HDF5 files pattern (e.g., `data/*.h5`)
   - Output directory

2. **Configure Integration**:
   - Number of points (default: 4000)
   - Unit (2th_deg, q_A^-1, q_nm^-1, r_mm)
   - Dataset path in HDF5 (default: `entry/data/data`)

3. **Define Sectors**:

   **Single Sector Mode**:
   - Start angle (0-360°)
   - End angle (0-360°)
   - Sector label

   **Multiple Sectors Mode**:
   - **Preset Templates**:
     - Quadrants: 4 sectors (0-90°, 90-180°, 180-270°, 270-360°)
     - Octants: 8 sectors (45° each)
     - Hemispheres: 2 sectors (0-180°, 180-360°)
   - **Custom Sectors**: Define your own angular ranges

4. **Run Integration**: Click the "Run Azimuthal Integration" button

5. **Output**:
   - Individual XY files for each input file and sector
   - Optional: Merged CSV file with all results

## Azimuthal Angle Convention

```
     90° (↑)
      |
180°--+--0° (→)
      |
    270° (↓)
```

Angles are measured counter-clockwise from the right horizontal axis.

## File Structure

```
curve-fitting/
├── xrd_gui_main.py           # Main application entry point
├── prefernce.py              # Base GUI classes and components
├── radial_module.py          # Azimuthal integration module
├── powder_module.py          # Powder XRD module (placeholder)
├── single_crystal_module.py  # Single crystal module (placeholder)
└── XRD_GUI_README.md        # This file
```

## Module Architecture

The application follows a modular design:

- **GUIBase**: Base class providing shared styling and utilities
- **ModernButton/ModernTab**: Custom styled UI components
- **CuteSheepProgressBar**: Animated progress indicator
- **Module Classes**: Each processing module (Powder, Radial, Single Crystal) follows the same pattern:
  - `__init__(parent, root)`: Initialize with parent frame and root window
  - `setup_ui()`: Build the module UI (called when tab is activated)

### Adding New Modules

To add a new module:

1. Create a new file `my_module.py`
2. Inherit from `GUIBase`
3. Implement `__init__(parent, root)` and `setup_ui()` methods
4. Import and add to `xrd_gui_main.py`

Example:
```python
from prefernce import GUIBase

class MyModule(GUIBase):
    def __init__(self, parent, root):
        super().__init__()
        self.parent = parent
        self.root = root

    def setup_ui(self):
        # Build your UI here
        pass
```

## Author

**Candice Wang**
Email: candicewang928@gmail.com
Created: November 15, 2025

## License

Apache License 2.0

## Acknowledgments

- pyFAI library for azimuthal integration
- h5py and hdf5plugin for HDF5 file handling
