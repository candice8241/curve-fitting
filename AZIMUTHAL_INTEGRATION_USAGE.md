# Azimuthal Integration CLI - Usage Guide

## Overview

This script performs azimuthal integration on X-ray diffraction (XRD) data using pyFAI 2025.3.0. It processes multiple HDF5 files and exports integrated data in .xy and .csv formats.

## Features

✨ **New Error Handling Features:**
- ✅ File existence validation before processing
- ✅ Clear, helpful error messages with troubleshooting steps
- ✅ Validation of HDF5 dataset paths
- ✅ Graceful handling of individual file errors
- ✅ Better debugging information

## Fixing the FileNotFoundError

If you encounter `FileNotFoundError: [Errno 2] No such file or directory`, follow these steps:

### Step 1: Verify File Paths

Open the script and check the paths in the `main()` function:

```python
# 1. Check if your PONI file exists
poni_file = r"D:\HEPS\ID31\test\using.poni"

# 2. Check if your input directory and files exist
input_pattern = r"D:\HEPS\ID31\test\input_dir\*.h5"

# 3. Check if mask file exists (or set to None if not using)
mask_file = r"D:\HEPS\ID31\test\use.edf"  # or None
```

### Step 2: Locate Your Files

Use Windows Explorer or command line to find your files:

```cmd
# In Windows Command Prompt, navigate to your data folder:
cd D:\HEPS\ID31\

# List all .poni files
dir /s *.poni

# List all .h5 files
dir /s *.h5

# List all .edf files (mask files)
dir /s *.edf
```

### Step 3: Update the Script

Update the paths in the script to match your actual file locations:

```python
def main():
    # Update these paths to match YOUR file locations:
    poni_file = r"D:\HEPS\ID31\calibration\sample.poni"  # ← Update this
    input_pattern = r"D:\HEPS\ID31\data\*.h5"            # ← Update this
    output_dir = r"D:\HEPS\ID31\output"                  # ← Update this
    mask_file = r"D:\HEPS\ID31\mask.edf"                 # ← Update this or set to None
```

### Step 4: Common Path Issues

**Issue 1: Wrong directory**
```python
# ❌ Wrong:
poni_file = r"D:\HEPS\ID31\test\using.poni"

# ✅ Correct (if file is actually in 'calibration' folder):
poni_file = r"D:\HEPS\ID31\calibration\using.poni"
```

**Issue 2: Missing backslash escaping**
```python
# ❌ Wrong:
poni_file = "D:\HEPS\ID31\test\using.poni"  # Missing 'r' prefix

# ✅ Correct:
poni_file = r"D:\HEPS\ID31\test\using.poni"  # Raw string with 'r' prefix
```

**Issue 3: No input files found**
```python
# ❌ Wrong pattern:
input_pattern = r"D:\HEPS\ID31\*.h5"  # No files in this directory

# ✅ Correct:
input_pattern = r"D:\HEPS\ID31\input_dir\*.h5"  # Files are in subdirectory
```

## Usage Example

### Basic Usage

```python
from azimuthal_integration_cli import process_azimuthal_integration

# Simple single-sector integration
results = process_azimuthal_integration(
    poni_file=r"C:\data\calibration.poni",
    input_pattern=r"C:\data\*.h5",
    output_dir=r"C:\output"
)
```

### Advanced Usage with Multiple Sectors

```python
# Define custom sectors
sectors = [
    (0, 90, "Q1_0-90"),
    (90, 180, "Q2_90-180"),
    (180, 270, "Q3_180-270"),
    (270, 360, "Q4_270-360")
]

results = process_azimuthal_integration(
    poni_file=r"C:\data\calibration.poni",
    input_pattern=r"C:\data\*.h5",
    output_dir=r"C:\output",
    dataset_path="entry/data/data",
    mask_file=r"C:\data\mask.edf",
    sectors=sectors,
    npt=4000,
    unit='2th_deg',
    save_csv=True
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poni_file` | str | Required | Path to PONI calibration file |
| `input_pattern` | str | Required | Glob pattern for input H5 files |
| `output_dir` | str | Required | Output directory path |
| `dataset_path` | str | `"entry/data/data"` | HDF5 dataset path |
| `mask_file` | str | `None` | Path to mask file (.edf or .npy) |
| `sectors` | list | `[(0, 90, "Sector_0-90")]` | List of (start, end, label) tuples |
| `npt` | int | `4000` | Number of integration points |
| `unit` | str | `'2th_deg'` | Unit for radial axis |
| `save_csv` | bool | `True` | Save merged CSV files |

## Supported Units

- `'2th_deg'` - Two-theta in degrees
- `'q_A^-1'` - Q in inverse Angstroms
- `'q_nm^-1'` - Q in inverse nanometers
- `'r_mm'` - Radius in millimeters

## Output Files

### Individual .xy Files
One file per input file per sector:
```
input_file_0001_Sector_0-90.xy
input_file_0002_Sector_0-90.xy
...
```

### Merged CSV Files
One CSV per sector with all data:
```
azimuthal_integration_Sector_0-90.csv
```

## Troubleshooting

### Error: "No input files found"
- Check that the glob pattern is correct
- Verify files exist in the specified directory
- Ensure file extensions match (.h5, .hdf5, etc.)

### Error: "Dataset 'entry/data/data' not found"
- Open the H5 file with HDFView or h5py to find the correct dataset path
- Update the `dataset_path` parameter

### Error: "PONI file not found"
- Follow Steps 1-3 above to locate and update the PONI file path

## Requirements

```
pyFAI >= 2025.3.0
numpy
pandas
h5py
fabio  # For .edf mask files
```

Install with:
```bash
pip install pyFAI numpy pandas h5py fabio
```

## Example Directory Structure

```
D:\HEPS\ID31\
├── calibration\
│   └── sample.poni         ← Your PONI file
├── data\
│   ├── scan_0001.h5
│   ├── scan_0002.h5
│   └── ...
├── masks\
│   └── detector_mask.edf   ← Optional mask file
└── output\                 ← Results will be saved here
    ├── scan_0001_Sector_0-90.xy
    ├── scan_0002_Sector_0-90.xy
    └── azimuthal_integration_Sector_0-90.csv
```

## Contact

For issues or questions, please refer to the pyFAI documentation:
https://pyfai.readthedocs.io/
