# BMFitter Class - Encapsulated BM Equation Fitting

## Overview

This is an object-oriented, encapsulated version of the Birch-Murnaghan equation fitting module. It provides a clean, flexible API for fitting P-V data with both 2nd and 3rd order BM equations.

## Features

✅ **Object-Oriented Design**: Clean class-based interface
✅ **Flexible Parameter Control**: Customize bounds, initial guesses
✅ **Single & Multi-Phase**: Support for one or multiple phases
✅ **Automatic Comparison**: Built-in fit quality comparison
✅ **Easy Plotting**: One-line plotting functions
✅ **Data Export**: Save parameters to CSV
✅ **Type Hints**: Full type annotations for better IDE support
✅ **Error Handling**: Robust error checking and validation

## Quick Start

### Installation

```python
# No installation needed - just import the module
from bm_fitting_class import BMFitter, BMMultiPhaseFitter, quick_fit
```

### Simplest Usage

```python
import numpy as np
from bm_fitting_class import quick_fit

# Your data
V_data = np.array([16.5, 16.0, 15.5, 15.0, 14.5, 14.0])
P_data = np.array([0.5, 3.0, 6.5, 11.0, 16.5, 23.0])

# One function call does everything!
fitter = quick_fit(V_data, P_data,
                   phase_name="My Phase",
                   save_dir="./output")
```

This will:
- Fit both 2nd and 3rd order BM equations
- Generate and save P-V curve plots
- Generate and save residual plots
- Save fitting parameters to CSV
- Compare fits and recommend which to use

## Core Classes

### 1. BMFitter

Main class for fitting a single phase.

#### Basic Usage

```python
from bm_fitting_class import BMFitter

# Create fitter
fitter = BMFitter(V_data, P_data, phase_name="Original Phase")

# Fit both orders
fitter.fit_both_orders()

# Compare results
fitter.compare_fits()

# Plot
fitter.plot_pv_curve(save_path="pv_curve.png")
fitter.plot_residuals(save_path="residuals.png")

# Get parameters
df = fitter.get_parameters()
print(df)

# Save parameters
fitter.save_parameters("parameters.csv")
```

#### Advanced Usage - Custom Bounds

```python
# Fit with custom parameter bounds
fitter.fit_3rd_order(
    V0_bounds=(14.0, 17.0),          # Custom V0 range
    B0_bounds=(100, 300),             # Custom B0 range
    B0_prime_bounds=(3.5, 5.0),       # Custom B0' range
    initial_guess=(16.5, 150, 4.0)    # Custom initial guess
)
```

### 2. BMMultiPhaseFitter

Class for fitting multiple phases and comparing them.

```python
from bm_fitting_class import BMMultiPhaseFitter

# Create multi-phase fitter
multi_fitter = BMMultiPhaseFitter()

# Add phases
multi_fitter.add_phase("Original Phase", V_orig, P_orig)
multi_fitter.add_phase("New Phase", V_new, P_new)

# Fit all
multi_fitter.fit_all()

# Plot comparison
multi_fitter.plot_comparison(save_path="comparison.png", order=3)

# Get all parameters
df_all = multi_fitter.get_all_parameters()

# Save
multi_fitter.save_all_parameters("all_parameters.csv")
```

## API Reference

### BMFitter Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `__init__(V, P, name)` | Initialize fitter | BMFitter |
| `fit_2nd_order(...)` | Fit 2nd order BM | dict |
| `fit_3rd_order(...)` | Fit 3rd order BM | dict |
| `fit_both_orders()` | Fit both orders | tuple |
| `plot_pv_curve(...)` | Plot P-V curve | Figure |
| `plot_residuals(...)` | Plot residuals | Figure |
| `get_parameters()` | Get params as DataFrame | DataFrame |
| `save_parameters(path)` | Save params to CSV | None |
| `compare_fits()` | Compare 2nd vs 3rd | str |

### Static Methods

```python
# Can be used independently
P = BMFitter.birch_murnaghan_2nd(V, V0, B0)
P = BMFitter.birch_murnaghan_3rd(V, V0, B0, B0_prime)
```

## Usage Examples

### Example 1: Load from CSV and Fit

```python
import pandas as pd
from bm_fitting_class import BMFitter

# Load data
df = pd.read_csv("my_data.csv")
V_data = df['V_atomic'].values
P_data = df['Pressure (GPa)'].values

# Fit
fitter = BMFitter(V_data, P_data, phase_name="My Phase")
fitter.fit_both_orders()

# Get recommendation
recommendation = fitter.compare_fits()

# Save everything
fitter.plot_pv_curve(save_path="output/pv_curve.png")
fitter.plot_residuals(save_path="output/residuals.png")
fitter.save_parameters("output/parameters.csv")
```

### Example 2: Programmatic Access to Results

```python
# Fit
fitter = BMFitter(V_data, P_data, phase_name="Test")
fitter.fit_both_orders()

# Access results directly
if fitter.results_3rd and fitter.results_3rd['success']:
    V0 = fitter.results_3rd['V0']
    B0 = fitter.results_3rd['B0']
    B0_prime = fitter.results_3rd['B0_prime']
    R_squared = fitter.results_3rd['R_squared']

    print(f"V₀ = {V0:.4f} Å³/atom")
    print(f"B₀ = {B0:.2f} GPa")
    print(f"B₀' = {B0_prime:.3f}")
    print(f"R² = {R_squared:.6f}")
```

### Example 3: Batch Processing

```python
from bm_fitting_class import BMFitter
import glob

# Process multiple CSV files
csv_files = glob.glob("data/*.csv")

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    phase_name = csv_file.split('/')[-1].replace('.csv', '')

    fitter = BMFitter(df['V_atomic'].values,
                     df['Pressure (GPa)'].values,
                     phase_name=phase_name)

    fitter.fit_both_orders(verbose=False)
    fitter.save_parameters(f"output/{phase_name}_params.csv")
    fitter.plot_pv_curve(save_path=f"output/{phase_name}_pv.png")
```

### Example 4: Customize Plot Appearance

```python
# Create fitter and fit
fitter = BMFitter(V_data, P_data, phase_name="Custom Plot")
fitter.fit_both_orders()

# Plot with custom settings
fig = fitter.plot_pv_curve(
    save_path="custom_plot.png",
    show_2nd=True,      # Show 2nd order fit
    show_3rd=True,      # Show 3rd order fit
    figsize=(14, 10),   # Custom figure size
    dpi=600             # High resolution
)
```

## Real-World Workflow

Here's a complete workflow for analyzing your experimental data:

```python
import pandas as pd
import os
from bm_fitting_class import BMFitter, BMMultiPhaseFitter

# Set up paths
data_dir = "D:/HEPS/ID31/dioptas_data/Al0"
output_dir = os.path.join(data_dir, "BM_fitting_output")
os.makedirs(output_dir, exist_ok=True)

# Load original phase data
df_orig = pd.read_csv(os.path.join(data_dir, "all_results_original_peaks_lattice.csv"))
V_orig = df_orig['V_atomic'].dropna().values
P_orig = df_orig['Pressure (GPa)'].dropna().values

# Load new phase data
df_new = pd.read_csv(os.path.join(data_dir, "all_results_new_peaks_lattice.csv"))
V_new = df_new['V_atomic'].dropna().values
P_new = df_new['Pressure (GPa)'].dropna().values

# Fit original phase
print("Fitting Original Phase...")
fitter_orig = BMFitter(V_orig, P_orig, phase_name="Original Phase")
fitter_orig.fit_both_orders()
fitter_orig.compare_fits()
fitter_orig.plot_pv_curve(save_path=os.path.join(output_dir, "original_pv.png"))
fitter_orig.plot_residuals(save_path=os.path.join(output_dir, "original_res.png"))
fitter_orig.save_parameters(os.path.join(output_dir, "original_params.csv"))

# Fit new phase
print("\nFitting New Phase...")
fitter_new = BMFitter(V_new, P_new, phase_name="New Phase")
fitter_new.fit_both_orders()
fitter_new.compare_fits()
fitter_new.plot_pv_curve(save_path=os.path.join(output_dir, "new_pv.png"))
fitter_new.plot_residuals(save_path=os.path.join(output_dir, "new_res.png"))
fitter_new.save_parameters(os.path.join(output_dir, "new_params.csv"))

# Combined comparison
print("\nCreating Combined Comparison...")
multi_fitter = BMMultiPhaseFitter()
multi_fitter.add_phase("Original Phase", V_orig, P_orig)
multi_fitter.add_phase("New Phase", V_new, P_new)
multi_fitter.fit_all(verbose=False)
multi_fitter.plot_comparison(save_path=os.path.join(output_dir, "comparison.png"))
multi_fitter.save_all_parameters(os.path.join(output_dir, "all_params.csv"))

print(f"\n✅ All results saved to: {output_dir}")
```

## Advantages of This Encapsulated Version

| Feature | Original Script | Encapsulated Class |
|---------|----------------|-------------------|
| **Reusability** | Low - need to copy functions | High - import and use |
| **Flexibility** | Fixed parameters | Customizable everything |
| **Code Organization** | All in main() | Clean methods |
| **Testing** | Difficult | Easy to unit test |
| **Documentation** | Comments | Docstrings + type hints |
| **Error Handling** | Basic | Robust validation |
| **API** | Function calls | Object methods |
| **State Management** | Manual | Automatic |
| **Extensibility** | Modify source | Inherit and extend |

## When to Use Each Version

### Use Original Script (`bm_fitting.py`) when:
- You have exactly 2 phases to compare
- You want a simple, self-contained script
- You're running a one-time analysis
- You prefer a procedural workflow

### Use Encapsulated Class (`bm_fitting_class.py`) when:
- You need to fit many phases
- You want programmatic control
- You're building a larger analysis pipeline
- You need to customize fitting parameters
- You want to reuse the code in other projects
- You need better error handling

## Performance Comparison

Both versions have identical computational performance - they use the same underlying algorithms. The encapsulated version adds negligible overhead (~0.1ms) for object creation.

## Dependencies

```
numpy
pandas
matplotlib
scipy
```

Install with:
```bash
pip install numpy pandas matplotlib scipy
```

## Tips and Best Practices

1. **Data Quality**: Ensure your data has at least 5-6 points for reliable fitting
2. **Bounds Selection**: Use reasonable bounds based on your material properties
3. **Model Selection**: Use `compare_fits()` to decide between 2nd and 3rd order
4. **Parameter Access**: Results are stored in `results_2nd` and `results_3rd` dictionaries
5. **Error Handling**: Check `results['success']` before accessing fitted parameters
6. **Batch Processing**: Use loops to process multiple phases efficiently
7. **Custom Bounds**: Adjust bounds if fitting fails with default values

## Troubleshooting

**Problem**: Fitting fails with "Optimal parameters not found"
- **Solution**: Adjust bounds or initial guess values

**Problem**: Large uncertainties in B₀'
- **Solution**: Use 2nd order fit or collect more data points

**Problem**: Poor fit quality (low R²)
- **Solution**: Check data quality, remove outliers, ensure proper pressure calibration

## License

Apache License 2.0

## Author

candicewang928@gmail.com

## Version

1.0 - 2025-11-13
