# PyInstaller pyFAI Circular Import Fix Guide

## Problem Summary
You encountered this error when building your XRD application with PyInstaller:
```
ImportError: cannot import name 'splitPixelFullCSC' from 'pyFAI.integrator.load_engines'
```

## Root Cause
This is a **circular import issue** caused by PyInstaller not properly detecting and collecting pyFAI's compiled extension modules (`.pyd`/`.so` files). The `splitPixelFullCSC` function is in the `pyFAI.ext.splitPixelFullCSC` module, which wasn't being included in your build.

## The Solution

### Changes Made to the Spec File

1. **Uncommented all pyFAI hidden imports** - All the commented-out pyFAI modules in your original spec file have been uncommented and expanded.

2. **Added critical extension modules** - The most important addition is:
   - `'pyFAI.ext.splitPixelFullCSC'` ← **This was the missing module causing your error!**
   - Plus many other `pyFAI.ext.*` modules that are compiled C/Cython extensions

3. **Added engine modules**:
   - `'pyFAI.engines.CSR_engine'`
   - `'pyFAI.engines.CSC_engine'`
   - `'pyFAI.engines.histogram_engine'`

4. **Added integrator modules**:
   - `'pyFAI.integrator.load_engines'`
   - `'pyFAI.integrator.azimuthal'`
   - `'pyFAI.integrator.common'`

## How to Build Your Application

### Method 1: Using the Fixed Spec File (Recommended)

```bash
# Copy the fixed spec file to your project directory
# Then build with:
pyinstaller XRD_PostProcessing_FIXED.spec --clean
```

### Method 2: Using the PyInstaller Hook + Spec File (Most Robust)

```bash
# 1. Create a 'hooks' directory in your project
mkdir hooks

# 2. Copy hook-pyFAI.py to the hooks directory
copy hook-pyFAI.py hooks/

# 3. Build using the spec file with the hook directory
pyinstaller XRD_PostProcessing_FIXED.spec --additional-hooks-dir=hooks --clean
```

### Method 3: Using --collect-all Flag (Alternative)

If you prefer to use your original spec file approach, you can use:

```bash
pyinstaller your_spec_file.spec --collect-all pyFAI --clean
```

**However**, this approach may include unnecessary files and increase build size. The fixed spec file approach is more targeted.

## Verification Steps

After building, test your application by:

1. Navigate to the `dist/XRD_PostProcessing/` directory
2. Run `XRD_PostProcessing.exe`
3. Try to use the radial integration features that use pyFAI
4. Check that no import errors occur

## Additional Troubleshooting

If you still encounter issues:

### Enable Debug Mode
Temporarily set `console=True` in the spec file to see error messages:
```python
exe = EXE(
    # ... other parameters ...
    console=True,  # Change from False to True
)
```

### Check for Missing DLLs
If you get DLL errors on Windows, you may need to include additional binaries:
```python
binaries=[
    # Add any missing DLLs here
],
```

### Verify pyFAI Installation
Ensure pyFAI is properly installed in your Python environment:
```bash
python -c "import pyFAI; print(pyFAI.__version__)"
python -c "from pyFAI.ext import splitPixelFullCSC; print('Success!')"
```

## Understanding the Circular Import Issue

The import chain that was failing:
```
main.py (line 15)
  → radial_module.py (line 22)
    → pyFAI.integrator.azimuthal (line 42)
      → pyFAI.integrator.common (line 53)
        → pyFAI.integrator.load_engines
          → [FAILED] pyFAI.ext.splitPixelFullCSC (missing!)
```

PyInstaller's static analysis couldn't detect the `splitPixelFullCSC` module because:
1. It's a compiled extension module (not pure Python)
2. It's imported dynamically by pyFAI's engine loading system
3. The import happens through string-based module loading, which PyInstaller can't trace

## Files Provided

1. **XRD_PostProcessing_FIXED.spec** - Your spec file with all pyFAI imports properly configured
2. **hook-pyFAI.py** - PyInstaller hook that automatically collects all pyFAI modules
3. **PYINSTALLER_FIX_GUIDE.md** - This guide

## Best Practices for Future Builds

1. **Always use `--clean`** when rebuilding to avoid cached issues
2. **Keep the hook file** for easier builds
3. **Test thoroughly** after building, especially pyFAI-dependent features
4. **Version control** your spec file and hooks

## Questions?

If you continue to experience issues:
- Check PyInstaller version: `pyinstaller --version` (recommend 5.0+)
- Check pyFAI version: `python -c "import pyFAI; print(pyFAI.__version__)"`
- Ensure all dependencies are installed in the same Python environment
- Try building in a clean virtual environment

---

**Summary**: The fix adds `pyFAI.ext.splitPixelFullCSC` and related extension modules to the hidden imports. This resolves the circular import error you were experiencing.
