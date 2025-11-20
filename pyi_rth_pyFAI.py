"""
PyInstaller Runtime Hook for pyFAI
This hook runs at startup to force import all pyFAI extension modules.
This solves the dynamic import issue that causes:
  ImportError: cannot import name 'splitPixelFullCSC' from 'pyFAI.integrator.load_engines'
"""

import sys

# Force import all critical pyFAI extension modules at startup
# This must happen BEFORE any code tries to use pyFAI
try:
    # Import all the extension modules that pyFAI dynamically loads
    import pyFAI.ext.splitPixelFullCSC
    import pyFAI.ext.splitPixelFullCSR
    import pyFAI.ext.splitPixelFullLUT
    import pyFAI.ext.splitPixelFull
    import pyFAI.ext.splitPixel
    import pyFAI.ext.splitBBox
    import pyFAI.ext.splitBBoxLUT
    import pyFAI.ext.splitBBoxCSR
    import pyFAI.ext.splitBBoxCSC
    import pyFAI.ext.histogram
    import pyFAI.ext.preproc

    # Also import engine modules
    import pyFAI.engines.CSR_engine
    import pyFAI.engines.CSC_engine
    import pyFAI.engines.histogram_engine

    # Import the load_engines module and inject the modules into it
    import pyFAI.integrator.load_engines as load_engines

    # Make sure the modules are accessible from load_engines
    load_engines.splitPixelFullCSC = pyFAI.ext.splitPixelFullCSC
    load_engines.splitPixelFullCSR = pyFAI.ext.splitPixelFullCSR
    load_engines.splitPixelFullLUT = pyFAI.ext.splitPixelFullLUT
    load_engines.splitPixelFull = pyFAI.ext.splitPixelFull

    print("[PyInstaller Runtime Hook] Successfully pre-loaded pyFAI extension modules")

except ImportError as e:
    print(f"[PyInstaller Runtime Hook] Warning: Could not pre-load some pyFAI modules: {e}")
    # Don't fail here - let the application try to continue
