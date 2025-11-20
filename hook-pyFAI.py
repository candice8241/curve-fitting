"""
PyInstaller hook for pyFAI library
This hook ensures all pyFAI extension modules and dependencies are properly collected.
Place this file in your PyInstaller hooks directory or specify it with --additional-hooks-dir
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Collect all submodules from pyFAI
hiddenimports = collect_submodules('pyFAI')

# Explicitly add critical extension modules that might be missed
critical_modules = [
    'pyFAI.ext.splitPixelFullCSC',
    'pyFAI.ext.splitPixelFullCSR',
    'pyFAI.ext.splitPixelFullLUT',
    'pyFAI.ext.splitPixelFull',
    'pyFAI.ext.splitPixel',
    'pyFAI.ext.splitBBox',
    'pyFAI.ext.splitBBoxLUT',
    'pyFAI.ext.splitBBoxCSR',
    'pyFAI.ext.splitBBoxCSC',
    'pyFAI.ext.histogram',
    'pyFAI.ext.preproc',
    'pyFAI.ext.bilinear',
    'pyFAI.ext.reconstruct',
    'pyFAI.ext.relabel',
    'pyFAI.ext.watershed',
    'pyFAI.ext.morphology',
    'pyFAI.ext.fastcrc',
    'pyFAI.ext.invert_geometry',
    'pyFAI.ext.sparse_utils',
    'pyFAI.engines.CSR_engine',
    'pyFAI.engines.CSC_engine',
    'pyFAI.engines.histogram_engine',
    'pyFAI.engines.preproc',
    'pyFAI.integrator.load_engines',
    'pyFAI.integrator.azimuthal',
    'pyFAI.integrator.common',
]

for module in critical_modules:
    if module not in hiddenimports:
        hiddenimports.append(module)

# Collect data files (calibration files, OpenCL kernels, etc.)
datas, binaries, _ = collect_all('pyFAI')

print(f"[pyFAI Hook] Collected {len(hiddenimports)} hidden imports")
print(f"[pyFAI Hook] Collected {len(datas)} data files")
print(f"[pyFAI Hook] Collected {len(binaries)} binary files")
