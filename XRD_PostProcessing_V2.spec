# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for XRD Data Post-Processing application
VERSION 2 - With Runtime Hook for pyFAI circular import fix
"""

import os
import sys

block_cipher = None

# List of additional data files to include
added_files = [
    ('resources', 'resources'),  # Include resources folder
]

a = Analysis(
    ['main.py'],
    pathex=[r'D:\HEPS\ID31\dioptas_data\github_felicity\batch\HP_full_package'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # Tkinter GUI modules
        'tkinter',
        'tkinter.ttk',
        'tkinter.font',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'tkinter.scrolledtext',

        # Custom application modules
        'theme_module',
        'powder_module',
        'radial_module',
        'single_crystal_module',

        # Core scientific computing libraries
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'scipy',
        'scipy.optimize',
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.integrate',
        'scipy.stats',
        'pandas',

        # Data visualization
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',

        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'PIL.ImageFont',

        # Data formats
        'h5py',
        'json',
        'csv',

        # ============ CRITICAL: pyFAI modules ============
        # Core pyFAI modules
        'pyFAI',
        'pyFAI.azimuthalIntegrator',
        'pyFAI.detectors',
        'pyFAI.geometryRefinement',
        'pyFAI.calibrant',
        'pyFAI.worker',

        # Integrator modules
        'pyFAI.integrator',
        'pyFAI.integrator.load_engines',
        'pyFAI.integrator.azimuthal',
        'pyFAI.integrator.common',
        'pyFAI.integrator.splitPixel',
        'pyFAI.integrator.splitBBox',
        'pyFAI.integrator.splitBBoxLUT',
        'pyFAI.integrator.splitBBoxCSR',

        # Extension modules (C/Cython compiled) - CRITICAL!
        'pyFAI.ext',
        'pyFAI.ext.splitPixel',
        'pyFAI.ext.splitPixelFull',
        'pyFAI.ext.splitPixelFullLUT',
        'pyFAI.ext.splitPixelFullCSR',
        'pyFAI.ext.splitPixelFullCSC',
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

        # Engine modules
        'pyFAI.engines',
        'pyFAI.engines.CSR_engine',
        'pyFAI.engines.CSC_engine',
        'pyFAI.engines.histogram_engine',
        'pyFAI.engines.preproc',

        # Additional pyFAI modules
        'pyFAI.method_registry',
        'pyFAI.io',
        'pyFAI.io.image',
        'pyFAI.io.ponifile',
        'pyFAI.io.nexus',
        'pyFAI.utils',
        'pyFAI.utils.decorators',
        'pyFAI.utils.mathutil',
        'pyFAI.utils.shell',
        'pyFAI.containers',
        'pyFAI.gui',
        'pyFAI.opencl',
        'pyFAI.resources',

        # Fabio image format support
        'fabio',
        'fabio.adscimage',
        'fabio.edfimage',
        'fabio.cbfimage',
        'fabio.mar345image',
        'fabio.pilatusimage',
        'fabio.pixiimage',
        'fabio.mpaimage',
        'fabio.tifimage',
        'fabio.bruker100image',
        'fabio.brukerimage',
        'fabio.geimage',
        'fabio.imgcimage',
        'fabio.dm3image',
        'fabio.xsdimage',
        'fabio.fit2dimage',
        'fabio.raxisimage',
        'fabio.dtrekimage',
        'fabio.hdf5image',
        'fabio.nexusimage',

        # Progress bar and utilities
        'tqdm',

        # Additional potentially needed modules
        'lmfit',
        'skimage',
        'cv2',
        'openpyxl',
        'xlrd',
        'xlsxwriter',
        'peakutils',
        'silx',
        'silx.io',
        'silx.gui',
        'pycairo',
        'PyQt5',
        'multiprocessing',
        'concurrent.futures',
        'threading',
        'queue',

        # Crystallography libraries
        'pymatgen',
        'ase',
        'CifFile',

        # Math and fitting
        'sympy',
        'statsmodels',
        'sklearn',
    ],
    hookspath=[],
    runtime_hooks=['pyi_rth_pyFAI.py'],  # <-- CRITICAL: Runtime hook to pre-load pyFAI modules
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Check if icon file exists
icon_path = 'resources/app_icon.ico' if os.path.exists('resources/app_icon.ico') else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='XRD_PostProcessing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='XRD_PostProcessing',
)
