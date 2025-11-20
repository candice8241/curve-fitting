# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for XRD Data Post-Processing application
"""

import os

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

        # XRD-specific libraries (fabio only)
        # pyFAI modules commented out - uncomment if needed
        # 'pyFAI',
        # 'pyFAI.azimuthalIntegrator',
        # 'pyFAI.detectors',
        # 'pyFAI.geometryRefinement',
        # 'pyFAI.calibrant',
        # 'pyFAI.worker',
        # 'pyFAI.integrator',
        # 'pyFAI.integrator.load_engines',
        # 'pyFAI.integrator.splitPixel',
        # 'pyFAI.integrator.splitBBox',
        # 'pyFAI.integrator.splitBBoxLUT',
        # 'pyFAI.integrator.splitBBoxCSR',
        # 'pyFAI.ext',
        # 'pyFAI.ext.splitPixel',
        # 'pyFAI.ext.splitPixelFull',
        # 'pyFAI.ext.splitPixelFullLUT',
        # 'pyFAI.ext.splitPixelFullCSR',
        # 'pyFAI.ext.splitBBox',
        # 'pyFAI.ext.splitBBoxLUT',
        # 'pyFAI.ext.splitBBoxCSR',
        # 'pyFAI.ext.histogram',
        # 'pyFAI.ext.preproc',
        # 'pyFAI.engines',
        # 'pyFAI.engines.CSR_engine',
        # 'pyFAI.engines.histogram_engine',
        # 'pyFAI.method_registry',
        # 'pyFAI.io',
        # 'pyFAI.utils',
        # 'pyFAI.containers',
        # 'pyFAI.gui',

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
        'lmfit',           # Peak fitting
        'skimage',         # Image processing
        'cv2',             # OpenCV for image operations
        'openpyxl',        # Excel file support
        'xlrd',            # Excel reading
        'xlsxwriter',      # Excel writing
        'peakutils',       # Peak detection
        'silx',            # Synchrotron data analysis
        'pycairo',         # Cairo graphics
        'PyQt5',           # Qt GUI (if needed)
        'multiprocessing', # Parallel processing
        'concurrent.futures',
        'threading',
        'queue',

        # Crystallography libraries
        'pymatgen',        # Materials analysis
        'ase',             # Atomic simulation environment
        'CifFile',         # CIF file handling

        # Math and fitting
        'sympy',           # Symbolic mathematics
        'statsmodels',     # Statistical models
        'sklearn',         # Machine learning (scikit-learn)
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Note: pyFAI is needed by radial_module.py and batch_integration.py
        # Do NOT exclude it. Use --collect-all pyFAI when building
    ],
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
    console=False,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,  # Application icon (optional)
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
