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
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.font',
        'tkinter.messagebox',
        'theme_module',
        'powder_module',
        'radial_module',
        'single_crystal_module',
        # Add any other modules that might not be automatically detected
        # Uncomment the following if your modules use these libraries:
        # 'numpy',
        # 'scipy',
        # 'matplotlib',
        # 'PIL',
        # 'pandas',
    ],
    hookspath=[],
    runtime_hooks=[],
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
