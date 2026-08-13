# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Fathom Fibers desktop application (Linux onedir).

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = []
datas += collect_data_files("matplotlib")
datas += collect_data_files("skimage")
datas += collect_data_files("tifffile")
datas += collect_data_files("scipy")
datas += collect_data_files("PIL")

hiddenimports = []
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("scipy.ndimage")
hiddenimports += collect_submodules("tifffile")
hiddenimports += [
    "scipy.interpolate",
    "scipy.interpolate._bsplines",
    "scipy.spatial._ckdtree",
    "scipy.stats._stats_py",
    "scipy.optimize",
    "PIL._tkinter_finder",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]

a = Analysis(
    ["launcher.py"],
    pathex=["../src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FathomFibers",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FathomFibers",
)
