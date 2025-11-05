# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('util', 'util'), ('C:\\Users\\mark_\\anaconda3\\envs\\DocExtractor\\Lib\\site-packages\\tkinterdnd2\\tkdnd', 'tkinterdnd2/tkdnd')]
binaries = []
hiddenimports = ['cv2', 'PIL._tkinter_finder', 'PIL.Image', 'PIL.ImageTk', 'numpy', 'pandas', 'platformdirs']
hiddenimports += collect_submodules('pearl_edit')
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fitz')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['D:\\Programs\\PearlEdit\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PearlEdit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\Programs\\PearlEdit\\util\\icons\\PearlEdit.png'],
)
