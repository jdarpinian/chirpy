# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('torchvision', include_py_files=True)
datas += collect_data_files('torchaudio', include_py_files=True)
datas += collect_data_files('TTS', include_py_files=True)
datas += collect_data_files('trainer', include_py_files=True)
datas += collect_data_files('gruut', include_py_files=True)
datas += collect_data_files('gruut_lang_en', include_py_files=True)
datas += collect_data_files('jamo', include_py_files=True)
datas += collect_data_files('unidic_lite', include_py_files=True)
datas += collect_data_files('pycrfsuite', include_py_files=True)
datas += collect_data_files('librosa', include_py_files=True)


a = Analysis(
    ['chirp.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='chirp',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='chirp',
)
