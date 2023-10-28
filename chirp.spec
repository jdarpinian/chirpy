# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files,collect_submodules,copy_metadata

datas = [('third_party/StyleTTS2', 'StyleTTS2')]
datas += collect_data_files('torchvision', include_py_files=True)
datas += collect_data_files('torchaudio', include_py_files=True)
datas += collect_data_files('librosa', include_py_files=True)
datas += collect_data_files('einops_exts', include_py_files=True)
datas += collect_data_files('language_tags', include_py_files=True)
# recursive is overkill, probably only a couple of dependencies need this, but who cares really
datas += copy_metadata('transformers', recursive=True)

# TODO: these may not be necessary?
hiddenimports = []
hiddenimports += collect_submodules('pyaudio')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('torchaudio.lib.libtorchaudio')

a = Analysis(
    ['chirp.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    contents_directory='chirp_data',
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
