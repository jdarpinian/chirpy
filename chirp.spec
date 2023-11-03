# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files,collect_submodules,copy_metadata

datas = [('third_party/StyleTTS2', 'StyleTTS2'),
         ('/src/models/nltk_data', 'nltk_data'),
         ('/src/models/exllama2/OpenHermes-2-Mistral-7B-5.0bpw-h6-exl2', 'OpenHermes-2-Mistral-7B-5.0bpw-h6-exl2'),
         ('/src/models/models--guillaumekln--faster-whisper-base.en', 'models--guillaumekln--faster-whisper-base.en'),
         ('/src/models/eSpeak', 'eSpeak'),
         ('/src/models/dlls', '.'),]
datas += collect_data_files('torchvision', include_py_files=True)
datas += collect_data_files('torchaudio', include_py_files=True)
datas += collect_data_files('librosa', include_py_files=True)
datas += collect_data_files('einops_exts', include_py_files=True)
datas += collect_data_files('language_tags', include_py_files=True)
datas += collect_data_files('cuda-python', include_py_files=True)
datas += collect_data_files('cuda-nvrtc-dev', include_py_files=True)
# recursive is overkill, probably only a couple of dependencies need this, but who cares really
datas += copy_metadata('transformers', recursive=True)

# TODO: these may not be necessary?
hiddenimports = []
hiddenimports += collect_submodules('pyaudio')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('torchaudio.lib.libtorchaudio')
hiddenimports += collect_submodules('cuda-python')
hiddenimports += collect_submodules('cuda-nvrtc-dev')

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

# Delete .git directories, we don't need them
to_keep = []
for (dest, source, kind) in a.datas:
    if '/.git/' not in dest and '\\.git\\' not in dest:
        to_keep.append((dest, source, kind))
a.datas = to_keep

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
    manifest="""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0" xmlns:asmv3="urn:schemas-microsoft-com:asm.v3">
  <asmv3:application>
    <asmv3:windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </asmv3:windowsSettings>
  </asmv3:application>
</assembly>""",
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
