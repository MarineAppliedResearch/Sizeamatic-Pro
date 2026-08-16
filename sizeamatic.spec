# PyInstaller build spec for Sizeamatic Pro (ROADMAP.md Phase 9).
#
# Produces a single-file, onefile Windows .exe with a Tk root window icon
# and an embedded Windows version resource. No PyInstaller bootloader
# --splash is used here - main.py has its own Tk-based startup splash
# (_show_startup_splash) that shows identically whether run from source
# or from this .exe, so a separate bootloader splash would just show
# twice in a row for no benefit (tried it; it did exactly that).
#
# Build with:
#   uv run pyinstaller sizeamatic.spec
#
# See AGENTS.md's Packaging section for the full build/verify workflow.

# A .spec file is executed as a plain Python script by pyinstaller, so
# the icon conversion happens right here rather than needing a separate
# manual step: assets/default-icon.png is the one source-of-truth art
# file the project owner drops into the repo, and every build
# regenerates assets/icon.ico from it via create_app_icon.py. That means
# icon.ico is a build artifact (gitignored), not something to keep
# manually in sync with default-icon.png.
#
# SPECPATH (this spec file's own directory) is injected by pyinstaller
# into the spec's exec namespace, but isn't on sys.path by default - add
# it so a plain top-level import of a repo-root module works the same
# way it would running `python create_app_icon.py` directly.
import sys
import tomllib

sys.path.insert(0, SPECPATH)
import build_version_info
import create_app_icon

create_app_icon.build_icon("assets/default-icon.png", "assets/icon.ico")

# The .exe's own Windows version resource (Explorer's file Properties
# dialog) - read from pyproject.toml, the one place the version number
# lives, rather than a separately maintained constant.
with open("pyproject.toml", "rb") as f:
    APP_VERSION = tomllib.load(f)["project"]["version"]
build_version_info.write_version_file(APP_VERSION, "version_info.txt")

a = Analysis(
    ["main.py"],
    pathex=[],
    # assets/icon.ico is bundled as a data file (not just embedded as the
    # .exe's own icon below) because main.py's resource_path("assets/
    # icon.ico") reads it again at runtime, to set the Tk window's own
    # icon via root.iconbitmap() - both need the same file, for two
    # different purposes (the .exe's file icon vs. the window's icon).
    # assets/splash-pro.png and pyproject.toml are bundled because
    # main.py's own startup splash (_show_startup_splash/get_app_version)
    # reads both again at runtime via resource_path, inside the bundle
    # just as it would from the source tree.
    datas=[
        ("assets/icon.ico", "assets"),
        ("assets/splash-pro.png", "assets"),
        ("pyproject.toml", "."),
    ],
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
    a.binaries,
    a.datas,
    [],
    name=f"SizeamaticPro-v{APP_VERSION}",
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
    icon="assets/icon.ico",
    version="version_info.txt",
    onefile=True,
)
