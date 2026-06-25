"""Global settings: TdxQuant (tqcenter) path resolution + project paths.

The tqcenter API ships inside the 通达信 install dir at ``PYPlugins/user``.
Resolve it once here so every entry script can do::

    from config.settings import ensure_tqcenter_on_path
    ensure_tqcenter_on_path()
    from tqcenter import tq        # now importable
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- Project layout ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RENKO_DIR = DATA_DIR / "renko"
RESULTS_DIR = PROJECT_ROOT / "results"
PARAMS_FILE = PROJECT_ROOT / "config" / "params.yaml"

# --- TdxQuant (tqcenter) resolution ----------------------------------------
# Manual override — set this if registry autodetect fails (e.g. WSL dev box).
# Example: r"C:/new_tdx_test"
TDX_ROOT_OVERRIDE: str | None = r"C:\new_tdx_test"


def _autodetect_tdx_root() -> str | None:
    """Read the 通达信 install dir from the Windows registry, or None."""
    try:
        import winreg
    except ImportError:
        return None  # non-Windows — rely on TDX_ROOT_OVERRIDE
    key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\通达信金融终端64"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            root, _ = winreg.QueryValueEx(key, "InstallLocation")
            return root
    except OSError:
        return None


def ensure_tqcenter_on_path() -> Path:
    """Insert ``PYPlugins/user`` at the FRONT of sys.path so tqcenter imports.

    Must be called before ``from tqcenter import tq``. Uses ``insert(0, ...)``
    (not append) per tqcenter's requirement to win over same-named modules.
    """
    root = TDX_ROOT_OVERRIDE or _autodetect_tdx_root()
    if root is None:
        raise RuntimeError(
            "Could not locate the 通达信 install dir. "
            "Set TDX_ROOT_OVERRIDE in config/settings.py."
        )
    plugin_dir = Path(root) / "PYPlugins" / "user"
    if not plugin_dir.exists():
        raise FileNotFoundError(f"tqcenter dir not found: {plugin_dir}")
    p = str(plugin_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return plugin_dir


def load_params() -> dict:
    """Load config/params.yaml as a plain dict."""
    import yaml

    with open(PARAMS_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
