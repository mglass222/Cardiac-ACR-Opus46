import os
import platform
import sys

from cardiac_acr import config as cg


class OpenSlideDependencyError(RuntimeError):
    """Raised when the OpenSlide Python package or native library is unavailable."""


def _current_platform():
    """Return a user-facing platform label."""
    if sys.platform.startswith("win"):
        return "Windows"
    if sys.platform == "darwin":
        return "macOS"
    return platform.system() or "Linux"


def _windows_bin_status():
    """Describe the configured Windows DLL directory."""
    openslide_path = cg.OPENSLIDE_BIN_PATH
    if not openslide_path:
        return "OPENSLIDE_BIN_PATH is not set."
    if not os.path.isdir(openslide_path):
        return (
            f"OPENSLIDE_BIN_PATH points to a missing directory: {openslide_path!r}."
        )
    return f"OPENSLIDE_BIN_PATH is set to: {openslide_path!r}."


def _install_instructions():
    """Return platform-specific install guidance for OpenSlide."""
    system = _current_platform()
    if system == "Windows":
        return (
            "Install with `pip install openslide-python`, download the OpenSlide "
            "Windows binaries, and set OPENSLIDE_BIN_PATH to the extracted `bin` "
            "directory. "
            + _windows_bin_status()
        )
    if system == "macOS":
        return (
            "Install with `brew install openslide` and "
            "`pip install openslide-python`."
        )
    return (
        "Install the system library with your package manager "
        "(for example `apt install libopenslide0 openslide-tools` on Debian/Ubuntu) "
        "and install the Python package with `pip install openslide-python`."
    )


def _configure_windows_dll_path():
    """Register the OpenSlide DLL directory on Windows when configured."""
    openslide_path = cg.OPENSLIDE_BIN_PATH
    if not hasattr(os, "add_dll_directory"):
        return
    if not openslide_path:
        return
    if not os.path.isdir(openslide_path):
        raise OpenSlideDependencyError(
            "OpenSlide could not be configured on Windows because "
            f"OPENSLIDE_BIN_PATH points to a missing directory: {openslide_path!r}. "
            "Update the environment variable to the folder containing "
            "`libopenslide-0.dll` and related DLLs."
        )
    try:
        os.add_dll_directory(openslide_path)
    except OSError as exc:
        raise OpenSlideDependencyError(
            "OpenSlide could not register the Windows DLL directory. "
            f"Path: {openslide_path!r}. Original error: {exc}. "
            + _install_instructions()
        ) from exc


def _load_openslide():
    """Import OpenSlide and raise actionable errors when dependencies are missing."""
    _configure_windows_dll_path()

    try:
        import openslide as openslide_module
        from openslide import OpenSlideError as openslide_error
    except ModuleNotFoundError as exc:
        if exc.name != "openslide":
            raise
        raise OpenSlideDependencyError(
            "The Python package `openslide-python` is not installed. "
            + _install_instructions()
        ) from exc
    except OSError as exc:
        raise OpenSlideDependencyError(
            "The OpenSlide native library could not be loaded. "
            f"Platform: {_current_platform()}. Original error: {exc}. "
            + _install_instructions()
        ) from exc

    return openslide_module, openslide_error


def check_openslide_runtime():
    """Return a short dependency summary or raise an actionable error."""
    openslide_module, _ = _load_openslide()
    return {
        "platform": _current_platform(),
        "openslide_python_version": getattr(openslide_module, "__version__", "unknown"),
        "windows_bin_status": _windows_bin_status()
        if _current_platform() == "Windows"
        else "Not required on this platform.",
    }


def __getattr__(name):
    """Lazily expose OpenSlide symbols so callers get actionable errors."""
    if name == "openslide":
        openslide_module, _ = _load_openslide()
        return openslide_module
    if name == "OpenSlideError":
        _, openslide_error = _load_openslide()
        return openslide_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
