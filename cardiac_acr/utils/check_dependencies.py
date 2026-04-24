"""Environment checks for runtime dependencies."""

import importlib.util
import os

from cardiac_acr import config as cg
from cardiac_acr.preprocessing.openslide_compat import (
    OpenSlideDependencyError,
    check_openslide_runtime,
)


INFERENCE_PYTHON_DEPENDENCIES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "torch": "torch",
    "torchvision": "torchvision",
}


def _format_package_install_command(package_name):
    """Return a short pip install hint for a dependency."""
    return f"pip install {package_name}"


def format_missing_python_dependency(module_name):
    """Convert an import name into a user-facing installation hint."""
    package_name = INFERENCE_PYTHON_DEPENDENCIES.get(module_name, module_name)
    return (
        f"The Python module `{module_name}` is not installed. "
        f"Install the package with `{_format_package_install_command(package_name)}`."
    )


def find_missing_python_dependencies():
    """Return missing Python dependencies for the inference pipeline."""
    missing = []
    for module_name, package_name in INFERENCE_PYTHON_DEPENDENCIES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    return missing


def check_font_file():
    """Return a short status string for the required font asset."""
    if os.path.isfile(cg.FONT_PATH):
        return f"Found font file at {cg.FONT_PATH}"
    return (
        f"Missing expected font file at {cg.FONT_PATH}. "
        "Tile summary rendering may fail until a .ttf file is placed there."
    )


def main():
    """Validate runtime dependencies and print a short summary."""
    missing = find_missing_python_dependencies()
    if missing:
        lines = ["Python dependency check failed."]
        for module_name, package_name in missing:
            lines.append(
                f"- Missing `{module_name}` "
                f"(install with `{_format_package_install_command(package_name)}`)"
            )
        raise SystemExit("\n".join(lines))

    try:
        status = check_openslide_runtime()
    except OpenSlideDependencyError as exc:
        raise SystemExit(
            "OpenSlide dependency check failed.\n"
            f"{exc}"
        ) from exc

    print("Dependency check passed.")
    print(f"Platform: {status['platform']}")
    print(f"openslide-python: {status['openslide_python_version']}")
    print(f"OpenSlide setup: {status['windows_bin_status']}")
    print(f"Font file: {check_font_file()}")


if __name__ == "__main__":
    main()
