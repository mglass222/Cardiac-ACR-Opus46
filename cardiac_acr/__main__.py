"""Run the Cardiac-ACR inference pipeline via ``python -m cardiac_acr``."""

from cardiac_acr.check_dependencies import format_missing_python_dependency
from cardiac_acr.openslide_compat import OpenSlideDependencyError


def run():
    """Load the inference pipeline lazily so dependency errors stay readable."""
    try:
        from cardiac_acr.cardiac_acr_diagnose_wsi import main
    except OpenSlideDependencyError as exc:
        raise SystemExit(
            "Cardiac-ACR cannot start because OpenSlide is not available.\n"
            f"{exc}\n"
            "Tip: run `python -m cardiac_acr.check_dependencies` to validate setup."
        ) from exc
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Cardiac-ACR cannot start because a required Python package is missing.\n"
            f"{format_missing_python_dependency(exc.name)}\n"
            "Tip: run `python -m cardiac_acr.check_dependencies` to validate setup."
        ) from exc

    main()


if __name__ == "__main__":
    run()
