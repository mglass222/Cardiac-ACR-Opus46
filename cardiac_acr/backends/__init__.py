"""Pluggable model backends.

Each backend (``uni``, ``resnet``) lives in a subpackage and exposes a
``load_classifier(device, checkpoint_path=None) -> BackendClassifier``
function. The WSI-diagnosis code under ``cardiac_acr.wsi.diagnose``
consumes a ``BackendClassifier`` and does not know which backend
produced it — that is the whole point of this indirection.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import importlib
import torch


@dataclass
class BackendClassifier:
    """Per-backend bundle consumed by backend-agnostic inference code."""

    # Short name, e.g. "uni" or "resnet". Useful for logs.
    name: str

    # (B, 3, H, W) tensor on ``device`` → (B, num_classes) logits on device.
    classify: Callable[[torch.Tensor], torch.Tensor]

    # Class name list. Alphabetical, matching ImageFolder ordering.
    classes: Sequence[str]

    # PIL.Image -> Tensor[3, H, W]. Applied to each patch before stacking.
    transform: Callable

    # Device that ``classify`` expects its input to live on.
    device: torch.device

    # Per-backend output locations. These are separated from the shared
    # config so two backends can write into the same data/ tree without
    # clobbering each other.
    saved_database_dir: str
    slide_dx_dir: str
    annotated_png_dir: str
    test_slide_predictions_dir: str
    test_slide_annotations_dir: str


_BACKENDS = ("uni", "resnet")


def load_classifier(
    name: str,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> BackendClassifier:
    """Dispatch to the named backend's ``classifier.load_classifier``."""
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; expected one of {_BACKENDS}"
        )
    module = importlib.import_module(
        f"cardiac_acr.backends.{name}.classifier"
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return module.load_classifier(device=device, checkpoint_path=checkpoint_path)
