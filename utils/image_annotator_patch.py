"""Runtime compatibility patch for gradio-image-annotation viewport events."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

_PAN_MOUSEUP_ORIGINAL = (
    '    Z === s.erase && L && Je(), ve("change");'
)
_PAN_MOUSEUP_PATCHED = (
    '    Z === s.erase && L && Je(), '
    '!(Z === s.drag && C.isDragging) && ve("change");'
)
_RESIZE_ORIGINAL = (
    '      oe(), ve("change");\n'
    '    }\n'
    '  }\n'
    '  const zu = new ResizeObserver(Ns);'
)
_RESIZE_PATCHED = (
    '      oe();\n'
    '    }\n'
    '  }\n'
    '  const zu = new ResizeObserver(Ns);'
)
def patch_image_annotator_bundle(bundle_path: Path) -> bool:
    """Patch one installed frontend bundle; return whether it was modified."""
    source = bundle_path.read_text(encoding="utf-8")
    patched = source

    if _PAN_MOUSEUP_ORIGINAL in patched:
        patched = patched.replace(
            _PAN_MOUSEUP_ORIGINAL, _PAN_MOUSEUP_PATCHED, 1
        )
    elif _PAN_MOUSEUP_PATCHED not in patched:
        raise RuntimeError("Image annotator pan handler signature was not recognized")

    if _RESIZE_ORIGINAL in patched:
        patched = patched.replace(_RESIZE_ORIGINAL, _RESIZE_PATCHED, 1)
    elif _RESIZE_PATCHED not in patched:
        raise RuntimeError("Image annotator resize handler signature was not recognized")

    if patched == source:
        return False

    bundle_path.write_text(patched, encoding="utf-8", newline="\n")
    return True


def apply_image_annotator_viewport_patch() -> bool:
    """Patch the active gradio-image-annotation installation if necessary.

    The component emits ``change`` after every Move-mode mouseup and resize.
    Those events contain no annotation change for viewport-only operations, but
    they still start Gradio callbacks and loading UI. The patch keeps change
    events for shape movement while suppressing pure panning and resize events.
    """
    try:
        spec = importlib.util.find_spec("gradio_image_annotation")
        if spec is None or spec.origin is None:
            logger.warning("gradio_image_annotation is unavailable; patch skipped")
            return False

        bundle_path = (
            Path(spec.origin).parent / "templates" / "component" / "index.js"
        )
        modified = patch_image_annotator_bundle(bundle_path)
        if modified:
            logger.info("Patched image annotator viewport-only change events")
        return modified
    except Exception as exc:
        logger.warning("Could not patch image annotator viewport events: %s", exc)
        return False
