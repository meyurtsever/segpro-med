"""
Segmentation Router
====================
Automatic segmentation via SAM2AutomaticMaskGenerator.
Mirrors the logic in ui/medsam2_handlers.py::run_sam2_fast_masking() and
_process_dicom_slice_fast() / _preprocess_dicom_for_sam() exactly.

POST /api/v1/segmentation/auto  — run SAM2 on a session slice
GET  /api/v1/segmentation/configs — list available config profiles
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = SEGPRO_ROOT / "models" / "medsam2"
SAM2_DIR     = MODELS_DIR / "sam2"
UTILS_DIR    = MODELS_DIR / "utils"

# SEGPRO_ROOT must have highest priority so project packages (utils, api, …)
# are never shadowed by medsam2 sub-dirs (models/medsam2/utils/ etc.)
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))
# medsam2-specific dirs are appended so build_sam / automatic_mask_generator
# are importable without clobbering project-level packages.
for _p in [str(MODELS_DIR), str(SAM2_DIR), str(UTILS_DIR)]:
    if _p not in sys.path:
        sys.path.append(_p)

from api.services.session_manager import get_session_manager
from brain_segmentation_configs import BRAIN_CONFIGS, CONFIG_DESCRIPTIONS

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Model paths (from medsam2_handlers.py)
# ---------------------------------------------------------------------------
_CONFIG_PATH     = str(MODELS_DIR / "configs"     / "sam2.1_hiera_b+.yaml")
_CHECKPOINT_PATH = str(MODELS_DIR / "checkpoints" / "sam2.1_hiera_base_plus.pt")

# ---------------------------------------------------------------------------
# Cached SAM2 model singleton
# ---------------------------------------------------------------------------
_sam2_model_cache      = None
_sam2_config_cached    = None
_sam2_checkpoint_cached = None


def _clear_hydra():
    """
    Aggressively purge ALL Hydra state from the Python process.
    The Editor tab (medsam2_handlers) and this router share the same process,
    so Hydra may already be initialized with a stale internal type.
    Strategy: clear GlobalHydra FIRST (while the module is still importable),
    then remove ALL hydra modules — including hydra.core.global_hydra itself —
    so the next build_sam2() call gets a completely fresh Hydra.
    """
    # Phase 1: clear the singleton while the module is still loaded
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        pass

    # Phase 2: purge ALL hydra modules (including global_hydra itself)
    to_remove = [m for m in list(sys.modules.keys()) if "hydra" in m.lower()]
    for m in to_remove:
        try:
            del sys.modules[m]
        except Exception:
            pass


def _get_or_load_sam2_model():
    """Load SAM2 model once and cache it (same caching strategy as medsam2_handlers)."""
    global _sam2_model_cache, _sam2_config_cached, _sam2_checkpoint_cached

    if _sam2_model_cache is not None:
        logger.debug("Using cached SAM2 model")
        return _sam2_model_cache

    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"SAM2 config not found: {_CONFIG_PATH}")
    if not os.path.exists(_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {_CHECKPOINT_PATH}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM2 model on {device} …")

    # Clear Hydra with retry (mirrors medsam2_handlers.py)
    for attempt in range(3):
        _clear_hydra()
        time.sleep(0.05 * attempt)

    # build_sam2 is importable because MODELS_DIR is on sys.path
    from build_sam import build_sam2  # noqa
    model = build_sam2(_CONFIG_PATH, _CHECKPOINT_PATH, device=device)

    _sam2_model_cache       = model
    _sam2_config_cached     = _CONFIG_PATH
    _sam2_checkpoint_cached = _CHECKPOINT_PATH
    logger.info("SAM2 model loaded and cached.")
    return model


# ---------------------------------------------------------------------------
# Image preprocessing — mirrors _preprocess_dicom_for_sam() exactly
# ---------------------------------------------------------------------------

def _preprocess_for_sam(arr: np.ndarray) -> np.ndarray:
    """
    Normalise raw 2-D (H,W) float/int slice to uint8 RGB.
    Exact copy of MedSAM2Handlers._preprocess_dicom_for_sam().
    """
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        scaled = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    # SAM expects 3-channel RGB input
    rgb = np.stack([scaled, scaled, scaled], axis=2) if scaled.ndim == 2 else scaled
    return rgb


def _extract_slice(volume: np.ndarray, slice_index: int, view: str, axis_map: dict | None = None) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume.

    Uses axis_map from session metadata when available (matches _render_slice in
    data.py).  Falls back to the same default only for NIfTI/MAT files that
    don't store an axis_map.
    """
    if axis_map is None:
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = axis_map.get(view, next(iter(axis_map.values()), 2))
    vol  = volume[..., 0] if volume.ndim == 4 else volume
    total = vol.shape[axis]
    idx   = max(0, min(slice_index, total - 1))
    slc   = np.take(vol, idx, axis=axis).astype(np.float32)
    logger.debug(
        "_extract_slice: volume=%s  view=%s  axis=%d  idx=%d/%d  slice_shape=%s",
        vol.shape, view, axis, idx, total, slc.shape,
    )
    return slc


# ---------------------------------------------------------------------------
# Mask filtering + polygon conversion
# ---------------------------------------------------------------------------

MASK_COLORS = [
    "#4fc3f7", "#81c784", "#ff8a65", "#ba68c8",
    "#ffd54f", "#e57373", "#4db6ac", "#7986cb",
    "#f06292", "#aed581", "#ffb74d", "#9575cd",
    "#4dd0e1", "#dce775", "#ff8a80", "#b39ddb",
]


# Polygon-conversion constants — must match mask_to_polygons() in
# utils/visualization.py so the workflow produces identical shapes to the Editor.
_CONTOUR_MIN_AREA = 50          # cv2.contourArea filter (same as mask_to_polygons)
_SIMPLIFY_TOLERANCE = 2.0       # fixed epsilon for cv2.approxPolyDP (same)


def _masks_to_polygons(masks: list, gray_image: np.ndarray | None = None) -> list:
    """
    1. Filter SAM2 masks: area >= 500 AND predicted_iou >= 0.8
       (same rule as _process_dicom_slice_fast in medsam2_handlers.py).
    2. Optionally filter masks whose mean pixel brightness is below a threshold
       (removes background / very dark regions such as skull exterior).
    3. Convert each binary mask → polygon(s) using the EXACT same algorithm
       as mask_to_polygons() in utils/visualization.py:
       • cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
       • contourArea >= 50  (skip tiny contours)
       • cv2.approxPolyDP with **fixed** epsilon = 2.0  (not %-based)
       • At least 3 vertices required
    """
    filtered = [
        m for m in masks
        if m.get("area", 0) >= 500 and m.get("predicted_iou", 0.0) >= 0.8
    ]
    # Remove masks whose mean intensity is very dark (background / non-tissue)
    if gray_image is not None:
        _DARK_THRESHOLD = 30.0  # mean pixel value (0-255) below which mask is rejected
        filtered = [
            m for m in filtered
            if float(gray_image[m["segmentation"] > 0].mean()) >= _DARK_THRESHOLD
        ]
    filtered.sort(key=lambda m: m.get("area", 0), reverse=True)

    shapes = []
    for i, mask_result in enumerate(filtered):
        seg  = mask_result["segmentation"]
        iou  = mask_result.get("predicted_iou", 0.0)
        area = mask_result.get("area", 0)

        mask_u8 = (seg > 0).astype(np.uint8)       # ensure binary
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Contour-area filter (same as mask_to_polygons in visualization.py)
            if cv2.contourArea(contour) < _CONTOUR_MIN_AREA:
                continue

            # Fixed-epsilon simplification (same as Editor tab)
            approx = cv2.approxPolyDP(contour, _SIMPLIFY_TOLERANCE, True)
            if len(approx) < 3:
                continue

            shapes.append({
                "type":   "polygon",
                "points": [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in approx],
                "label":  f"ROI {i + 1}  IoU={iou:.2f}  area={area}",
                "color":  MASK_COLORS[i % len(MASK_COLORS)],
            })

    return shapes


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PointSchema(BaseModel):
    x: int
    y: int


class AnnotationShapeSchema(BaseModel):
    type: str = "polygon"
    points: list[PointSchema] = []
    label: str = ""
    color: str = "#4fc3f7"


class AutoSegmentRequest(BaseModel):
    session_id: str
    slice_index: int = 0
    view: str = "axial"
    config_name: str = "fast"


class AutoSegmentResponse(BaseModel):
    shapes: list[AnnotationShapeSchema]
    count: int
    raw_mask_count: int = 0
    config_used: str
    elapsed_seconds: float
    message: str


class ConfigListResponse(BaseModel):
    configs: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auto", response_model=AutoSegmentResponse)
async def auto_segment(req: AutoSegmentRequest):
    """
    Run SAM2 automatic segmentation on a single slice.
    Returns polygon annotations ready to merge into the InteractiveAnnotator.
    Mirrors run_sam2_fast_masking() + _process_dicom_slice_fast() logic from
    ui/medsam2_handlers.py exactly.
    """
    t0 = time.time()

    # 1. Validate config
    if req.config_name not in BRAIN_CONFIGS:
        available = list(BRAIN_CONFIGS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown config '{req.config_name}'. Available: {available}",
        )

    # 2. Get session volume
    mgr = get_session_manager()
    session = mgr.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")
    if session.volume is None:
        raise HTTPException(status_code=400, detail="Session has no loaded volume")

    # 3. Extract slice using session's axis_map (same logic as data.py _render_slice)
    try:
        session_axis_map = session.metadata.get("axis_map") or None
        slice_2d = _extract_slice(session.volume, req.slice_index, req.view, session_axis_map)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Slice extraction failed: {e}")

    rgb_image = _preprocess_for_sam(slice_2d)
    logger.info(
        "Slice extracted: shape=%s  rgb_shape=%s  view=%s  slice_idx=%d",
        slice_2d.shape, rgb_image.shape, req.view, req.slice_index,
    )

    # 4. Load SAM2 model (cached singleton)
    try:
        sam2_model = _get_or_load_sam2_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM2 model loading failed: {e}")

    # 5. Build mask generator with brain config
    brain_config = BRAIN_CONFIGS[req.config_name].copy()

    # Resolved via sys.path (MODELS_DIR is on sys.path — same pattern as medsam2_handlers.py)
    from automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, **brain_config)

    # 6. Generate masks
    logger.info(
        "Running auto-segmentation: session=%s  slice=%d  view=%s  config=%s",
        req.session_id, req.slice_index, req.view, req.config_name,
    )
    masks = mask_generator.generate(rgb_image)
    raw_count = len(masks)
    logger.info("SAM2 generated %d raw masks", raw_count)

    # 7. Filter (area>=500, iou>=0.8, non-dark) + convert to polygon shapes
    # Pass the normalised 0-255 gray channel so the brightness threshold is meaningful
    shapes = _masks_to_polygons(masks, gray_image=rgb_image[:, :, 0])
    logger.info("After filter: %d polygons (filtered from %d)", len(shapes), raw_count)

    elapsed = time.time() - t0
    return AutoSegmentResponse(
        shapes=shapes,
        count=len(shapes),
        raw_mask_count=raw_count,
        config_used=req.config_name,
        elapsed_seconds=round(elapsed, 3),
        message=(
            f"Generated {len(shapes)} polygons "
            f"({raw_count} raw masks, {raw_count - len(shapes)} filtered) "
            f"using '{req.config_name}' config in {elapsed:.2f}s"
        ),
    )


@router.get("/configs", response_model=ConfigListResponse)
async def list_configs():
    """List all available SAM2 segmentation configuration profiles."""
    configs = []
    for name, params in BRAIN_CONFIGS.items():
        configs.append({
            "name":                 name,
            "description":          CONFIG_DESCRIPTIONS.get(name, ""),
            "points_per_side":      params.get("points_per_side", 32),
            "pred_iou_thresh":      params.get("pred_iou_thresh", 0.65),
            "min_mask_region_area": params.get("min_mask_region_area", 25),
        })
    return ConfigListResponse(configs=configs)
