"""
Data I/O Router
================
Endpoints for loading medical images, retrieving slices, and querying metadata.

Implements the HYBRID approach for file loading:
  - PRIMARY: Path reference (JSON body with local path) — fast, no copying
  - SECONDARY: File upload (multipart/form-data) — for browser-based uploads

Per §5.3: /api/v1/data/*
"""

import os
import sys
import io
import base64
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import Optional

# Ensure segpro-med root is importable
SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))

from api.schemas.data import (
    DataLoadPathRequest,
    DataLoadResponse,
    SliceResponse,
    MetadataResponse,
    SessionListResponse,
)
from api.services.session_manager import get_session_manager

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_file_type(path: str) -> str:
    """Detect whether the path is DICOM, NIfTI, or MAT."""
    p = path.lower()
    if os.path.isdir(path):
        # Check if directory contains .dcm files
        for f in os.listdir(path):
            if f.lower().endswith(".dcm"):
                return "dicom"
        # Could be a DICOM dir without .dcm extension — try pydicom on first file
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if files:
            try:
                import pydicom
                pydicom.dcmread(os.path.join(path, files[0]), stop_before_pixels=True)
                return "dicom"
            except Exception:
                pass
        raise ValueError(f"Directory does not contain recognizable medical images: {path}")
    elif p.endswith(".dcm"):
        return "dicom"
    elif p.endswith(".nii") or p.endswith(".nii.gz"):
        return "nifti"
    elif p.endswith(".mat"):
        return "mat"
    else:
        raise ValueError(f"Unsupported file type: {path}")


def _load_volume(file_path: str, file_type: str) -> tuple[np.ndarray, dict]:
    """Load a medical image volume using existing SegPro-Med modules."""
    if file_type == "dicom":
        from utils.dicom_utils import load_dicom_series
        volume, metadata, _ = load_dicom_series(file_path)
        canonical_view, axis_map = _detect_axis_map_from_iop(metadata)
        metadata["canonical_view"] = canonical_view
        metadata["axis_map"] = axis_map
        dicom_affine = _build_dicom_affine(metadata, volume.shape[0])
        if dicom_affine is not None:
            metadata["dicom_affine"] = dicom_affine.tolist()
        return volume, _sanitize_metadata(metadata)

    elif file_type == "nifti":
        import nibabel as nib
        img = nib.load(file_path)
        volume = img.get_fdata().astype(np.float32)
        canonical_view, axis_map = _detect_axis_map_from_affine(img.affine)
        metadata = {
            "shape": list(img.shape),
            "affine": img.affine.tolist(),
            "voxel_sizes": [float(v) for v in img.header.get_zooms()],
            "data_dtype": str(img.get_data_dtype()),
            "canonical_view": canonical_view,
            "axis_map": axis_map,
        }
        # Try extracting description
        if hasattr(img.header, "get") and "descrip" in img.header:
            metadata["description"] = str(img.header["descrip"])
        return volume, _sanitize_metadata(metadata)

    elif file_type == "mat":
        from scipy.io import loadmat
        mat_data = loadmat(file_path)
        # Find the volume array (largest ndarray in the .mat)
        volume = None
        for key, val in mat_data.items():
            if isinstance(val, np.ndarray) and not key.startswith("__"):
                if volume is None or val.size > volume.size:
                    volume = val
        if volume is None:
            raise ValueError(f"No numpy array found in MAT file: {file_path}")
        metadata = {
            "keys": [k for k in mat_data.keys() if not k.startswith("__")],
        }
        return volume.astype(np.float32), _sanitize_metadata(metadata)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _sanitize_metadata(obj):
    """Recursively convert numpy/non-JSON-serializable types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def _build_dicom_affine(metadata: dict, num_slices: int) -> np.ndarray | None:
    """Build a 4×4 affine matrix (voxel → world/mm) for a DICOM volume stored as
    (num_slices, rows, cols).

    Uses IOP (row/col direction cosines), IPP of first slice (origin),
    IPP of last slice (precise inter-slice direction + spacing), and PixelSpacing.
    Falls back to SliceThickness when only one slice is available.
    """
    iop         = metadata.get("ImageOrientationPatient")   # [Xx,Xy,Xz, Yx,Yy,Yz]
    ipp_first   = metadata.get("ImagePositionPatient")       # [x,y,z] of first slice
    ipp_last    = metadata.get("ImagePositionPatientLast")   # [x,y,z] of last slice
    ps          = metadata.get("PixelSpacing")               # [row_spacing, col_spacing]

    if not all([iop, ipp_first, ps]):
        return None

    row_cos = np.array(iop[:3], dtype=float)   # X: direction of increasing col index
    col_cos = np.array(iop[3:6], dtype=float)  # Y: direction of increasing row index

    # Precise inter-slice vector from first→last IPP
    if ipp_last and num_slices > 1:
        delta        = np.array(ipp_last, dtype=float) - np.array(ipp_first, dtype=float)
        slice_spacing = np.linalg.norm(delta) / (num_slices - 1)
        slice_dir     = delta / (np.linalg.norm(delta) + 1e-12)
    else:
        # Fallback: normal from row × col cosines + SliceThickness
        n             = np.cross(row_cos, col_cos)
        slice_dir     = n / (np.linalg.norm(n) + 1e-12)
        slice_spacing = float(metadata.get("SliceThickness") or 1.0)

    affine = np.eye(4)
    affine[:3, 0] = slice_dir * slice_spacing       # axis 0: slice direction
    affine[:3, 1] = col_cos * float(ps[0])          # axis 1: row direction (Y × row_spacing)
    affine[:3, 2] = row_cos * float(ps[1])          # axis 2: col direction (X × col_spacing)
    affine[:3, 3] = np.array(ipp_first, dtype=float)  # origin: first slice position
    return affine


def _detect_axis_map_from_iop(metadata: dict) -> tuple[str, dict]:
    """For a DICOM series stacked as (num_slices, rows, cols), infer the
    anatomical axis_map and canonical view from ImageOrientationPatient.

    Returns (canonical_view, axis_map).
    """
    iop = metadata.get("ImageOrientationPatient")
    if iop and len(iop) == 6:
        try:
            row_cos = np.array(iop[:3], dtype=float)
            col_cos = np.array(iop[3:], dtype=float)
            # Slice normal = row × col — its dominant component tells us the plane
            normal = np.cross(row_cos, col_cos)
            anat_labels = {0: "sagittal", 1: "coronal", 2: "axial"}
            stack_view   = anat_labels[int(np.argmax(np.abs(normal)))]
            row_view     = anat_labels[int(np.argmax(np.abs(row_cos)))]
            col_view     = anat_labels[int(np.argmax(np.abs(col_cos)))]
            axis_map = {stack_view: 0, row_view: 1, col_view: 2}
            if len(axis_map) == 3:           # all three are distinct
                return stack_view, axis_map
        except Exception:
            pass
    # Fallback: axial brain DICOM assumption (each slice is axial, stacked on axis 0)
    return "axial", {"axial": 0, "sagittal": 1, "coronal": 2}


def _detect_axis_map_from_affine(affine) -> tuple[str, dict]:
    """Use a NIfTI affine matrix to determine the correct axis_map and canonical view.

    Returns (canonical_view, axis_map).
    """
    try:
        import nibabel as nib
        ornt = nib.orientations.io_orientation(np.asarray(affine))
        # ornt[mem_axis] = (anat_axis, flip) where anat_axis: 0=R/L, 1=A/P, 2=I/S
        label_map = {0: "sagittal", 1: "coronal", 2: "axial"}
        axis_map: dict[str, int] = {}
        for mem_axis, (anat_axis, _) in enumerate(ornt):
            lbl = label_map.get(int(anat_axis))
            if lbl:
                axis_map[lbl] = mem_axis
        if len(axis_map) == 3:
            return "axial", axis_map
    except Exception:
        pass
    # Fallback: standard RAS
    return "axial", {"axial": 2, "coronal": 1, "sagittal": 0}


def _render_slice(
    volume: np.ndarray,
    slice_idx: int,
    view: str = "axial",
    axis_map: dict | None = None,
) -> tuple[np.ndarray, int]:
    """Extract a 2D slice from a 3D volume along the given view axis.

    Returns (slice_2d, total_slices_for_this_view).
    """
    ndim = getattr(volume, "ndim", len(getattr(volume, "shape", ())))

    if ndim == 2:
        return volume, 1

    if axis_map is None:
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = axis_map.get(view, next(iter(axis_map.values()), 2))

    # For 4D volumes, take first timepoint
    vol = volume[..., 0] if ndim == 4 else volume
    vol_ndim = getattr(vol, "ndim", len(getattr(vol, "shape", ())))

    total = vol.shape[axis]
    idx = max(0, min(slice_idx, total - 1))

    if vol_ndim >= 3 and axis == 0:
        slc = vol[idx, :, :]
    elif vol_ndim >= 3 and axis == 1:
        slc = vol[:, idx, :]
    elif vol_ndim >= 3 and axis == 2:
        slc = vol[:, :, idx]
    else:
        slc = np.take(vol, idx, axis=axis)
    return slc.astype(np.float32), total


def _coerce_display_float(value) -> float | None:
    """Convert a DICOM display metadata value to float when possible."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_slice_for_display(
    arr: np.ndarray,
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    """Normalize a slice to uint8, preferring DICOM window center/width."""
    data = arr.astype(np.float32)

    if window_center is not None and window_width is not None and window_width > 0:
        low = window_center - (window_width / 2.0)
        high = window_center + (window_width / 2.0)
        clipped = np.clip(data, low, high)
        return ((clipped - low) / (high - low) * 255).astype(np.uint8)

    mn, mx = float(np.nanmin(data)), float(np.nanmax(data))
    if mx - mn > 0:
        return ((data - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(data, dtype=np.uint8)


def _array_to_base64_png(
    arr: np.ndarray,
    window_center: float | None = None,
    window_width: float | None = None,
) -> str:
    """Convert a 2D numpy array to a base64-encoded PNG string."""
    normalized = _normalize_slice_for_display(arr, window_center, window_width)

    img = Image.fromarray(normalized, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Segmentation overlay helpers
# ---------------------------------------------------------------------------

# Module-level cache: absolute path → (volume, axis_map)
_seg_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load_seg(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI segmentation volume and its affine; cached by absolute path.

    Returns (volume, affine_4x4).
    """
    abs_path = os.path.abspath(path)
    if abs_path in _seg_cache:
        return _seg_cache[abs_path]
    import nibabel as nib
    img = nib.load(abs_path)
    vol = img.get_fdata().astype(np.float32)
    affine = np.array(img.affine, dtype=float)
    _seg_cache[abs_path] = (vol, affine)
    return vol, affine


# Resampled-seg cache keyed by "session_id:abs_seg_path"
_resampled_seg_cache: dict[str, np.ndarray] = {}


def _resample_seg_to_dicom(
    seg_vol: np.ndarray,
    seg_affine: np.ndarray,
    dicom_shape: tuple,
    dicom_affine: np.ndarray,
) -> np.ndarray:
    """Resample a NIfTI segmentation into a DICOM volume's voxel grid.

    For every voxel position (s, r, c) in the DICOM grid the function finds
    the corresponding NIfTI voxel via the world-space transform:

        nii_voxel = inv(nii_affine) @ dicom_affine @ [s, r, c, 1]ᵀ

    Uses nearest-neighbour interpolation so integer label values are preserved.
    """
    from scipy.ndimage import affine_transform

    # DICOM IOP/IPP coordinates are in LPS (Left-Posterior-Superior) space.
    # nibabel affines are in RAS (Right-Anterior-Superior) space.
    # The two conventions differ by negating the X and Y axes.
    # Without this conversion the overlay is rotated 180° in the slice plane.
    lps_to_ras    = np.diag([-1.0, -1.0, 1.0, 1.0])
    dicom_aff_ras = lps_to_ras @ dicom_affine       # DICOM voxel → RAS world mm

    world_to_nii  = np.linalg.inv(seg_affine)       # RAS world mm → NIfTI voxel
    dicom_to_nii  = world_to_nii @ dicom_aff_ras    # DICOM voxel → NIfTI voxel

    matrix = dicom_to_nii[:3, :3]
    offset = dicom_to_nii[:3, 3]

    resampled = affine_transform(
        seg_vol,
        matrix,
        offset=offset,
        output_shape=tuple(dicom_shape[:3]),
        order=0,            # nearest-neighbour → preserves integer labels
        mode="constant",
        cval=0.0,
    )
    return resampled.astype(np.float32)


_LABEL_COLORS = [
    (255,  60,  60),   # 1 – red
    ( 60, 230,  60),   # 2 – green
    ( 60, 120, 255),   # 3 – blue
    (255, 240,  50),   # 4 – yellow
    ( 50, 240, 220),   # 5 – cyan
    (220,  50, 220),   # 6 – magenta
    (255, 150,  50),   # 7 – orange
]


def _overlay_to_base64_png(
    base_slice: np.ndarray,
    seg_slice: np.ndarray,
    alpha: float = 0.45,
    window_center: float | None = None,
    window_width: float | None = None,
) -> str:
    """Composite a segmentation mask onto a grayscale slice; return base64 PNG."""
    norm = _normalize_slice_for_display(base_slice, window_center, window_width)

    # Grayscale → float32 RGB
    rgb = np.stack([norm, norm, norm], axis=2).astype(np.float32)

    # Resize seg to match base if spatial dims differ
    if seg_slice.shape != base_slice.shape:
        seg_pil = Image.fromarray(seg_slice.astype(np.uint8))
        seg_pil = seg_pil.resize(
            (base_slice.shape[1], base_slice.shape[0]), Image.NEAREST
        )
        seg_slice = np.array(seg_pil)

    for label_val in np.unique(seg_slice):
        if label_val == 0:
            continue
        color = _LABEL_COLORS[(int(label_val) - 1) % len(_LABEL_COLORS)]
        mask = seg_slice == label_val
        for c in range(3):
            rgb[mask, c] = np.clip(
                (1.0 - alpha) * rgb[mask, c] + alpha * color[c], 0, 255
            )

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/load",
    response_model=DataLoadResponse,
    summary="Load medical image data from a local path",
    description=(
        "PRIMARY approach: provide a local filesystem path to a DICOM directory, "
        "NIfTI file (.nii/.nii.gz), or MAT file. Returns a session_id for subsequent operations."
    ),
)
async def load_data_from_path(request: DataLoadPathRequest):
    path = request.path

    # Validate path exists
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        file_type = _detect_file_type(path)
        volume, metadata = _load_volume(path, file_type)

        sm = get_session_manager()
        session = sm.create_session(
            file_path=os.path.abspath(path),
            file_type=file_type,
            volume=volume,
            metadata=metadata,
        )

        return DataLoadResponse(
            session_id=session.session_id,
            file_path=session.file_path,
            file_type=file_type,
            volume_shape=list(volume.shape),
            metadata=metadata,
            message=f"Loaded {file_type.upper()} volume with shape {list(volume.shape)}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")


@router.post(
    "/upload",
    response_model=DataLoadResponse,
    summary="Upload a medical image file",
    description=(
        "SECONDARY approach: upload a file via multipart/form-data. "
        "Supports .nii, .nii.gz, .mat, .dcm, or .zip (containing DICOM series). "
        "The file is saved to a temp directory, then loaded into a session."
    ),
)
async def upload_data(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    # Create persistent upload directory
    upload_dir = SEGPRO_ROOT / "api" / "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    try:
        # Save uploaded file
        file_bytes = await file.read()
        save_path = os.path.join(upload_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # Handle ZIP files (DICOM series)
        if suffix == ".zip":
            extract_dir = os.path.join(upload_dir, Path(filename).stem)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(save_path)  # Remove the ZIP
            save_path = extract_dir

        file_type = _detect_file_type(save_path)
        volume, metadata = _load_volume(save_path, file_type)

        sm = get_session_manager()
        session = sm.create_session(
            file_path=os.path.abspath(save_path),
            file_type=file_type,
            volume=volume,
            metadata=metadata,
        )
        session.is_upload = True  # Mark for cleanup when session ends

        return DataLoadResponse(
            session_id=session.session_id,
            file_path=session.file_path,
            file_type=file_type,
            volume_shape=list(volume.shape),
            metadata=metadata,
            message=f"Uploaded and loaded {file_type.upper()} volume with shape {list(volume.shape)}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get(
    "/slice/{session_id}",
    response_model=SliceResponse,
    summary="Get a specific slice from a loaded volume",
)
async def get_slice(
    session_id: str,
    slice: int = Query(0, ge=0, description="Slice index (0-based)"),
    view: str = Query("axial", description="View orientation: axial, sagittal, coronal"),
    seg_path: Optional[str] = Query(
        None,
        description=(
            "Absolute path to a NIfTI segmentation file (e.g. Untitled.nii). "
            "When provided the segmentation mask is blended onto the slice image."
        ),
    ),
):
    sm = get_session_manager()
    session = sm.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if session.volume is None:
        raise HTTPException(status_code=400, detail="Session has no loaded volume")

    try:
        session_axis_map = session.metadata.get("axis_map") or None
        slice_2d, total = _render_slice(session.volume, slice, view, session_axis_map)
        window_center = _coerce_display_float(session.metadata.get("WindowCenter"))
        window_width = _coerce_display_float(session.metadata.get("WindowWidth"))

        if seg_path and os.path.exists(seg_path):
            seg_vol, seg_affine = _load_seg(seg_path)

            # --- World-space alignment ---
            # If the session was loaded from DICOM we have a precise 4×4 affine that
            # maps DICOM voxels → mm.  The NIfTI seg also has its own affine.
            # Resampling the seg into the DICOM voxel grid is the only way to
            # guarantee pixel-perfect alignment regardless of how the NIfTI was saved.
            dicom_affine_raw = session.metadata.get("dicom_affine")
            if dicom_affine_raw is not None:
                cache_key = f"{session_id}:{os.path.abspath(seg_path)}"
                if cache_key not in _resampled_seg_cache:
                    dicom_affine_arr = np.array(dicom_affine_raw, dtype=float)
                    dicom_shape      = session.volume.shape[:3]
                    _resampled_seg_cache[cache_key] = _resample_seg_to_dicom(
                        seg_vol, seg_affine, dicom_shape, dicom_affine_arr
                    )
                seg_for_overlay = _resampled_seg_cache[cache_key]
                # Resampled seg is now in DICOM voxel space → use the same axis_map
                seg_2d, _ = _render_slice(seg_for_overlay, slice, view, session_axis_map)
            else:
                # NIfTI-on-NIfTI (or unknown): use each volume's own axis_map
                _, seg_axis_map = _detect_axis_map_from_affine(seg_affine)
                seg_2d, _ = _render_slice(seg_vol, slice, view, seg_axis_map)

            img_b64 = _overlay_to_base64_png(
                slice_2d,
                seg_2d,
                window_center=window_center,
                window_width=window_width,
            )
        else:
            img_b64 = _array_to_base64_png(
                slice_2d,
                window_center=window_center,
                window_width=window_width,
            )

        return SliceResponse(
            session_id=session_id,
            slice_index=slice,
            view=view,
            total_slices=total,
            image_base64=img_b64,
            width=slice_2d.shape[1],
            height=slice_2d.shape[0],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Slice retrieval failed: {str(e)}")


@router.get(
    "/metadata/{session_id}",
    response_model=MetadataResponse,
    summary="Get metadata for a loaded volume",
)
async def get_metadata(session_id: str):
    sm = get_session_manager()
    session = sm.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return MetadataResponse(
        session_id=session_id,
        file_path=session.file_path,
        file_type=session.file_type,
        volume_shape=list(session.volume.shape) if session.volume is not None else [],
        metadata=session.metadata,
    )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List all active data sessions",
)
async def list_sessions():
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return SessionListResponse(sessions=sessions, count=len(sessions))


@router.delete(
    "/session/{session_id}",
    summary="Delete a data session and free memory",
)
async def delete_session(session_id: str):
    sm = get_session_manager()
    if sm.delete_session(session_id):
        return {"message": f"Session {session_id} deleted.", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
