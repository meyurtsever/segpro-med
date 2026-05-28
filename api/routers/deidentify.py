"""
Deidentification Router
=======================
Creates deidentified copies of loaded studies for automated release workflows.

The endpoint keeps source data untouched, blanks HIPAA Safe Harbor-style DICOM
metadata fields using utils.metadata_sanitizer, optionally defaces 3D NIfTI
volumes with utils.deidentification, and writes audit entries to
db/anonymization_audit.jsonl.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))

from api.routers.data import _detect_file_type, _load_volume
from api.services.session_manager import get_session_manager
from utils.audit_logger import log_deidentification, log_metadata_sanitization
from utils.metadata_sanitizer import get_phi_field_names, sanitize_metadata_for_display

router = APIRouter()


class DeidentifyRequest(BaseModel):
    session_id: str | None = None
    source_path: str | None = None
    output_path: str | None = None
    apply_deface: bool = True
    user_id: str | None = None


class DeidentifyResponse(BaseModel):
    success: bool
    session_id: str
    output_session_id: str
    source_path: str
    output_path: str
    file_type: str
    volume_shape: list[int]
    sanitized_metadata: dict[str, Any] = Field(default_factory=dict)
    sanitized_fields: list[str] = Field(default_factory=list)
    sanitized_field_count: int = 0
    files_processed: int = 0
    audit_path: str = "db/anonymization_audit.jsonl"
    message: str


def _study_hash(path: str) -> str:
    return hashlib.sha256(os.path.abspath(path).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _default_output_path(source_path: str, file_type: str) -> Path:
    output_root = SEGPRO_ROOT / "api" / "outputs" / "deidentified"
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).name
    if Path(source_path).is_dir():
        stem = Path(source_path).name or "dicom_series"
        return output_root / f"{stem}_deidentified"
    if source_path.lower().endswith(".nii.gz"):
        stem = Path(source_path).name[:-7]
        return output_root / f"{stem}_deidentified.nii.gz"
    suffix = Path(source_path).suffix or (".dcm" if file_type == "dicom" else "")
    return output_root / f"{Path(source_path).stem}_deidentified{suffix}"


def _iter_dicom_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    files: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".dcm":
            files.append(candidate)
            continue
        try:
            import pydicom

            pydicom.dcmread(str(candidate), stop_before_pixels=True)
            files.append(candidate)
        except Exception:
            pass
    return files


def _blank_dicom_phi(source_path: Path, output_path: Path) -> tuple[int, list[str]]:
    import pydicom

    dicom_files = _iter_dicom_files(source_path)
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {source_path}")

    sanitized_fields = get_phi_field_names()
    touched_fields: set[str] = set()

    if source_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for file_path in dicom_files:
        ds = pydicom.dcmread(str(file_path))
        for field in sanitized_fields:
            if hasattr(ds, field):
                setattr(ds, field, "")
                touched_fields.add(field)
        try:
            ds.remove_private_tags()
        except Exception:
            pass

        if source_path.is_dir():
            relative_path = file_path.relative_to(source_path)
            target_path = output_path / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            target_path = output_path
        ds.save_as(str(target_path), write_like_original=False)

    return len(dicom_files), sorted(touched_fields)


def _copy_or_deface_nifti(source_path: Path, output_path: Path, apply_deface: bool) -> tuple[int, bool]:
    import nibabel as nib
    from utils.deidentification import apply_deidentification_to_volume

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(source_path))
    volume = img.get_fdata().astype(np.float32)
    deidentified = apply_deidentification_to_volume(volume) if apply_deface else volume
    header = img.header.copy()
    for field in ("descrip", "db_name", "aux_file"):
        if field in header:
            header[field] = b""
    out_img = nib.Nifti1Image(deidentified.astype(volume.dtype), img.affine, header)
    nib.save(out_img, str(output_path))
    return 1, apply_deface


def _load_deidentified_output(output: Path, file_type: str) -> tuple[np.ndarray, dict[str, Any]]:
    if file_type == "dicom" and output.is_file():
        from utils.dicom_utils import load_single_dicom

        volume, metadata = load_single_dicom(str(output))
        if volume is None:
            raise ValueError(f"Could not load deidentified DICOM file: {output}")
        return np.asarray(volume), sanitize_metadata_for_display(metadata)
    volume, metadata = _load_volume(str(output), file_type)
    return volume, metadata


@router.post("/run", response_model=DeidentifyResponse)
async def run_deidentification(req: DeidentifyRequest):
    t0 = time.time()
    sm = get_session_manager()
    source_path = req.source_path
    source_session_id = req.session_id or ""

    if req.session_id:
        session = sm.get_session(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")
        source_path = source_path or session.file_path

    if not source_path:
        raise HTTPException(status_code=400, detail="Provide session_id or source_path")

    source = Path(source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Source path not found: {source_path}")

    try:
        file_type = _detect_file_type(str(source))
        output = Path(req.output_path) if req.output_path else _default_output_path(str(source), file_type)
        study_hash = _study_hash(str(source))
        sanitized_fields: list[str] = []
        files_processed = 0

        if file_type == "dicom":
            files_processed, sanitized_fields = _blank_dicom_phi(source, output)
            log_metadata_sanitization(
                user_id=req.user_id,
                study_hash=study_hash,
                fields_sanitized=sanitized_fields,
                fields_retained=[],
                trigger="deidentify_export",
            )
        elif file_type == "nifti":
            files_processed, defaced = _copy_or_deface_nifti(source, output, req.apply_deface)
            sanitized_fields = get_phi_field_names()
            log_metadata_sanitization(
                user_id=req.user_id,
                study_hash=study_hash,
                fields_sanitized=sanitized_fields,
                fields_retained=[],
                trigger="deidentify_export",
            )
            log_deidentification(
                user_id=req.user_id,
                study_hash=study_hash,
                method="pydeface" if req.apply_deface else "copy_without_deface",
                success=True,
                elapsed_seconds=round(time.time() - t0, 2),
                slices_processed=int(np.prod(_load_deidentified_output(output, file_type)[0].shape[-1:])) if defaced else None,
            )
        else:
            raise ValueError("Deidentify currently supports DICOM and NIfTI inputs.")

        volume, metadata = _load_deidentified_output(output, file_type)
        sanitized_metadata = sanitize_metadata_for_display(metadata)
        output_session = sm.create_session(
            file_path=os.path.abspath(output),
            file_type=file_type,
            volume=volume,
            metadata=sanitized_metadata,
        )

        return DeidentifyResponse(
            success=True,
            session_id=source_session_id or output_session.session_id,
            output_session_id=output_session.session_id,
            source_path=os.path.abspath(source),
            output_path=os.path.abspath(output),
            file_type=file_type,
            volume_shape=list(volume.shape),
            sanitized_metadata=sanitized_metadata,
            sanitized_fields=sanitized_fields,
            sanitized_field_count=len(sanitized_fields),
            files_processed=files_processed,
            message=f"Deidentified {files_processed} {file_type.upper()} file(s).",
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_deidentification(
            user_id=req.user_id,
            study_hash=_study_hash(str(source)),
            method="workflow_deidentify",
            success=False,
            error_message=str(exc)[:240],
        )
        raise HTTPException(status_code=500, detail=f"Deidentification failed: {exc}") from exc
