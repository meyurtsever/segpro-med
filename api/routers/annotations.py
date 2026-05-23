"""
Annotation Router
=================
Persistent annotation storage and export endpoints for workflow nodes.
"""

import os
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.annotation_manager import get_annotation_manager

router = APIRouter()


class AnnotationStoreRequest(BaseModel):
    user_id: str = Field(default="workflow_user")
    study_path: str
    annotation_type: str = Field(default="manual")
    slice_annotations: list[dict[str, Any]] = []
    study_metadata: Optional[dict[str, Any]] = None
    vlm_labels: Optional[dict[str, Any]] = None


class AnnotationLoadRequest(BaseModel):
    user_id: str = Field(default="workflow_user")
    study_path: str


class AnnotationExportRequest(BaseModel):
    user_id: str = Field(default="workflow_user")
    study_path: str
    output_format: str = Field(default="json")


class VlmLabelDecisionRequest(BaseModel):
    user_id: str = Field(default="workflow_user")
    study_path: str
    slice_idx: int
    view_type: str = Field(default="axial")
    label: str
    action: Literal["accepted", "rejected"]
    suggested_labels: list[str] = []
    source: str = Field(default="workflow")


class AnnotationStoreResponse(BaseModel):
    success: bool
    user_id: str
    study_path: str
    annotation_count: int
    message: str


class AnnotationLoadResponse(BaseModel):
    found: bool
    user_id: str
    study_path: str
    annotation_count: int
    data: Optional[dict[str, Any]] = None
    message: str


class AnnotationExportResponse(BaseModel):
    success: bool
    user_id: str
    study_path: str
    output_format: str
    export_path: Optional[str] = None
    message: str


class VlmLabelDecisionResponse(BaseModel):
    success: bool
    user_id: str
    study_path: str
    slice_idx: int
    view_type: str
    label: str
    action: Literal["accepted", "rejected"]
    saved_labels: list[str]
    message: str


def _clean_label(label: str) -> str:
    return " ".join(str(label or "").strip().split())


def _label_key(label: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    return "_".join(part for part in safe.split("_") if part)[:80] or "label"


def _labels_file_dir(study_path: str) -> str:
    return study_path if os.path.isdir(study_path) else os.path.dirname(study_path)


def _read_slice_labels(labels_file_path: str) -> dict[int, list[str]]:
    labels_by_slice: dict[int, list[str]] = {}
    if not os.path.exists(labels_file_path):
        return labels_by_slice

    with open(labels_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            slice_str, labels_str = line.split(":", 1)
            try:
                slice_num = int(slice_str.strip())
            except ValueError:
                continue
            labels_by_slice[slice_num] = [
                _clean_label(label)
                for label in labels_str.split(",")
                if _clean_label(label)
            ]
    return labels_by_slice


def _append_label_to_labels_file(study_path: str, slice_idx: int, label: str) -> list[str]:
    labels_dir = _labels_file_dir(study_path)
    if not labels_dir:
        raise ValueError("Cannot determine label directory from study_path")
    os.makedirs(labels_dir, exist_ok=True)
    labels_file_path = os.path.join(labels_dir, "labels.txt")

    labels_by_slice = _read_slice_labels(labels_file_path)
    current_labels = labels_by_slice.get(slice_idx, [])
    if label and label not in current_labels:
        current_labels.append(label)
    labels_by_slice[slice_idx] = current_labels

    with open(labels_file_path, "w", encoding="utf-8") as f:
        for slice_num in sorted(labels_by_slice.keys()):
            clean_labels = [
                _clean_label(slice_label)
                for slice_label in labels_by_slice[slice_num]
                if _clean_label(slice_label)
            ]
            if clean_labels:
                f.write(f"{slice_num}: {', '.join(clean_labels)}\n")

    return current_labels


@router.post("/store", response_model=AnnotationStoreResponse)
async def store_annotations(req: AnnotationStoreRequest):
    """Store workflow annotations using the shared AnnotationManager."""
    if not req.study_path.strip():
        raise HTTPException(status_code=400, detail="study_path is required")

    manager = get_annotation_manager()
    success = manager.save_annotations(
        user_id=req.user_id,
        study_path=req.study_path,
        slice_annotations=req.slice_annotations,
        annotation_type=req.annotation_type,
        study_metadata=req.study_metadata,
        vlm_labels=req.vlm_labels,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Annotation storage failed")

    return AnnotationStoreResponse(
        success=True,
        user_id=req.user_id,
        study_path=req.study_path,
        annotation_count=len(req.slice_annotations),
        message=f"Stored {len(req.slice_annotations)} annotation(s).",
    )


@router.post("/vlm-label-decision", response_model=VlmLabelDecisionResponse)
async def review_vlm_label_decision(req: VlmLabelDecisionRequest):
    """Accept or reject one VLM label suggestion for a slice."""
    if not req.study_path.strip():
        raise HTTPException(status_code=400, detail="study_path is required")

    label = _clean_label(req.label)
    if not label:
        raise HTTPException(status_code=400, detail="label is required")

    now = datetime.now().isoformat()
    saved_labels: list[str] = []

    if req.action == "accepted":
        try:
            saved_labels = _append_label_to_labels_file(req.study_path, req.slice_idx, label)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not save label: {exc}") from exc
    else:
        labels_file_path = os.path.join(_labels_file_dir(req.study_path), "labels.txt")
        saved_labels = _read_slice_labels(labels_file_path).get(req.slice_idx, [])

    manager = get_annotation_manager()
    record_key = f"slice_{req.slice_idx}_{req.view_type}_{_label_key(label)}"
    vlm_record = {
        "slice_idx": req.slice_idx,
        "view_type": req.view_type,
        "annotation_id": record_key,
        "suggested_labels": [_clean_label(item) for item in req.suggested_labels if _clean_label(item)],
        "reviewed_label": label,
        "accepted_label": label if req.action == "accepted" else None,
        "rejected_label": label if req.action == "rejected" else None,
        "decision": req.action,
        "source": req.source,
        "decided_at": now,
    }
    success = manager.save_annotations(
        user_id=req.user_id,
        study_path=req.study_path,
        slice_annotations=[],
        annotation_type="vlm_label_review",
        study_metadata={"latest_label_review_slice": req.slice_idx, "latest_label_review_view": req.view_type},
        vlm_labels={record_key: vlm_record},
    )
    if not success:
        raise HTTPException(status_code=500, detail="Could not store VLM label decision")

    return VlmLabelDecisionResponse(
        success=True,
        user_id=req.user_id,
        study_path=req.study_path,
        slice_idx=req.slice_idx,
        view_type=req.view_type,
        label=label,
        action=req.action,
        saved_labels=saved_labels,
        message=(
            f"Accepted '{label}' for slice {req.slice_idx}."
            if req.action == "accepted"
            else f"Rejected '{label}' for slice {req.slice_idx}."
        ),
    )


@router.post("/load", response_model=AnnotationLoadResponse)
async def load_annotations(req: AnnotationLoadRequest):
    """Load saved annotations for a user/study pair."""
    if not req.study_path.strip():
        raise HTTPException(status_code=400, detail="study_path is required")

    manager = get_annotation_manager()
    data = manager.load_annotations(req.user_id, req.study_path)
    annotation_count = len((data or {}).get("slice_annotations", []))

    return AnnotationLoadResponse(
        found=data is not None,
        user_id=req.user_id,
        study_path=req.study_path,
        annotation_count=annotation_count,
        data=data,
        message=(
            f"Loaded {annotation_count} annotation(s)."
            if data is not None
            else "No saved annotations found."
        ),
    )


@router.post("/export", response_model=AnnotationExportResponse)
async def export_annotations(req: AnnotationExportRequest):
    """Export saved annotations in the requested format."""
    if not req.study_path.strip():
        raise HTTPException(status_code=400, detail="study_path is required")

    manager = get_annotation_manager()
    export_path = manager.export_annotations(
        user_id=req.user_id,
        study_path=req.study_path,
        output_format=req.output_format,
    )
    if not export_path:
        raise HTTPException(status_code=404, detail="No annotations were available to export")

    return AnnotationExportResponse(
        success=True,
        user_id=req.user_id,
        study_path=req.study_path,
        output_format=req.output_format,
        export_path=export_path,
        message=f"Exported annotations to {export_path}.",
    )
