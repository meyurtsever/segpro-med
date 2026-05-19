"""
Annotation Router
=================
Persistent annotation storage and export endpoints for workflow nodes.
"""

from typing import Any, Optional

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
