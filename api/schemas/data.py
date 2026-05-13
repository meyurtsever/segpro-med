"""
Pydantic schemas for Data I/O endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# --- Request Schemas ---

class DataLoadPathRequest(BaseModel):
    """Load data from a local filesystem path (primary approach)."""
    path: str = Field(..., description="Absolute path to a DICOM directory, NIfTI file, or MAT file.")

    model_config = {"json_schema_extra": {
        "examples": [
            {"path": "C:/data/patient01/DICOM"},
            {"path": "/data/brain_scans/sub-001_T1w.nii.gz"},
        ]
    }}


# --- Response Schemas ---

class DataLoadResponse(BaseModel):
    """Response from loading medical image data."""
    session_id: str = Field(..., description="Unique session identifier for referencing this loaded volume.")
    file_path: str = Field(..., description="Resolved path to the loaded file/directory.")
    file_type: str = Field(..., description="Detected file type: 'dicom', 'nifti', or 'mat'.")
    volume_shape: list[int] = Field(..., description="Shape of the loaded volume [D, H, W] or [H, W].")
    metadata: dict = Field(default_factory=dict, description="Extracted metadata (DICOM tags, NIfTI header, etc.).")
    message: str = Field(default="Data loaded successfully.")


class SliceResponse(BaseModel):
    """Response from retrieving a single slice."""
    session_id: str
    slice_index: int
    view: str = Field(default="axial", description="View orientation: 'axial', 'sagittal', or 'coronal'.")
    total_slices: int
    image_base64: str = Field(..., description="Base64-encoded PNG of the slice.")
    width: int
    height: int


class MetadataResponse(BaseModel):
    """Response from metadata query."""
    session_id: str
    file_path: str
    file_type: str
    volume_shape: list[int]
    metadata: dict


class SessionListResponse(BaseModel):
    """Response listing all active sessions."""
    sessions: list[dict]
    count: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
