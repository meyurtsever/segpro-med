"""
Pydantic schemas for Format Conversion endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ConvertRequest(BaseModel):
    """Request to convert between medical image formats."""
    input_path: str = Field(
        ..., description="Path to input file or directory (DICOM dir, .nii.gz, .mat)."
    )
    output_path: Optional[str] = Field(
        None, description="Output path. Auto-generated if not provided."
    )
    conversion_type: str = Field(
        ..., description="Conversion type: 'DICOM to NIFTI', 'NIFTI to MAT', 'DICOM to MAT', 'NIFTI to PNG'."
    )
    axis: Optional[int] = Field(
        2, description="Axis for PNG slicing (0=sagittal, 1=coronal, 2=axial). Only for NIFTI to PNG."
    )
    compress: Optional[bool] = Field(
        True, description="Whether to compress NIfTI output (.nii.gz vs .nii)."
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "input_path": "C:/data/patient01/DICOM",
                "conversion_type": "DICOM to NIFTI",
            },
            {
                "input_path": "/data/brain.nii.gz",
                "output_path": "/data/brain.mat",
                "conversion_type": "NIFTI to MAT",
            },
        ]
    }}


class ConvertResponse(BaseModel):
    """Response from a format conversion."""
    input_path: str
    output_path: str
    conversion_type: str
    success: bool
    message: str
    output_size_bytes: Optional[int] = None
    metadata: Optional[dict] = None
