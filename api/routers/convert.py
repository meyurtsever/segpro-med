"""
Format Conversion Router
=========================
Wraps utils/conversion.py as a REST endpoint.

Per §5.3: /api/v1/convert/*
Per WF-1: Medical Image Format Conversion pipeline
"""

import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))

from api.schemas.convert import ConvertRequest, ConvertResponse

router = APIRouter()


# Supported conversion types (matches utils/conversion.py)
VALID_CONVERSIONS = {
    "DICOM to NIFTI",
    "NIFTI to MAT",
    "DICOM to MAT",
    "NIFTI to PNG",
}


@router.get(
    "/types",
    summary="List supported conversion types",
)
async def list_conversion_types():
    """Return all supported conversion types and their required inputs."""
    return {
        "conversions": [
            {
                "type": "DICOM to NIFTI",
                "input": "DICOM directory",
                "output": ".nii.gz file",
                "description": "Convert DICOM series to NIfTI using dcm2niix",
            },
            {
                "type": "NIFTI to MAT",
                "input": ".nii or .nii.gz file",
                "output": ".mat file",
                "description": "Convert NIfTI to MATLAB format (volume + affine + header)",
            },
            {
                "type": "DICOM to MAT",
                "input": "DICOM directory",
                "output": ".mat file",
                "description": "Convert DICOM series to MATLAB format",
            },
            {
                "type": "NIFTI to PNG",
                "input": ".nii or .nii.gz file",
                "output": "Directory of PNG slices",
                "description": "Convert NIfTI to a series of PNG images along a given axis",
            },
        ]
    }


@router.post(
    "/run",
    response_model=ConvertResponse,
    summary="Run a format conversion",
    description=(
        "Convert between medical image formats. "
        "Wraps the existing perform_conversion() from utils/conversion.py."
    ),
)
async def run_conversion(request: ConvertRequest):
    # Validate conversion type
    if request.conversion_type not in VALID_CONVERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid conversion type: '{request.conversion_type}'. "
                   f"Valid types: {sorted(VALID_CONVERSIONS)}",
        )

    # Validate input path exists
    if not os.path.exists(request.input_path):
        raise HTTPException(
            status_code=404,
            detail=f"Input path not found: {request.input_path}",
        )

    # Validate input type matches conversion
    input_path = request.input_path
    conv_type = request.conversion_type

    if conv_type.startswith("DICOM") and not os.path.isdir(input_path):
        # Single .dcm file — check extension
        if not input_path.lower().endswith(".dcm"):
            raise HTTPException(
                status_code=400,
                detail=f"DICOM conversion requires a directory or .dcm file, got: {input_path}",
            )

    if conv_type.startswith("NIFTI"):
        if not (input_path.lower().endswith(".nii") or input_path.lower().endswith(".nii.gz")):
            raise HTTPException(
                status_code=400,
                detail=f"NIfTI conversion requires .nii or .nii.gz file, got: {input_path}",
            )

    try:
        from utils.conversion import perform_conversion

        output_path = perform_conversion(
            input_path=request.input_path,
            output_path=request.output_path,
            conversion_type=request.conversion_type,
        )

        # Get output size
        output_size = None
        if os.path.isfile(output_path):
            output_size = os.path.getsize(output_path)
        elif os.path.isdir(output_path):
            output_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, filenames in os.walk(output_path)
                for f in filenames
            )

        return ConvertResponse(
            input_path=request.input_path,
            output_path=str(output_path),
            conversion_type=request.conversion_type,
            success=True,
            message=f"Conversion '{request.conversion_type}' completed successfully.",
            output_size_bytes=output_size,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion tool not available: {str(e)}. "
                   f"Ensure required tools (e.g. dcm2niix) are installed or check input paths.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during conversion: {str(e)}",
        )
