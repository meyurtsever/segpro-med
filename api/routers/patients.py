"""
Patient Retrieval Router
=========================
Endpoints for searching patients by ICD-10 code or free-text keyword.

Wraps utils/patient_retrieval.py + utils/icd10_mapping.py for the
React Flow workflow engine.

Per §5.3: /api/v1/patients/*
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

# Ensure segpro-med root is importable
SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))

# Use the global singleton — same instance as the Gradio app, same default root dir.
from utils.patient_retrieval import patient_retrieval as _global_retrieval, PatientRetrieval, set_root_directory
from utils.icd10_mapping import get_icd10_for_folder

router = APIRouter()


@router.get(
    "/search",
    summary="Search patients by ICD-10 code or keyword",
    description=(
        "Searches a structured medical imaging directory for patients matching "
        "the given query. Supports ICD-10 codes (e.g. 'D18.02'), clinical aliases "
        "(e.g. 'cavernoma', 'glioblastoma') and raw folder/patient name substrings. "
        "Returns up to 50 matches, ranked by relevance (ICD hits first). "
        "When root_dir is supplied the singleton root is updated for the lifetime of "
        "the server process."
    ),
)
async def search_patients(
    q: str = Query(..., min_length=1, description="ICD-10 code or keyword to search"),
    root_dir: Optional[str] = Query(
        None,
        description=(
            "Override the root directory of the structured dataset "
            "(anomaly_class/patient_id/FLAIR/ layout). "
            "Defaults to the path configured in utils/patient_retrieval.py."
        ),
    ),
):
    # If caller passes a custom root, update the singleton so the cache is
    # built from that directory (and invalidate the old cache).
    if root_dir:
        set_root_directory(root_dir)

    retrieval = _global_retrieval
    display_names = retrieval.search_patients(q)

    results = []
    for dn in display_names:
        info = retrieval.get_patient_info(dn)
        if not info:
            continue
        icd_info = get_icd10_for_folder(info["anomaly_class"]) or {}
        results.append(
            {
                "display_name": dn,
                "path": info.get("flair_path") or "",
                "anomaly_class": info["anomaly_class"],
                "icd10_code": icd_info.get("icd10_code", ""),
                "icd10_description": icd_info.get("icd10_description", ""),
                "segmentation_path": info.get("segmentation_path") or "",
            }
        )

    return {"results": results, "total": len(results)}


@router.get(
    "/root",
    summary="Get or set the dataset root directory",
    description="Returns the current root directory used by the patient retrieval singleton.",
)
async def get_root():
    return {
        "root_directory": _global_retrieval.root_directory,
        "directory_exists": _global_retrieval.is_directory_valid(),
        "cache_initialized": _global_retrieval._cache_initialized,
        "total_patients": len(_global_retrieval._patient_cache) if _global_retrieval._cache_initialized else None,
    }

