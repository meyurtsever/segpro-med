"""
Anonymization Audit Logger

Records structured audit events for all de-identification and metadata
sanitization operations.  Each entry is a single JSON line appended to
``db/anonymization_audit.jsonl`` (JSON Lines format for easy parsing).

This module satisfies the HIPAA Security Rule §164.312(b) requirement for
audit controls that record and examine activity in information systems
containing ePHI.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AUDIT_DIR = Path("db")
_AUDIT_FILE = _AUDIT_DIR / "anonymization_audit.jsonl"
_write_lock = threading.Lock()


def _write_event(event: Dict[str, Any]):
    """Append a single JSON-line event to the audit log (thread-safe)."""
    try:
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with _write_lock:
            with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        logger.error(f"Failed to write audit event: {e}")


def log_metadata_sanitization(
    *,
    user_id: Optional[str] = None,
    study_hash: Optional[str] = None,
    fields_sanitized: Optional[List[str]] = None,
    fields_retained: Optional[List[str]] = None,
    trigger: str = "display",
):
    """Record a metadata sanitization event.

    Parameters
    ----------
    user_id : str, optional
        Authenticated user who triggered the operation.
    study_hash : str, optional
        SHA-256 hash prefix identifying the study (no raw path).
    fields_sanitized : list[str], optional
        DICOM tag names that were removed / blanked.
    fields_retained : list[str], optional
        Clinical fields intentionally kept.
    trigger : str
        What triggered the sanitization (``"display"``, ``"export"``, ``"api"``).
    """
    _write_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "metadata_sanitization",
        "user_id": user_id,
        "study_hash": study_hash,
        "fields_sanitized_count": len(fields_sanitized) if fields_sanitized else 0,
        "fields_sanitized": fields_sanitized,
        "fields_retained": fields_retained,
        "trigger": trigger,
    })


def log_deidentification(
    *,
    user_id: Optional[str] = None,
    study_hash: Optional[str] = None,
    method: str = "pydeface",
    success: bool = True,
    elapsed_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
    slices_processed: Optional[int] = None,
):
    """Record a facial de-identification event.

    Parameters
    ----------
    user_id : str, optional
        Authenticated user who triggered the operation.
    study_hash : str, optional
        SHA-256 hash prefix identifying the study.
    method : str
        De-identification method (``"pydeface"``, ``"pydeface_cli"``).
    success : bool
        Whether the operation completed without error.
    elapsed_seconds : float, optional
        Wall-clock seconds for the operation.
    error_message : str, optional
        Error description if ``success`` is False.
    slices_processed : int, optional
        Number of slices processed.
    """
    _write_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "deidentification",
        "user_id": user_id,
        "study_hash": study_hash,
        "method": method,
        "success": success,
        "elapsed_seconds": elapsed_seconds,
        "error_message": error_message,
        "slices_processed": slices_processed,
    })


def log_upload_cleanup(
    *,
    session_id: Optional[str] = None,
    file_path_hash: Optional[str] = None,
    success: bool = True,
):
    """Record the deletion of an uploaded file (PHI removal)."""
    _write_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "upload_cleanup",
        "session_id": session_id,
        "file_path_hash": file_path_hash,
        "success": success,
    })


def log_data_erasure(
    *,
    user_id: str,
    study_hash: Optional[str] = None,
    scope: str = "study",
    success: bool = True,
):
    """Record a GDPR Article 17 data-erasure request.

    Parameters
    ----------
    user_id : str
        User whose data is being erased.
    study_hash : str, optional
        If erasing a specific study; None if erasing all user data.
    scope : str
        ``"study"`` or ``"all_user_data"``.
    success : bool
        Whether the erasure completed without error.
    """
    _write_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "data_erasure",
        "user_id": user_id,
        "study_hash": study_hash,
        "scope": scope,
        "success": success,
    })
