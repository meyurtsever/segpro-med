"""
Filesystem Browser Router
==========================
Provides directory listing for the file/folder picker UI.

Per §5.3: /api/v1/fs/*
"""

import os
import platform
import string
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class NativeDialogResponse(BaseModel):
    """Response from the native OS file/folder picker."""

    path: str | None
    cancelled: bool
    mode: Literal["file", "directory"]


def _initial_directory(initial_path: str) -> str | None:
    if not initial_path:
        return None

    normalized = os.path.normpath(initial_path)
    if os.path.isdir(normalized):
        return normalized

    parent = os.path.dirname(normalized)
    return parent if parent and os.path.isdir(parent) else None


def _open_native_dialog(
    mode: Literal["file", "directory"],
    initial_path: str,
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Native OS file dialog is unavailable: {exc}",
        ) from exc

    options: dict[str, object] = {
        "title": "Select medical image file" if mode == "file" else "Select directory",
    }
    initial_dir = _initial_directory(initial_path)
    if initial_dir:
        options["initialdir"] = initial_dir

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()

        if mode == "file":
            selected = filedialog.askopenfilename(
                **options,
                filetypes=[
                    ("Medical image files", "*.dcm *.nii *.nii.gz *.mat *.zip"),
                    ("DICOM files", "*.dcm"),
                    ("NIfTI files", "*.nii *.nii.gz"),
                    ("MATLAB files", "*.mat"),
                    ("ZIP archives", "*.zip"),
                    ("All files", "*.*"),
                ],
            )
        else:
            selected = filedialog.askdirectory(**options)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Native OS file dialog failed: {exc}",
        ) from exc
    finally:
        if root is not None:
            root.destroy()

    return os.path.abspath(selected) if selected else None


@router.get(
    "/browse",
    summary="Browse filesystem directory",
    description=(
        "Returns a directory listing for the file/folder picker. "
        "On Windows, browsing root shows available drive letters."
    ),
)
async def browse_directory(
    path: str = Query("", description="Directory path to browse. Empty = show drives/root."),
):
    # Handle empty/root path — list drives on Windows, / on Unix
    if not path or path in ("/", "\\"):
        if platform.system() == "Windows":
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter}:",
                        "path": f"{letter}:/",
                        "is_dir": True,
                        "size": None,
                        "extension": "",
                    })
            return {"path": "/", "parent": None, "entries": drives}
        else:
            path = "/"

    # Normalize path
    path = os.path.normpath(path)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    entries = []
    try:
        for entry in sorted(
            os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())
        ):
            try:
                size = None
                if entry.is_file(follow_symlinks=False):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        pass

                entries.append({
                    "name": entry.name,
                    "path": entry.path.replace("\\", "/"),
                    "is_dir": entry.is_dir(follow_symlinks=False),
                    "size": size,
                    "extension": (
                        os.path.splitext(entry.name)[1].lower()
                        if not entry.is_dir(follow_symlinks=False)
                        else ""
                    ),
                })
            except (PermissionError, OSError):
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    parent_path = os.path.dirname(path)
    parent = parent_path.replace("\\", "/") if parent_path != path else None

    return {
        "path": path.replace("\\", "/"),
        "parent": parent,
        "entries": entries,
    }


@router.get(
    "/dialog",
    response_model=NativeDialogResponse,
    summary="Open native OS file/folder selector",
    description=(
        "Opens a native dialog on the API host and returns the selected absolute path. "
        "This is intended for local desktop use where the React UI and API run on the same machine."
    ),
)
def open_native_filesystem_dialog(
    mode: Literal["file", "directory"] = Query(
        ...,
        description="Select a file or a directory.",
    ),
    initial_path: str = Query(
        "",
        description="Optional initial file/directory path.",
    ),
):
    selected = _open_native_dialog(mode, initial_path)
    return NativeDialogResponse(
        path=selected,
        cancelled=selected is None,
        mode=mode,
    )
