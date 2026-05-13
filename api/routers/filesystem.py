"""
Filesystem Browser Router
==========================
Provides directory listing for the file/folder picker UI.

Per §5.3: /api/v1/fs/*
"""

import os
import platform
import string

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


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
