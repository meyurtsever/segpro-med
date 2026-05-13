"""
Session Manager
===============
In-memory store for loaded medical imaging volumes.
Each loaded file gets a unique session_id that downstream
API calls use to reference the data without re-loading.
"""

import os
import shutil
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a loaded medical imaging session."""
    session_id: str
    file_path: str
    file_type: str               # "dicom", "nifti", "mat"
    volume: np.ndarray | None    # 3D/4D numpy array
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    is_upload: bool = False       # True when file was uploaded (should be cleaned up)

    def touch(self):
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()


class SessionManager:
    """Thread-safe in-memory session store."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}  # dict[str, Session]
            return cls._instance

    def create_session(
        self,
        file_path: str,
        file_type: str,
        volume: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> Session:
        """Create a new session and return it."""
        session_id = uuid.uuid4().hex[:12]
        session = Session(
            session_id=session_id,
            file_path=file_path,
            file_type=file_type,
            volume=volume,
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID, updating its last_accessed time."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def _cleanup_upload(self, session: Session):
        """Remove uploaded file/directory from disk if the session was created via upload."""
        if not session.is_upload or not session.file_path:
            return
        try:
            target = session.file_path
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
                logger.info(f"Cleaned up uploaded directory: {target}")
            elif os.path.isfile(target):
                os.remove(target)
                logger.info(f"Cleaned up uploaded file: {target}")
        except Exception as e:
            logger.warning(f"Failed to cleanup upload {session.file_path}: {e}")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID, cleaning up uploaded files. Returns True if it existed."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            self._cleanup_upload(session)
            return True
        return False

    def list_sessions(self) -> list[dict]:
        """List all active sessions (without volume data)."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "file_path": s.file_path,
                    "file_type": s.file_type,
                    "metadata": s.metadata,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                    "volume_shape": list(s.volume.shape) if s.volume is not None else None,
                }
                for s in self._sessions.values()
            ]

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove sessions older than max_age_seconds. Returns count removed."""
        now = time.time()
        to_remove = []
        removed_sessions = []
        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_accessed > max_age_seconds:
                    to_remove.append(sid)
            for sid in to_remove:
                removed_sessions.append(self._sessions.pop(sid))
        # Cleanup uploads outside the lock
        for session in removed_sessions:
            self._cleanup_upload(session)
        return len(to_remove)


# Singleton accessor
def get_session_manager() -> SessionManager:
    return SessionManager()
