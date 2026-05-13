"""
SegPro-Med API Server
=====================
FastAPI application that wraps existing SegPro-Med Python modules
into REST API endpoints for the React Flow workflow engine.

Structured per §5.3 of WORKFLOW_TRANSFORMATION_ROADMAP.md
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure segpro-med root is on sys.path so we can import utils/, models/, xai/
SEGPRO_ROOT = Path(__file__).resolve().parent.parent
if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))

from api.routers import data, convert, filesystem, segmentation, patients


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- Startup ---
    print(f"[API] SegPro-Med API starting...")
    print(f"[API] Project root: {SEGPRO_ROOT}")
    print(f"[API] Upload directory: {SEGPRO_ROOT / 'api' / 'uploads'}")
    os.makedirs(SEGPRO_ROOT / "api" / "uploads", exist_ok=True)
    yield
    # --- Shutdown ---
    print("[API] SegPro-Med API shutting down.")


app = FastAPI(
    title="SegPro-Med API",
    description=(
        "REST API wrapping SegPro-Med's medical imaging capabilities. "
        "Used by the React Flow workflow engine and callable from any HTTP client."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
# Allow the React Flow dev server (localhost:5173) and Gradio (localhost:7860)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alt dev server
        "http://localhost:7860",   # Gradio
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(data.router, prefix="/api/v1/data", tags=["Data I/O"])
app.include_router(convert.router, prefix="/api/v1/convert", tags=["Format Conversion"])
app.include_router(filesystem.router, prefix="/api/v1/fs", tags=["Filesystem"])
app.include_router(segmentation.router, prefix="/api/v1/segmentation", tags=["Segmentation"])
app.include_router(patients.router, prefix="/api/v1/patients", tags=["Patients"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "SegPro-Med API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
