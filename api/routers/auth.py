"""
Authentication Router
=====================
Workflow login endpoints backed by the same ``db/users.txt`` file used by the
Gradio crowdsourcing implementation.

This router is intentionally lightweight: it authenticates a user, returns their
role, and for expert users includes their remaining annotation assignments from
``db/assignments.json`` so the React Flow app can open the next task directly.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from auth.auth_manager import AuthManager
from crowdsourcing.campaign_manager import CrowdsourcingManager

router = APIRouter()

VALID_MODALITIES = ["flair", "t1", "t1c", "t2"]


class LoginRequest(BaseModel):
    username: str
    password: str


class CrowdsourcingTask(BaseModel):
    campaign_id: str
    patient_id: str
    dataset_path: str
    patient_path: str | None = None
    load_path: str | None = None
    modality: str | None = None


class LoginResponse(BaseModel):
    success: bool
    user_id: str
    role: str
    expert_score: float
    remaining_tasks: list[CrowdsourcingTask] = []
    assigned_tasks: list[CrowdsourcingTask] = []
    message: str


def _task_with_paths(task: dict[str, Any]) -> CrowdsourcingTask:
    campaign_id = str(task.get("campaign_id") or task.get("campaign") or "")
    patient_id = str(task.get("patient_id") or "")
    dataset_path = str(task.get("dataset_path") or "")
    patient_path = os.path.join(dataset_path, patient_id) if dataset_path and patient_id else None
    load_path = patient_path if patient_path and os.path.exists(patient_path) else None
    modality: str | None = None

    if patient_path and os.path.isdir(patient_path):
        for candidate in VALID_MODALITIES:
            candidate_path = os.path.join(patient_path, candidate)
            if os.path.isdir(candidate_path):
                load_path = candidate_path
                modality = candidate
                break

        if modality is None:
            for entry in os.listdir(patient_path):
                entry_path = os.path.join(patient_path, entry)
                if os.path.isdir(entry_path) and entry.lower() in VALID_MODALITIES:
                    load_path = entry_path
                    modality = entry
                    break

    return CrowdsourcingTask(
        campaign_id=campaign_id,
        patient_id=patient_id,
        dataset_path=dataset_path,
        patient_path=patient_path,
        load_path=load_path,
        modality=modality,
    )


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    username = req.username.strip()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    user = AuthManager().authenticate(username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    manager = CrowdsourcingManager()
    manager.load_assignments()

    remaining_tasks: list[CrowdsourcingTask] = []
    assigned_tasks: list[CrowdsourcingTask] = []
    if user["role"] == "expert":
        remaining_tasks = [
            _task_with_paths(task)
            for task in manager.get_remaining_assignments_for_user(user["user_id"])
        ]
        assigned_tasks = [
            _task_with_paths(task)
            for task in manager.get_assigned_tasks(user["user_id"])
        ]

    return LoginResponse(
        success=True,
        user_id=str(user["user_id"]),
        role=str(user["role"]),
        expert_score=float(user["expert_score"]),
        remaining_tasks=remaining_tasks,
        assigned_tasks=assigned_tasks,
        message=(
            f"{len(remaining_tasks)} remaining task(s) found."
            if user["role"] == "expert"
            else "Admin login successful."
        ),
    )
