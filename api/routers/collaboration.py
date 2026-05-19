"""
Collaboration Router
====================
Workflow API endpoints for the existing crowdsourcing campaign system.

The storage and domain rules remain owned by ``CrowdsourcingManager``. These
endpoints provide a typed, workflow-friendly surface for creating campaigns,
assigning patients to experts, and checking campaign progress.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auth.auth_manager import AuthManager
from crowdsourcing.campaign_manager import CrowdsourcingManager

router = APIRouter()

AssignmentMode = Literal["selected", "allUnassigned"]
VALID_MODALITIES = ["flair", "t1", "t1c", "t2"]


class CampaignScanRequest(BaseModel):
    dataset_path: str


class CampaignCreateRequest(BaseModel):
    campaign_name: str
    dataset_path: str
    description: str = ""
    overwrite: bool = False


class PatientAssignRequest(BaseModel):
    campaign_name: str
    expert_id: str
    patient_ids: list[str] = []
    assignment_mode: AssignmentMode = "selected"


class AssignmentCompleteRequest(BaseModel):
    campaign_name: str
    expert_id: str
    patient_id: str
    annotation_data: dict[str, Any] | None = None


class CampaignProgress(BaseModel):
    total_patients: int
    assigned_patients: int
    completed: int
    reviewed: int
    unassigned_patients: int


class ExpertAssignment(BaseModel):
    expert_id: str
    assigned_patients: list[str]
    completed_patients: list[str]
    pending_patients: list[str]


class CampaignInfo(BaseModel):
    name: str
    dataset_path: str
    description: str = ""
    created_at: str | None = None
    total_patients: int
    patients: list[str]
    progress: CampaignProgress
    unassigned_patients: list[str]
    assignments: list[ExpertAssignment]


class CampaignScanResponse(BaseModel):
    dataset_path: str
    total_patients: int
    patients: list[str]


class CampaignCreateResponse(BaseModel):
    success: bool
    created: bool
    campaign: CampaignInfo
    message: str


class CampaignListResponse(BaseModel):
    campaigns: list[CampaignInfo]


class AssignmentOptionsResponse(BaseModel):
    campaign: CampaignInfo
    experts: list[str]
    unassigned_patients: list[str]


class PatientAssignResponse(BaseModel):
    success: bool
    campaign_name: str
    expert_id: str
    assigned_patients: list[str]
    assignment_count: int
    campaign: CampaignInfo
    message: str


class ExpertTasksResponse(BaseModel):
    expert_id: str
    tasks: list[dict[str, Any]]


class AssignmentCompleteResponse(BaseModel):
    success: bool
    campaign_name: str
    expert_id: str
    patient_id: str
    campaign: CampaignInfo
    message: str


def _manager() -> CrowdsourcingManager:
    manager = CrowdsourcingManager()
    manager.load_assignments()
    return manager


def _auth_manager() -> AuthManager:
    return AuthManager()


def _progress_model(progress: dict[str, int] | None) -> CampaignProgress:
    value = progress or {}
    return CampaignProgress(
        total_patients=int(value.get("total_patients", 0)),
        assigned_patients=int(value.get("assigned_patients", 0)),
        completed=int(value.get("completed", 0)),
        reviewed=int(value.get("reviewed", 0)),
        unassigned_patients=int(value.get("unassigned_patients", 0)),
    )


def _campaign_info(manager: CrowdsourcingManager, campaign_name: str) -> CampaignInfo:
    campaign = manager.assignments.get(campaign_name)
    if campaign is None:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_name}")

    assigned = campaign.get("assigned", {})
    completed = campaign.get("completed", {})
    assignments: list[ExpertAssignment] = []
    for expert_id, patient_ids in sorted(assigned.items()):
        assigned_patients = list(patient_ids or [])
        completed_patients = list(completed.get(expert_id, []) or [])
        completed_set = set(completed_patients)
        assignments.append(ExpertAssignment(
            expert_id=expert_id,
            assigned_patients=assigned_patients,
            completed_patients=completed_patients,
            pending_patients=[
                patient_id for patient_id in assigned_patients
                if patient_id not in completed_set
            ],
        ))

    return CampaignInfo(
        name=str(campaign.get("name") or campaign_name),
        dataset_path=str(campaign.get("dataset_path") or ""),
        description=str(campaign.get("description") or ""),
        created_at=campaign.get("created_at"),
        total_patients=int(campaign.get("total_patients") or 0),
        patients=list(campaign.get("patients") or []),
        progress=_progress_model(manager.get_campaign_progress(campaign_name)),
        unassigned_patients=manager.get_unassigned_patients(campaign_name),
        assignments=assignments,
    )


def _task_with_paths(task: dict[str, Any]) -> dict[str, Any]:
    campaign_id = str(task.get("campaign_id") or task.get("campaign") or "")
    patient_id = str(task.get("patient_id") or "")
    dataset_path = str(task.get("dataset_path") or "")
    patient_path = os.path.join(dataset_path, patient_id) if dataset_path and patient_id else None
    load_path = patient_path if patient_path and os.path.exists(patient_path) else None
    modality = None

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

    return {
        "campaign_id": campaign_id,
        "patient_id": patient_id,
        "dataset_path": dataset_path,
        "patient_path": patient_path,
        "load_path": load_path,
        "modality": modality,
    }


@router.post("/scan", response_model=CampaignScanResponse)
async def scan_dataset(req: CampaignScanRequest):
    if not req.dataset_path.strip():
        raise HTTPException(status_code=400, detail="dataset_path is required")

    manager = _manager()
    total_patients, patients = manager.scan_dataset(req.dataset_path)
    return CampaignScanResponse(
        dataset_path=req.dataset_path,
        total_patients=total_patients,
        patients=patients,
    )


@router.get("/campaigns", response_model=CampaignListResponse)
async def list_campaigns():
    manager = _manager()
    campaigns = [
        _campaign_info(manager, campaign_name)
        for campaign_name in sorted(manager.assignments.keys())
    ]
    return CampaignListResponse(campaigns=campaigns)


@router.get("/campaigns/{campaign_name}", response_model=CampaignInfo)
async def get_campaign(campaign_name: str):
    return _campaign_info(_manager(), campaign_name)


@router.post("/campaigns", response_model=CampaignCreateResponse)
async def create_campaign(req: CampaignCreateRequest):
    campaign_name = req.campaign_name.strip()
    dataset_path = req.dataset_path.strip()
    if not campaign_name:
        raise HTTPException(status_code=400, detail="campaign_name is required")
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")

    manager = _manager()
    if campaign_name in manager.assignments and not req.overwrite:
        return CampaignCreateResponse(
            success=True,
            created=False,
            campaign=_campaign_info(manager, campaign_name),
            message="Campaign already exists; using the existing campaign.",
        )

    success = manager.create_campaign(
        campaign_name,
        dataset_path,
        description=req.description,
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Campaign creation failed. Check that the dataset path contains valid patient folders.",
        )

    manager.load_assignments()
    return CampaignCreateResponse(
        success=True,
        created=True,
        campaign=_campaign_info(manager, campaign_name),
        message=f"Created campaign '{campaign_name}'.",
    )


@router.get("/campaigns/{campaign_name}/assignment-options", response_model=AssignmentOptionsResponse)
async def get_assignment_options(campaign_name: str):
    manager = _manager()
    campaign = _campaign_info(manager, campaign_name)
    return AssignmentOptionsResponse(
        campaign=campaign,
        experts=_auth_manager().get_experts(),
        unassigned_patients=campaign.unassigned_patients,
    )


@router.post("/assignments", response_model=PatientAssignResponse)
async def assign_patients(req: PatientAssignRequest):
    campaign_name = req.campaign_name.strip()
    expert_id = req.expert_id.strip()
    if not campaign_name:
        raise HTTPException(status_code=400, detail="campaign_name is required")
    if not expert_id:
        raise HTTPException(status_code=400, detail="expert_id is required")

    manager = _manager()
    if campaign_name not in manager.assignments:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_name}")

    patient_ids = [patient_id.strip() for patient_id in req.patient_ids if patient_id.strip()]
    if req.assignment_mode == "allUnassigned":
        patient_ids = manager.get_unassigned_patients(campaign_name)

    if not patient_ids:
        return PatientAssignResponse(
            success=True,
            campaign_name=campaign_name,
            expert_id=expert_id,
            assigned_patients=[],
            assignment_count=0,
            campaign=_campaign_info(manager, campaign_name),
            message="No unassigned patients were available for this assignment.",
        )

    success = manager.assign_patients(campaign_name, expert_id, patient_ids)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Assignment failed. Check that every patient belongs to the campaign.",
        )

    manager.load_assignments()
    return PatientAssignResponse(
        success=True,
        campaign_name=campaign_name,
        expert_id=expert_id,
        assigned_patients=patient_ids,
        assignment_count=len(patient_ids),
        campaign=_campaign_info(manager, campaign_name),
        message=f"Assigned {len(patient_ids)} patient(s) to {expert_id}.",
    )


@router.get("/experts", response_model=list[str])
async def list_experts():
    return _auth_manager().get_experts()


@router.get("/experts/{expert_id}/tasks", response_model=ExpertTasksResponse)
async def get_expert_tasks(expert_id: str, remaining_only: bool = False):
    manager = _manager()
    tasks = (
        manager.get_remaining_assignments_for_user(expert_id)
        if remaining_only
        else manager.get_assigned_tasks(expert_id)
    )
    return ExpertTasksResponse(
        expert_id=expert_id,
        tasks=[_task_with_paths(task) for task in tasks],
    )


@router.post("/complete", response_model=AssignmentCompleteResponse)
async def complete_assignment(req: AssignmentCompleteRequest):
    campaign_name = req.campaign_name.strip()
    expert_id = req.expert_id.strip()
    patient_id = req.patient_id.strip()
    if not campaign_name:
        raise HTTPException(status_code=400, detail="campaign_name is required")
    if not expert_id:
        raise HTTPException(status_code=400, detail="expert_id is required")
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")

    manager = _manager()
    if campaign_name not in manager.assignments:
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_name}")

    success = manager.mark_completed(
        campaign_name,
        expert_id,
        patient_id,
        annotation_data=req.annotation_data,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Assignment completion failed")

    manager.load_assignments()
    return AssignmentCompleteResponse(
        success=True,
        campaign_name=campaign_name,
        expert_id=expert_id,
        patient_id=patient_id,
        campaign=_campaign_info(manager, campaign_name),
        message=f"Marked {patient_id} as completed for {expert_id}.",
    )
