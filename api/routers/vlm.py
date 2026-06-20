"""
VLM Router
==========
REST endpoints for MedGemma, SmolVLM, Med-R1, and label suggestion workflows.

The Gradio app already defines modality-specific prompt presets under
``prompts/*_vlm_prompts.json``. This router reuses those files instead of
duplicating prompt text in the React Flow UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from api.services.session_manager import get_session_manager

SEGPRO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = SEGPRO_ROOT / "models"

if str(SEGPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGPRO_ROOT))
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))
for _model_dir in [
    MODELS_DIR / "medgemma",
    MODELS_DIR / "medgemma-1.5-4b-it",
    MODELS_DIR / "medgemma-1.5-4b-it-GGUF",
    MODELS_DIR / "smolvlm",
    MODELS_DIR / "med-r1",
]:
    if str(_model_dir) not in sys.path:
        sys.path.append(str(_model_dir))

router = APIRouter()
_active_vlm_model: str | None = None

VlmModel = Literal["medgemma", "medgemma-1.5", "medgemma-1.5-gguf", "smolvlm", "med-r1"]

MODEL_LABELS: dict[str, str] = {
    "medgemma": "MedGemma-4B",
    "medgemma-1.5": "MedGemma-1.5-4B",
    "medgemma-1.5-gguf": "MedGemma-1.5-4B GGUF Q8",
    "smolvlm": "SmolVLM",
    "med-r1": "Med-R1",
}

PROMPT_FILES = {
    "MRI": "mri_vlm_prompts.json",
    "CT": "ct_vlm_prompts.json",
    "MG": "mg_vlm_prompts.json",
}

STOP_WORDS = {
    "the", "and", "or", "in", "on", "at", "to", "for", "of", "with", "a",
    "an", "is", "are", "was", "were", "this", "that", "image", "slice",
    "visible", "possible", "labels", "label", "finding", "findings",
}

EXTRA_PROMPTS: dict[str, list[dict[str, Any]]] = {
    "MRI": [
        {
            "key": "annotation_label_candidates",
            "title": "Annotation Label Candidates",
            "description": "Short labels for visible or highlighted MRI regions",
            "prompt": (
                "You are assisting a medical image annotator. Review this brain MRI slice "
                "and propose concise annotation labels for visible structures or highlighted "
                "regions. Prefer standard labels such as frontal lobe, temporal lobe, "
                "ventricle, corpus callosum, cerebellum, brainstem, gray matter, white "
                "matter, CSF, tumor, lesion, hemorrhage, infarct. Return only useful "
                "label candidates as a comma-separated list."
            ),
            "parameters": {"max_tokens": 160, "temperature": 0.2, "focus": "label_candidates"},
        },
        {
            "key": "structured_radiology_review",
            "title": "Structured Radiology Review",
            "description": "Findings, impression, and uncertainty in a compact structure",
            "prompt": (
                "Analyze this brain MRI slice as a radiology assistant. Respond with "
                "three sections: Findings, Impression, and Uncertainty. Mention visible "
                "anatomy, symmetry, image quality, and any suspected abnormality. If a "
                "finding is uncertain, say so explicitly."
            ),
            "parameters": {"max_tokens": 420, "temperature": 0.2, "focus": "structured_review"},
        },
    ],
    "CT": [
        {
            "key": "annotation_label_candidates",
            "title": "Annotation Label Candidates",
            "description": "Short labels for visible or highlighted abdominal CT regions",
            "prompt": (
                "You are assisting a medical image annotator. Review this abdominal CT "
                "slice and propose concise annotation labels for visible structures or "
                "highlighted regions. Prefer organ, vessel, bone, and pathology labels "
                "such as liver, kidney, spleen, pancreas, bowel, aorta, IVC, vertebra, "
                "rib, tumor, cyst, calcification, fluid collection. Return only useful "
                "label candidates as a comma-separated list."
            ),
            "parameters": {"max_tokens": 160, "temperature": 0.2, "focus": "label_candidates"},
        },
    ],
    "MG": [
        {
            "key": "annotation_label_candidates",
            "title": "Annotation Label Candidates",
            "description": "Short labels using breast imaging terminology",
            "prompt": (
                "You are assisting a breast image annotator. Review this mammogram and "
                "propose concise annotation labels for visible structures or highlighted "
                "regions. Prefer BI-RADS terminology where appropriate, such as dense "
                "tissue, fatty tissue, mass, calcifications, microcalcifications, "
                "architectural distortion, asymmetry, nipple, pectoral muscle. Return "
                "only useful label candidates as a comma-separated list."
            ),
            "parameters": {"max_tokens": 160, "temperature": 0.2, "focus": "label_candidates"},
        },
    ],
}


class PointSchema(BaseModel):
    x: float
    y: float


class AnnotationShapeSchema(BaseModel):
    type: str = "polygon"
    points: list[PointSchema] = []
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    radius: float | None = None
    label: str = ""
    color: str = "#4fc3f7"


class VlmPromptPreset(BaseModel):
    key: str
    title: str
    description: str = ""
    prompt: str
    modality: str
    source: str
    parameters: dict[str, Any] = {}


class VlmPromptsResponse(BaseModel):
    modality: str
    prompts: list[VlmPromptPreset]


class VlmModelInfo(BaseModel):
    id: str
    label: str
    description: str
    strengths: list[str]


class VlmModelsResponse(BaseModel):
    models: list[VlmModelInfo]


class VlmAnalysisRequest(BaseModel):
    session_id: str
    slice_index: int = 0
    view: str = "axial"
    model: VlmModel = "medgemma-1.5-gguf"
    modality: str = "MRI"
    prompt_key: str = "describe_slice"
    custom_prompt: str | None = None
    max_tokens: int | None = Field(default=None, ge=32, le=1024)
    include_reasoning: bool = False
    annotations: list[AnnotationShapeSchema] = []
    use_overlay: bool = False


class VlmAnalysisResponse(BaseModel):
    success: bool
    model: str
    model_label: str
    modality: str
    prompt_key: str
    prompt_title: str
    prompt_used: str
    session_id: str
    slice_index: int
    view: str
    text: str
    labels: list[str] = []
    elapsed_seconds: float


class VlmLabelSuggestRequest(VlmAnalysisRequest):
    prompt_key: str = "annotation_label_candidates"
    current_labels: list[str] = []
    max_labels: int = Field(default=12, ge=1, le=32)
    use_overlay: bool = True


class VlmLabelSuggestResponse(VlmAnalysisResponse):
    labels: list[str]


def _normalize_modality(modality: str | None) -> str:
    value = (modality or "MRI").strip().upper()
    if value in {"MAMMO", "MAMMOGRAPHY"}:
        return "MG"
    if value not in PROMPT_FILES:
        return "MRI"
    return value


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()


def _load_prompt_presets(modality: str) -> list[VlmPromptPreset]:
    normalized = _normalize_modality(modality)
    prompt_path = SEGPRO_ROOT / "prompts" / PROMPT_FILES[normalized]
    presets: list[VlmPromptPreset] = []

    if prompt_path.exists():
        with prompt_path.open("r", encoding="utf-8") as handle:
            raw_prompts = json.load(handle)

        for key, value in raw_prompts.items():
            if isinstance(value, str):
                presets.append(VlmPromptPreset(
                    key=key,
                    title=_title_from_key(key),
                    description="Gradio prompt preset",
                    prompt=value,
                    modality=normalized,
                    source="gradio",
                    parameters={},
                ))
            elif isinstance(value, dict) and isinstance(value.get("prompt"), str):
                presets.append(VlmPromptPreset(
                    key=key,
                    title=str(value.get("title") or _title_from_key(key)),
                    description=str(value.get("description") or ""),
                    prompt=str(value["prompt"]),
                    modality=normalized,
                    source="gradio",
                    parameters=dict(value.get("parameters") or {}),
                ))

    for extra in EXTRA_PROMPTS.get(normalized, []):
        presets.append(VlmPromptPreset(
            key=str(extra["key"]),
            title=str(extra["title"]),
            description=str(extra.get("description") or ""),
            prompt=str(extra["prompt"]),
            modality=normalized,
            source="workflow",
            parameters=dict(extra.get("parameters") or {}),
        ))

    return presets


def _resolve_prompt(req: VlmAnalysisRequest) -> tuple[str, str, str, dict[str, Any]]:
    if req.custom_prompt and req.custom_prompt.strip():
        return req.custom_prompt.strip(), "custom", "Custom Prompt", {}

    presets = _load_prompt_presets(req.modality)
    prompt = next((preset for preset in presets if preset.key == req.prompt_key), None)
    if prompt is None:
        prompt = next((preset for preset in presets if preset.key == "describe_slice"), None)
    if prompt is None and presets:
        prompt = presets[0]
    if prompt is None:
        raise HTTPException(status_code=500, detail="No VLM prompts are available")

    return prompt.prompt, prompt.key, prompt.title, prompt.parameters


def _extract_slice(volume: np.ndarray, slice_index: int, view: str, axis_map: dict | None = None) -> np.ndarray:
    if axis_map is None:
        axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = axis_map.get(view, next(iter(axis_map.values()), 2))
    vol = volume[..., 0] if volume.ndim == 4 else volume
    total = vol.shape[axis] if vol.ndim >= 3 else 1
    idx = max(0, min(slice_index, total - 1))
    return np.take(vol, idx, axis=axis).astype(np.float32) if vol.ndim >= 3 else vol.astype(np.float32)


def _slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx > mn:
        arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        arr = np.zeros(arr.shape, dtype=np.uint8)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    raise ValueError(f"Unsupported slice shape: {arr.shape}")


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = (color or "#4fc3f7").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (79, 195, 247)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (79, 195, 247)


def _draw_annotations(image_array: np.ndarray, annotations: list[AnnotationShapeSchema]) -> Image.Image:
    image = Image.fromarray(image_array).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    base_draw = ImageDraw.Draw(image)

    for index, annotation in enumerate(annotations, start=1):
        color = _hex_to_rgb(annotation.color)
        fill = color + (92,)
        outline = color + (255,)
        label = annotation.label or f"Region {index}"

        if annotation.points:
            points = [(float(point.x), float(point.y)) for point in annotation.points]
            if len(points) >= 3:
                overlay_draw.polygon(points, fill=fill)
                base_draw.line(points + [points[0]], fill=outline, width=2)
        elif annotation.type == "rect" and annotation.x is not None and annotation.y is not None:
            x1 = float(annotation.x)
            y1 = float(annotation.y)
            x2 = x1 + float(annotation.width or 0)
            y2 = y1 + float(annotation.height or 0)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)
        elif annotation.type == "circle" and annotation.x is not None and annotation.y is not None:
            radius = float(annotation.radius or 6)
            x = float(annotation.x)
            y = float(annotation.y)
            overlay_draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=outline, width=2)
        elif annotation.type == "point" and annotation.x is not None and annotation.y is not None:
            x = float(annotation.x)
            y = float(annotation.y)
            overlay_draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=outline)

        if annotation.x is not None and annotation.y is not None:
            base_draw.text((float(annotation.x) + 4, float(annotation.y) + 4), label, fill=outline)

    return Image.alpha_composite(image, overlay).convert("RGB")


def _annotation_context(annotations: list[AnnotationShapeSchema]) -> str:
    if not annotations:
        return ""

    descriptions = []
    for index, annotation in enumerate(annotations, start=1):
        label = annotation.label or "unlabeled"
        if annotation.points:
            descriptions.append(f"Region {index}: {annotation.type} annotation '{label}' with {len(annotation.points)} points")
        elif annotation.type == "rect":
            descriptions.append(f"Region {index}: rectangle annotation '{label}'")
        else:
            descriptions.append(f"Region {index}: {annotation.type} annotation '{label}'")

    return (
        "\n\nThe image contains visible annotation overlays. "
        "Analyze the highlighted regions specifically: " + "; ".join(descriptions)
    )


def _prepare_image(req: VlmAnalysisRequest) -> Image.Image:
    mgr = get_session_manager()
    session = mgr.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")
    if session.volume is None:
        raise HTTPException(status_code=400, detail="Session has no loaded volume")

    try:
        axis_map = session.metadata.get("axis_map") or None
        slice_2d = _extract_slice(session.volume, req.slice_index, req.view, axis_map)
        rgb = _slice_to_rgb(slice_2d)
        if req.use_overlay and req.annotations:
            return _draw_annotations(rgb, req.annotations)
        return Image.fromarray(rgb).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Slice preparation failed: {exc}") from exc


def _with_med_r1_reasoning(prompt: str, include_reasoning: bool) -> str:
    if not include_reasoning:
        return prompt
    lower = prompt.lower()
    if "<think>" in lower or "thinking process" in lower:
        return prompt
    return (
        f"{prompt} First output the thinking process in <think> </think> tags "
        "and final answer in <answer> </answer> tags."
    )


def _save_temp_image(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as handle:
        path = handle.name
    image.convert("RGB").save(path, "JPEG", quality=95)
    return path


def _release_services_on_model_change(model: str) -> None:
    global _active_vlm_model

    if _active_vlm_model == model:
        return

    try:
        from medgemma.medgemma_service import cleanup_service as cleanup_transformers_medgemma
    except Exception:
        cleanup_transformers_medgemma = None

    try:
        from medgemma.medgemma_gguf_service import cleanup_service as cleanup_gguf
    except Exception:
        cleanup_gguf = None

    try:
        from smolvlm.smolvlm_service import cleanup_service as cleanup_smolvlm
    except Exception:
        cleanup_smolvlm = None

    try:
        from med_r1_service import cleanup_service as cleanup_med_r1
    except Exception:
        cleanup_med_r1 = None

    if cleanup_transformers_medgemma is not None:
        cleanup_transformers_medgemma()
    if cleanup_gguf is not None:
        cleanup_gguf()
    if cleanup_smolvlm is not None:
        cleanup_smolvlm()
    if cleanup_med_r1 is not None:
        cleanup_med_r1()

    _active_vlm_model = model


def _run_model(model: str, image: Image.Image, prompt: str, max_tokens: int) -> str:
    _release_services_on_model_change(model)

    if model == "medgemma":
        try:
            from medgemma.medgemma_service import get_report_service
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"MedGemma service is not available: {exc}") from exc
        service = get_report_service(device="auto")
        return service.generate_response(image=image, prompt=prompt, max_new_tokens=max_tokens)

    if model == "medgemma-1.5":
        try:
            from medgemma.medgemma_service import get_medgemma15_service
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"MedGemma 1.5 service is not available: {exc}") from exc
        service = get_medgemma15_service(device="auto")
        return service.generate_response(image=image, prompt=prompt, max_new_tokens=max_tokens)

    if model == "medgemma-1.5-gguf":
        try:
            from medgemma.medgemma_gguf_service import get_service
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"MedGemma GGUF service is not available: {exc}") from exc
        try:
            return get_service().generate_response(image=image, prompt=prompt, max_new_tokens=max_tokens)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if model == "smolvlm":
        try:
            from smolvlm.smolvlm_service import get_service
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"SmolVLM service is not available: {exc}") from exc
        image_path = _save_temp_image(image)
        try:
            result = get_service(device="auto").generate_caption(image_path, prompt, max_tokens=max_tokens)
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error") or "SmolVLM inference failed")
        return str(result.get("caption") or "")

    if model == "med-r1":
        med_r1_dir = MODELS_DIR / "med-r1"
        if str(med_r1_dir) not in sys.path:
            sys.path.append(str(med_r1_dir))
        try:
            from med_r1_service import get_service
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Med-R1 service is not available: {exc}") from exc
        image_path = _save_temp_image(image)
        try:
            result = get_service(device="auto").generate_caption(image_path, prompt, max_tokens=max_tokens)
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error") or "Med-R1 inference failed")
        return str(result.get("caption") or "")

    raise HTTPException(status_code=400, detail=f"Unsupported VLM model: {model}")


def _strip_reasoning_blocks(text: str) -> str:
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _parse_labels(text: str, current_labels: list[str] | None = None, max_labels: int = 12) -> list[str]:
    cleaned_text = _strip_reasoning_blocks(text)
    candidates: list[str] = []

    region_matches = re.findall(
        r"(?:Region|ROI)\s*\d+\s*[:\-]\s*([A-Za-z][A-Za-z0-9 /_\-]{1,48})",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    candidates.extend(region_matches)

    bullet_matches = re.findall(
        r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*([A-Za-z][A-Za-z0-9 /_\-]{1,48})",
        cleaned_text,
    )
    candidates.extend(bullet_matches)

    if "," in cleaned_text:
        candidates.extend(cleaned_text.split(","))

    if not candidates:
        candidates.extend(re.findall(r"\b[A-Za-z][A-Za-z ]{2,28}\b", cleaned_text))

    current = {label.lower().strip() for label in (current_labels or [])}
    labels: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        value = re.sub(r"\s+", " ", candidate.strip(" .;:()[]{}\"'\n\t"))
        value = re.sub(r"\b(confidence|location|clinical significance)\b.*$", "", value, flags=re.IGNORECASE).strip()
        if not value or len(value) < 2 or len(value) > 40:
            continue
        normalized = value.lower()
        if normalized in STOP_WORDS or normalized in current or normalized in seen:
            continue
        if len(normalized.split()) > 5:
            continue
        seen.add(normalized)
        labels.append(value)
        if len(labels) >= max_labels:
            break

    return labels


@router.get("/models", response_model=VlmModelsResponse)
async def list_models():
    return VlmModelsResponse(models=[
        VlmModelInfo(
            id="medgemma",
            label="MedGemma-4B",
            description="Medical VLM for radiology-style image analysis and label suggestions.",
            strengths=["medical terminology", "structured findings", "label suggestions"],
        ),
        VlmModelInfo(
            id="medgemma-1.5",
            label="MedGemma-1.5-4B",
            description="Local Transformers MedGemma 1.5 model for faster deterministic report generation.",
            strengths=["newer MedGemma checkpoint", "CUDA BF16 inference", "structured findings"],
        ),
        VlmModelInfo(
            id="medgemma-1.5-gguf",
            label="MedGemma-1.5 GGUF Q8",
            description="High-quality quantized GGUF model. Requires llama-cpp-python with GPU support.",
            strengths=["smaller model artifact", "llama.cpp backend ready", "quality-preserving Q8 quantization"],
        ),
        VlmModelInfo(
            id="smolvlm",
            label="SmolVLM",
            description="Lightweight VLM for fast captioning and first-pass slice descriptions.",
            strengths=["fast captioning", "low overhead", "general description"],
        ),
        VlmModelInfo(
            id="med-r1",
            label="Med-R1",
            description="Reasoning-oriented medical VLM, especially useful for MRI review.",
            strengths=["reasoning traces", "MRI analysis", "region label review"],
        ),
    ])


@router.get("/prompts", response_model=VlmPromptsResponse)
async def list_prompts(modality: str = "MRI"):
    normalized = _normalize_modality(modality)
    return VlmPromptsResponse(
        modality=normalized,
        prompts=_load_prompt_presets(normalized),
    )


@router.post("/analyze", response_model=VlmAnalysisResponse)
async def analyze(req: VlmAnalysisRequest):
    t0 = time.time()
    normalized_modality = _normalize_modality(req.modality)
    prompt, prompt_key, prompt_title, parameters = _resolve_prompt(req)

    if req.annotations and req.use_overlay:
        prompt += _annotation_context(req.annotations)
    if req.model == "med-r1":
        prompt = _with_med_r1_reasoning(prompt, req.include_reasoning)

    image = _prepare_image(req)
    max_tokens = int(req.max_tokens or parameters.get("max_tokens") or 256)
    max_tokens = max(32, min(max_tokens, 1024))
    text = _run_model(req.model, image, prompt, max_tokens)

    return VlmAnalysisResponse(
        success=True,
        model=req.model,
        model_label=MODEL_LABELS[req.model],
        modality=normalized_modality,
        prompt_key=prompt_key,
        prompt_title=prompt_title,
        prompt_used=prompt,
        session_id=req.session_id,
        slice_index=req.slice_index,
        view=req.view,
        text=text,
        labels=_parse_labels(text, max_labels=12) if prompt_key == "suggest_labels" else [],
        elapsed_seconds=round(time.time() - t0, 3),
    )


@router.post("/labels", response_model=VlmLabelSuggestResponse)
async def suggest_labels(req: VlmLabelSuggestRequest):
    t0 = time.time()
    normalized_modality = _normalize_modality(req.modality)
    prompt, prompt_key, prompt_title, parameters = _resolve_prompt(req)

    if req.annotations:
        prompt += _annotation_context(req.annotations)
    if req.model == "med-r1":
        prompt = _with_med_r1_reasoning(prompt, req.include_reasoning)

    image = _prepare_image(req)
    max_tokens = int(req.max_tokens or parameters.get("max_tokens") or 160)
    max_tokens = max(32, min(max_tokens, 1024))
    text = _run_model(req.model, image, prompt, max_tokens)
    labels = _parse_labels(text, req.current_labels, req.max_labels)

    return VlmLabelSuggestResponse(
        success=True,
        model=req.model,
        model_label=MODEL_LABELS[req.model],
        modality=normalized_modality,
        prompt_key=prompt_key,
        prompt_title=prompt_title,
        prompt_used=prompt,
        session_id=req.session_id,
        slice_index=req.slice_index,
        view=req.view,
        text=text,
        labels=labels,
        elapsed_seconds=round(time.time() - t0, 3),
    )
