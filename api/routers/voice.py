"""
Voice Router
============
Workflow API endpoints for voice prompts used by medical VLM nodes.

The Gradio editor already treats voice as a prompt source: it transcribes the
audio, infers whether the user is asking for anomaly detection, slice
description, or both, then routes that text into VLM analysis. This router keeps
that same behavior available to React Flow nodes.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from utils.voice_input import transcribe_voice_input

router = APIRouter()

VoiceIntent = Literal["describe", "anomaly", "both", "custom"]

ANOMALY_KEYWORDS = {
    "anomaly",
    "abnormal",
    "lesion",
    "tumor",
    "tumour",
    "pathology",
    "disease",
    "mass",
    "hemorrhage",
    "haemorrhage",
    "infarct",
}

DESCRIBE_KEYWORDS = {
    "describe",
    "anatomy",
    "structure",
    "region",
    "what is",
    "what do you see",
    "visible",
    "landmark",
}


class VoiceIntentRequest(BaseModel):
    transcript: str = Field(..., description="Transcribed or typed voice prompt text.")


class VoiceTranscribePathRequest(BaseModel):
    audio_path: str = Field(..., description="Local audio path on the API host.")


class VoicePromptResponse(BaseModel):
    success: bool
    transcript: str
    intent: VoiceIntent
    prompt_key: str
    identify_anomalies: bool
    describe_slice: bool
    source: str
    message: str


def infer_voice_intent(transcript: str) -> tuple[VoiceIntent, str, bool, bool]:
    """Infer VLM prompt intent using the same keyword style as the Gradio app."""
    text = transcript.strip()
    if not text:
        return "custom", "describe_slice", False, False

    lower = text.lower()
    identify_anomalies = any(keyword in lower for keyword in ANOMALY_KEYWORDS)
    describe_slice = any(keyword in lower for keyword in DESCRIBE_KEYWORDS)

    if identify_anomalies and describe_slice:
        return "both", "structured_radiology_review", True, True
    if identify_anomalies:
        return "anomaly", "identify_anomalies", True, False
    if describe_slice:
        return "describe", "describe_slice", False, True

    # Gradio defaults to both modes when no specific intent is detected.
    return "both", "structured_radiology_review", True, True


def _response(transcript: str, source: str) -> VoicePromptResponse:
    cleaned = transcript.strip()
    intent, prompt_key, identify_anomalies, describe_slice = infer_voice_intent(cleaned)
    return VoicePromptResponse(
        success=True,
        transcript=cleaned,
        intent=intent,
        prompt_key=prompt_key,
        identify_anomalies=identify_anomalies,
        describe_slice=describe_slice,
        source=source,
        message=f"Voice prompt ready as {intent} intent.",
    )


def _normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def _read_wave(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def _load_audio_file(path: str) -> tuple[np.ndarray, int]:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Audio path not found: {path}")
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"Audio path is not a file: {path}")

    errors: list[str] = []

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32")
        return _normalize_audio_array(np.asarray(audio)), int(sample_rate)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")

    try:
        import librosa

        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        return _normalize_audio_array(np.asarray(audio)), int(sample_rate)
    except Exception as exc:
        errors.append(f"librosa: {exc}")

    try:
        from scipy.io import wavfile

        sample_rate, audio = wavfile.read(path)
        return _normalize_audio_array(np.asarray(audio)), int(sample_rate)
    except Exception as exc:
        errors.append(f"scipy wavfile: {exc}")

    if Path(path).suffix.lower() == ".wav":
        try:
            return _read_wave(path)
        except Exception as exc:
            errors.append(f"wave: {exc}")

    raise HTTPException(
        status_code=415,
        detail=(
            "Unable to decode audio file. Use WAV/FLAC/MP3 with soundfile/librosa "
            f"available. Decoder errors: {'; '.join(errors[-3:])}"
        ),
    )


def _transcribe_audio_path(path: str, source: str) -> VoicePromptResponse:
    audio, sample_rate = _load_audio_file(path)
    transcript = transcribe_voice_input(audio, sample_rate=sample_rate)
    if not transcript or transcript.startswith("Error"):
        raise HTTPException(status_code=503, detail=transcript or "Voice transcription failed")
    if transcript.lower().startswith("transcription error"):
        raise HTTPException(status_code=503, detail=transcript)
    return _response(transcript, source=source)


@router.post("/intent", response_model=VoicePromptResponse)
async def infer_intent(req: VoiceIntentRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is required")
    return _response(req.transcript, source="typed")


@router.post("/transcribe-path", response_model=VoicePromptResponse)
async def transcribe_path(req: VoiceTranscribePathRequest):
    if not req.audio_path.strip():
        raise HTTPException(status_code=400, detail="audio_path is required")
    return _transcribe_audio_path(req.audio_path, source="audio_path")


@router.post("/transcribe", response_model=VoicePromptResponse)
async def transcribe_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = handle.name
        handle.write(await file.read())

    try:
        return _transcribe_audio_path(temp_path, source="upload")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
