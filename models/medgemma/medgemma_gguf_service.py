"""Persistent llama.cpp backend for MedGemma 1.5 GGUF multimodal inference."""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_gguf_service = None
_dll_directory_handles = []


def _add_windows_dll_dirs() -> None:
    """Expose llama.cpp and CUDA DLL folders when the API is launched normally."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    package_lib = Path(__file__).resolve().parents[2] / ".conda" / "Lib" / "site-packages" / "llama_cpp" / "lib"
    cuda_roots = [
        Path(os.environ["CUDA_PATH"]) if os.environ.get("CUDA_PATH") else None,
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"),
    ]
    dll_dirs = [package_lib]
    for root in cuda_roots:
        if root is None:
            continue
        dll_dirs.extend([root / "bin" / "x64", root / "bin"])

    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                _dll_directory_handles.append(os.add_dll_directory(str(dll_dir)))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                logger.debug("Could not add DLL directory %s", dll_dir, exc_info=True)

    if not os.environ.get("CUDA_PATH"):
        default_cuda = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
        if default_cuda.exists():
            os.environ["CUDA_PATH"] = str(default_cuda)


class MedGemmaGgufService:
    """llama.cpp service for MedGemma 1.5 GGUF multimodal inference."""

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "models"
        gguf_dir = models_dir / "medgemma-1.5-4b-it-GGUF"
        self.model_path = gguf_dir / "medgemma-1.5-4b-it-UD-Q8_K_XL.gguf"
        self.mmproj_path = gguf_dir / "mmproj-BF16.gguf"
        self.cli_path = Path(
            os.environ.get("LLAMA_MTMD_CLI_PATH", "C:/tmp/llama.cpp/build-cuda/bin/llama-mtmd-cli.exe")
        )
        self.server_path = Path(
            os.environ.get("LLAMA_SERVER_PATH", "C:/tmp/llama.cpp/build-cuda/bin/llama-server.exe")
        )
        self.host = os.environ.get("MEDGEMMA_GGUF_HOST", "127.0.0.1")
        self.port = int(os.environ.get("MEDGEMMA_GGUF_PORT", "8091"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.process: subprocess.Popen | None = None

        if not self.model_path.exists():
            raise RuntimeError(f"GGUF model not found: {self.model_path}")
        if not self.mmproj_path.exists():
            raise RuntimeError(f"GGUF multimodal projector not found: {self.mmproj_path}")
        if not self.server_path.exists() and not self.cli_path.exists():
            raise RuntimeError(
                "CUDA llama.cpp backend not found. Build llama.cpp with CUDA or set LLAMA_SERVER_PATH. "
                f"Expected server: {self.server_path}"
            )

        _add_windows_dll_dirs()
        atexit.register(self.shutdown)
        logger.info("Using persistent llama.cpp server backend at %s", self.server_path)

    @staticmethod
    def _image_to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

        arr = image
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    def generate_response(self, image: Union[Image.Image, np.ndarray], prompt: str, max_new_tokens: int = 256) -> str:
        image_pil = self._image_to_pil(image)
        self._ensure_server()

        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt_string = self._format_medgemma_prompt(prompt)
        payload = {
            "prompt": {
                "prompt_string": prompt_string,
                "multimodal_data": [image_data],
            },
            "n_predict": max_new_tokens,
            "temperature": 0,
            "top_k": 1,
        }
        response = self._post_json("/completions", payload, timeout=max(120, max_new_tokens * 4))
        if "content" in response:
            return str(response["content"]).strip()
        return str(response["choices"][0].get("text") or response["choices"][0]["message"]["content"]).strip()

    @staticmethod
    def _format_medgemma_prompt(prompt: str) -> str:
        """Mirror the Transformers MedGemma chat template for llama.cpp GGUF calls."""
        clean_prompt = prompt.strip()
        return (
            "<bos>\n"
            "<start_of_turn>user\n"
            "You are an expert radiologist.\n\n"
            "<__media__>\n"
            f"{clean_prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def _server_env(self) -> dict[str, str]:
        env = os.environ.copy()
        backend_bin = str(self.server_path.parent if self.server_path.exists() else self.cli_path.parent)
        cuda_root = env.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
        cuda_bin_x64 = str(Path(cuda_root) / "bin" / "x64")
        cuda_bin = str(Path(cuda_root) / "bin")
        env["PATH"] = os.pathsep.join([backend_bin, cuda_bin_x64, cuda_bin, env.get("PATH", "")])
        env.setdefault("LLAMA_MEDIA_MARKER", "<__media__>")
        return env

    def _ensure_server(self) -> None:
        if self._is_healthy():
            return
        if not self.server_path.exists():
            raise RuntimeError(f"llama-server.exe not found: {self.server_path}")

        command = [
            str(self.server_path),
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            "all",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "--temp",
            "0",
            "--chat-template",
            "gemma",
            "--no-ui",
            "--log-verbosity",
            "1",
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            env=self._server_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        deadline = time.time() + 180
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {self.process.returncode}")
            if self._is_healthy():
                logger.info("llama.cpp GGUF server is ready at %s", self.base_url)
                return
            time.sleep(1)
        raise RuntimeError(f"llama-server did not become healthy at {self.base_url}")

    def _is_healthy(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(request, timeout=2) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def _post_json(self, path: str, payload: dict, timeout: int) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server request failed with HTTP {exc.code}: {detail}") from exc

    def shutdown(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def get_service() -> MedGemmaGgufService:
    global _gguf_service
    if _gguf_service is None:
        _gguf_service = MedGemmaGgufService()
    return _gguf_service


def cleanup_service() -> None:
    global _gguf_service
    if _gguf_service is not None:
        _gguf_service.shutdown()
    _gguf_service = None
