"""
Med-R1 Model Download Script for SegMed-Pro

Downloads the Med-R1 VLM checkpoint from HuggingFace Hub.
The model is based on Qwen2-VL, fine-tuned for medical image analysis.

Usage:
    python models/med-r1/download_med_r1.py
    python models/med-r1/download_med_r1.py --force
"""

import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / "MRI"
HF_REPO_ID = "yuxianglai117/Med-R1"

REQUIRED_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
]


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    if not checkpoint_dir.exists():
        return False
    for f in REQUIRED_FILES:
        if not (checkpoint_dir / f).exists():
            return False
    # Also check at least one safetensors shard exists
    shards = list(checkpoint_dir.glob("*.safetensors"))
    return len(shards) > 0


def download_med_r1(force: bool = False) -> bool:
    """Download Med-R1 from HuggingFace Hub via snapshot_download."""
    if not force and is_checkpoint_complete(CHECKPOINT_DIR):
        logger.info(f"Med-R1 checkpoint already present at {CHECKPOINT_DIR}")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface-hub")
        return False

    logger.info(f"Downloading Med-R1 from {HF_REPO_ID} to {CHECKPOINT_DIR} ...")
    logger.info("This is a multi-GB model — please be patient.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(CHECKPOINT_DIR),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        logger.info(f"Med-R1 downloaded successfully to {CHECKPOINT_DIR}")
        return True
    except Exception as e:
        logger.error(f"Failed to download Med-R1: {e}")
        logger.info(
            "If the model requires authentication, run:\n"
            "  huggingface-cli login\n"
            "or set the HF_TOKEN environment variable."
        )
        return False


def ensure_med_r1_available(force: bool = False) -> bool:
    """Runtime helper — auto-download if checkpoint is missing."""
    if is_checkpoint_complete(CHECKPOINT_DIR):
        return True
    logger.info("Med-R1 checkpoint not found — starting automatic download ...")
    return download_med_r1(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Med-R1 model checkpoint")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()
    success = download_med_r1(force=args.force)
    sys.exit(0 if success else 1)
