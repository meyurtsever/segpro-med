"""
SmolVLM Model Download Script for SegMed-Pro

Pre-caches SmolVLM-Instruct from HuggingFace Hub into the local HF cache
so the first inference doesn't require internet access.

Usage:
    python models/smolvlm/download_smolvlm.py
    python models/smolvlm/download_smolvlm.py --force
"""

import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "HuggingFaceTB/SmolVLM-Instruct"


def download_smolvlm(force: bool = False) -> bool:
    """
    Cache SmolVLM via transformers AutoProcessor / snapshot_download.
    SmolVLM is fully public and does not require HF authentication.
    """
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache, constants

        # Quick check — if model already cached, skip unless forced
        if not force:
            cached = try_to_load_from_cache(HF_REPO_ID, "config.json")
            if cached is not None and Path(cached).exists():
                logger.info(f"SmolVLM is already cached ({HF_REPO_ID})")
                return True

    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface-hub")
        return False

    logger.info(f"Downloading SmolVLM ({HF_REPO_ID}) ...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        logger.info("SmolVLM downloaded and cached successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download SmolVLM: {e}")
        return False


def ensure_smolvlm_available(force: bool = False) -> bool:
    """Runtime helper called by smolvlm_service before loading."""
    return download_smolvlm(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SmolVLM-Instruct model")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    args = parser.parse_args()
    success = download_smolvlm(force=args.force)
    sys.exit(0 if success else 1)
