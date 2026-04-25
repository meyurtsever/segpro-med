"""
MedSAM2 Checkpoint Download Script for SegMed-Pro

Downloads MedSAM2 checkpoints and the SAM2.1 backbone from HuggingFace Hub.
Checkpoints are saved to models/medsam2/checkpoints/.

Usage:
    python models/medsam2/download_medsam2.py
    python models/medsam2/download_medsam2.py --model latest
    python models/medsam2/download_medsam2.py --model all
"""

import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

# HuggingFace repo for MedSAM2
MEDSAM2_HF_REPO = "bowang-lab/MedSAM2"

# SAM2.1 backbone hosted on HuggingFace by Meta
SAM2_BACKBONE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

AVAILABLE_CHECKPOINTS = {
    "latest": "MedSAM2_latest.pt",
    "2411": "MedSAM2_2411.pt",
    "ct_lesion": "MedSAM2_CTLesion.pt",
    "mri_liver": "MedSAM2_MRI_LiverLesion.pt",
    "us_heart": "MedSAM2_US_Heart.pt",
}


def _ensure_huggingface_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)


def check_checkpoint_exists(filename: str) -> bool:
    path = CHECKPOINTS_DIR / filename
    if path.exists() and path.stat().st_size > 1024 * 1024:  # > 1 MB sanity check
        return True
    return False


def download_sam2_backbone(force: bool = False) -> bool:
    """Download SAM2.1 hiera base+ backbone checkpoint"""
    filename = "sam2.1_hiera_base_plus.pt"
    dest = CHECKPOINTS_DIR / filename

    if not force and check_checkpoint_exists(filename):
        logger.info(f"SAM2.1 backbone already exists at {dest}")
        return True

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        from tqdm import tqdm

        logger.info(f"Downloading SAM2.1 backbone from {SAM2_BACKBONE_URL} ...")
        response = requests.get(SAM2_BACKBONE_URL, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info(f"SAM2.1 backbone saved to {dest}")
        return True

    except Exception as e:
        logger.warning(f"Direct download failed ({e}), trying HuggingFace Hub fallback ...")
        try:
            hf = _ensure_huggingface_hub()
            path = hf.hf_hub_download(
                repo_id="facebook/sam2.1-hiera-base-plus",
                filename=filename,
                local_dir=str(CHECKPOINTS_DIR),
            )
            logger.info(f"SAM2.1 backbone downloaded to {path}")
            return True
        except Exception as hf_err:
            logger.error(f"HuggingFace fallback also failed: {hf_err}")
            return False


def download_medsam2_checkpoint(model_key: str = "latest", force: bool = False) -> bool:
    """
    Download a specific MedSAM2 checkpoint from HuggingFace Hub.

    Args:
        model_key: One of 'latest', '2411', 'ct_lesion', 'mri_liver', 'us_heart'
        force: Re-download even if file exists
    """
    if model_key not in AVAILABLE_CHECKPOINTS:
        logger.error(f"Unknown model key '{model_key}'. Available: {list(AVAILABLE_CHECKPOINTS.keys())}")
        return False

    filename = AVAILABLE_CHECKPOINTS[model_key]

    if not force and check_checkpoint_exists(filename):
        logger.info(f"{filename} already exists at {CHECKPOINTS_DIR / filename}")
        return True

    hf = _ensure_huggingface_hub()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {filename} from HuggingFace Hub ({MEDSAM2_HF_REPO}) ...")
    try:
        path = hf.hf_hub_download(
            repo_id=MEDSAM2_HF_REPO,
            filename=filename,
            local_dir=str(CHECKPOINTS_DIR),
        )
        logger.info(f"Checkpoint saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        logger.info(
            "If the repository requires authentication, run:\n"
            "  huggingface-cli login\n"
            "or set the HF_TOKEN environment variable."
        )
        return False


def download_all(force: bool = False) -> bool:
    """Download all MedSAM2 checkpoints and the SAM2.1 backbone."""
    success = True
    success &= download_sam2_backbone(force=force)
    for key in AVAILABLE_CHECKPOINTS:
        success &= download_medsam2_checkpoint(key, force=force)
    return success


def ensure_default_checkpoint_available(force: bool = False) -> bool:
    """
    Called at runtime to ensure at least the default checkpoint is present.
    Downloads the latest model if missing.
    """
    if check_checkpoint_exists(AVAILABLE_CHECKPOINTS["latest"]):
        return True

    logger.info("MedSAM2 default checkpoint not found — starting automatic download ...")
    ok = download_sam2_backbone(force=force)
    ok &= download_medsam2_checkpoint("latest", force=force)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MedSAM2 model checkpoints")
    parser.add_argument(
        "--model",
        default="latest",
        choices=list(AVAILABLE_CHECKPOINTS.keys()) + ["all", "backbone"],
        help="Which checkpoint(s) to download (default: latest)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    if args.model == "all":
        success = download_all(force=args.force)
    elif args.model == "backbone":
        success = download_sam2_backbone(force=args.force)
    else:
        success = download_sam2_backbone(force=args.force)
        success &= download_medsam2_checkpoint(args.model, force=args.force)

    sys.exit(0 if success else 1)
