"""
SegMed-Pro Model Download Script
=================================
Downloads all AI model checkpoints needed by SegMed-Pro.

Models downloaded:
  - MedSAM2        — bowang-lab/MedSAM2          (interactive medical image segmentation)
  - SAM2.1 backbone— facebook/sam2.1-hiera-base+  (backbone for MedSAM2 and MG)
  - SmolVLM        — HuggingFaceTB/SmolVLM-Instruct (lightweight VLM, no auth required)
  - Med-R1         — yuxianglai117/Med-R1          (Qwen2-VL based medical VLM)
  - MedGemma-4B    — google/medgemma-4b-it         (requires HF authentication + model access)

Usage:
    python download_models.py                  # download all models
    python download_models.py --model medsam2  # specific model only
    python download_models.py --model smolvlm med-r1
    python download_models.py --skip medgemma  # skip gated models
    python download_models.py --force          # re-download everything

Authentication:
    MedGemma requires a HuggingFace account with access granted at:
    https://huggingface.co/google/medgemma-4b-it

    To authenticate:
        huggingface-cli login
    or set environment variable:
        export HF_TOKEN=hf_your_token_here
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent


def _add_model_paths():
    """Add model subdirectories to sys.path so download scripts can import their helpers."""
    for subdir in ["medsam2", "med-r1", "smolvlm", "medgemma"]:
        p = REPO_ROOT / "models" / subdir
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def download_medsam2(force: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Downloading MedSAM2 checkpoints ...")
    logger.info("=" * 60)
    try:
        _add_model_paths()
        from models.medsam2.download_medsam2 import (
            download_sam2_backbone,
            download_medsam2_checkpoint,
        )
        ok = download_sam2_backbone(force=force)
        ok &= download_medsam2_checkpoint("latest", force=force)
        return ok
    except Exception as e:
        logger.error(f"MedSAM2 download failed: {e}")
        return False


def download_smolvlm(force: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Downloading SmolVLM-Instruct ...")
    logger.info("=" * 60)
    try:
        _add_model_paths()
        from models.smolvlm.download_smolvlm import download_smolvlm as _dl
        return _dl(force=force)
    except Exception as e:
        logger.error(f"SmolVLM download failed: {e}")
        return False


def download_med_r1(force: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Downloading Med-R1 VLM ...")
    logger.info("=" * 60)
    try:
        _add_model_paths()
        from models.med_r1.download_med_r1 import download_med_r1 as _dl  # noqa: F401
        return _dl(force=force)
    except ModuleNotFoundError:
        # Try direct import with path adjustment
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "download_med_r1",
            REPO_ROOT / "models" / "med-r1" / "download_med_r1.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.download_med_r1(force=force)
    except Exception as e:
        logger.error(f"Med-R1 download failed: {e}")
        return False


def download_medgemma(force: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Downloading MedGemma-4B ...")
    logger.info("=" * 60)
    logger.info(
        "NOTE: MedGemma requires HuggingFace authentication and model access.\n"
        "      Visit https://huggingface.co/google/medgemma-4b-it to request access.\n"
        "      Then run: huggingface-cli login"
    )
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "download_medgemma",
            REPO_ROOT / "models" / "medgemma" / "download_medgemma.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.download_medgemma_model(force_redownload=force)
    except Exception as e:
        logger.error(f"MedGemma download failed: {e}")
        return False


ALL_MODELS = {
    "medsam2": download_medsam2,
    "smolvlm": download_smolvlm,
    "med-r1": download_med_r1,
    "medgemma": download_medgemma,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download all SegMed-Pro model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="*",
        choices=list(ALL_MODELS.keys()),
        default=None,
        help="Specific model(s) to download (default: all)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=list(ALL_MODELS.keys()),
        default=[],
        help="Model(s) to skip",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    to_download = args.model if args.model else list(ALL_MODELS.keys())
    to_download = [m for m in to_download if m not in (args.skip or [])]

    results = {}
    for model_name in to_download:
        fn = ALL_MODELS[model_name]
        results[model_name] = fn(force=args.force)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info(f"  {name:<20} {status}")
        if not ok:
            all_ok = False

    if not all_ok:
        logger.warning("Some downloads failed. Check the log above for details.")
        sys.exit(1)

    logger.info("All downloads completed successfully.")


if __name__ == "__main__":
    main()
