# KoGa Medical Imaging Annotation Tool

<p align="center">
  <b>AI-assisted medical image annotation and analysis platform</b><br>
  <i>Powered by MedSAM2 · MedGemma · Med-R1 · SmolVLM</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python Versions" />  <img src="https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x-76B900?logo=nvidia&logoColor=white" alt="CUDA" />  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />  <img src="https://img.shields.io/badge/Gradio-5.x-F97316?logo=gradio&logoColor=white" alt="Gradio" />  <img src="https://img.shields.io/badge/platform-Ubuntu%2022.04%2B%20%28recommended%29%20%7C%20Windows-orange?logo=linux&logoColor=white" alt="Platform" />  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License" />  <img src="https://img.shields.io/badge/models-HuggingFace-yellow?logo=huggingface&logoColor=white" alt="HuggingFace" />
</p>

---

## Overview

**KoGa Tool** (formerly SegMed-Pro) is a browser-based medical image annotation and AI-analysis platform built on [Gradio](https://gradio.app). It combines classical volumetric viewing and expert annotation tools with state-of-the-art AI models for interactive segmentation, visual question answering, and explainability — all in a single unified interface.

Supported imaging modalities include DICOM, and NIfTI series, covering brain MRI, abdominal CT, breast imaging, heart ultrasound, and more.

> **Platform note:** Ubuntu 22.04+ is the recommended operating system. CUDA-based features (Flash Attention, 4-bit quantization, multi-GPU) work best on Linux. Windows is supported for CPU-only and basic CUDA usage; some VLM features require Linux.

---

## Key Features

### Medical Image Viewing
- DICOM series loading with full metadata inspection
- NIfTI (`.nii`, `.nii.gz`) multi-planar reconstruction — axial / sagittal / coronal
- Mammography (MG) DICOM series support with dedicated viewer
- Crosshair-synchronized navigation across planes and slices
- Window / level adjustment and contrast controls
- MAT file and PNG export utilities

### Annotation

KoGa uses a custom annotator built on top of a fork of [gradio_image_annotator](https://github.com/edgarGracia/gradio_image_annotator). On top of the forked base component, the following tools and capabilities were implemented:

- **Polygon tool** — freehand polygon drawing with vertex editing
- **Bounding box tool** — click-and-drag box annotation
- **Point / marker tool** — single-point landmark placement
- **Eraser tool** — selective region erasing within an annotation
- **Undo / Redo** — full annotation history stack
- **Pan and zoom** — in-canvas zoom in/out and position controls
- **Label manager** — assign names, colors, and ICD-10 codes to annotation classes
- **Slice persistence** — annotations automatically persist across all slices and sessions
- **Brain ROI routing** — 12-region brain atlas routing for structured neuroimaging annotation
- **JSON export** — annotations exported as structured JSON per study

### AI-Assisted Segmentation (MedSAM2)
- Click-based interactive segmentation on any slice using [MedSAM2](https://github.com/bowang-lab/MedSAM2)
- Box and polygon prompt modes for guided segmentation
- Propagation of segmentation masks across volume slices
- One-click automatic mask generation (AMG) for entire volumes
- Export masks as NIfTI overlays or PNG images

### Visual Language Models (VLM)

KoGa integrates three VLMs for AI-generated medical reports, structured Q&A, and voice-driven report dictation:

| Model | Use Case | Auth Required |
|---|---|---|
| **SmolVLM-Instruct** | Fast report generation, lightweight Q&A | No |
| **Med-R1** (Qwen2-VL based) | Brain MRI analysis, chain-of-thought reasoning | No |
| **MedGemma-4B-IT** | Comprehensive medical Q&A, high accuracy | Yes (HF access) |

- Send any visible slice to a VLM with a single click
- **Voice input** — dictate your query using Whisper-based speech recognition; ideal for hands-free report generation while reviewing scans
- Visual grounding — automatically highlight the image regions the VLM attended to when generating a report
- Compare outputs from multiple VLMs side by side

### Explainability (XAI)
- LayerCAM, GradCAM, and attention-based visual grounding overlays
- Uncertainty maps integrated directly into the slice viewer
- VLM attention heatmap visualization for report grounding
- Side-by-side XAI method comparison view

### Crowdsourcing & Expert Annotation
- Multi-user login and role management (expert / annotator / admin)
- Campaign manager for distributing annotation tasks across users
- Per-user annotation tracking, submission, and progress dashboard
- Behavioral analytics — tool usage, timing, annotation patterns, and session replays

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.10 – 3.12 |
| PyTorch | 2.x (CUDA 11.8+ recommended) |
| CUDA (optional) | 11.8 / 12.x |
| RAM | >= 16 GB (>= 32 GB for MedGemma) |
| VRAM | >= 8 GB (>= 16 GB for MedGemma) |
| OS | Ubuntu 22.04+ (recommended), Windows 10/11 |

---

## Installation

### Ubuntu (Recommended)

```bash
# 1. Clone
git clone https://github.com/yourusername/koga-tool.git
cd koga-tool

# 2. Create a conda environment (Python 3.11 recommended)
conda create -n koga python=3.11 -y
conda activate koga

# 3. Install PyTorch with CUDA (adjust cu128 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install dcm2niix for DICOM conversion
sudo apt-get install -y dcm2niix
```

> **Flash Attention** (strongly recommended for VLM models on Linux):
> ```bash
> pip install flash-attn --no-build-isolation
> ```
> This is not available on Windows.

### Windows

```powershell
# 1. Clone
git clone https://github.com/yourusername/koga-tool.git
cd koga-tool

# 2. Create conda environment
conda create -n koga python=3.11 -y
conda activate koga

# 3. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install dependencies
pip install -r requirements.txt

# 5. dcm2niix: download the Windows binary from
#    https://github.com/rordenlab/dcm2niix/releases and add it to PATH
```

> Note: Flash Attention, some 4-bit quantization paths, and FSL-based de-identification are Linux-only.

### macOS (CPU only)

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

## Model Download

KoGa ships with a unified download script. Models are downloaded from [HuggingFace Hub](https://huggingface.co) and stored under `models/`.

### Download all models

```bash
python download_models.py
```

### Download specific models

```bash
python download_models.py --model medsam2
python download_models.py --model smolvlm med-r1
python download_models.py --skip medgemma       # skip gated models
```

### Individual model scripts

| Model | Script |
|---|---|
| MedSAM2 | `python models/medsam2/download_medsam2.py` |
| Med-R1 | `python models/med-r1/download_med_r1.py` |
| SmolVLM | `python models/smolvlm/download_smolvlm.py` |
| MedGemma-4B | `python models/medgemma/download_medgemma.py` |

### Auto-download at runtime

If a model checkpoint is missing when the application starts, the relevant service module automatically downloads it from HuggingFace Hub before loading. No manual intervention is needed for public models (SmolVLM, Med-R1, MedSAM2).

### MedGemma Authentication

MedGemma is a gated model. You must:
1. Request access at [https://huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
2. Authenticate:
   ```bash
   huggingface-cli login
   # or
   export HF_TOKEN=hf_your_token_here
   ```

---

## Running the Application

```bash
python app.py
```

The web interface will be available at **http://localhost:7860**

### Configuration

Key toggles at the top of `app.py`:

```python
ENABLE_AUTH = True                  # Enable crowdsourcing login system
ENABLE_BEHAVIORAL_TRACKING = True   # Enable behavioral analytics
```

Set `ENABLE_AUTH = False` to run in single-user mode without login.

### Database

The app uses a lightweight file-based store under `db/` for user accounts and session data. Passwords are hashed with [bcrypt](https://pypi.org/project/bcrypt/). This is a demonstration-grade setup — not intended for production.

Create the directory before first run:

```bash
mkdir db
```

The app initialises all required files automatically on startup.

---

## Application Tabs

### Viewer
- Load DICOM series (including MG), NIfTI volumes, or multi-frame images
- Navigate slices with synchronized crosshairs
- Multi-planar reconstruction (axial / sagittal / coronal)
- Adjust window / level
- Inspect full DICOM metadata

### Annotator
- Draw with polygon, bounding box, point, eraser, undo/redo, pan, and zoom tools
- Assign labels, colors, and ICD-10 codes
- Annotations persist across slices and sessions
- Export annotations as JSON

### Segmentation (MedSAM2)
- Click-, box-, or polygon-prompted interactive segmentation
- Mask propagation across volume slices
- Automatic mask generation (AMG)
- Export masks as NIfTI or PNG overlays

### VLM Analysis
- Send the current slice to SmolVLM, Med-R1, or MedGemma for report generation
- Voice dictation of queries via Whisper (speak → text → VLM)
- Structured Q&A with medical context
- Visual grounding — heatmap of VLM attention on the image
- Side-by-side comparison across models

### XAI / Explainability
- Select a model and layer
- Generate LayerCAM / GradCAM / attention overlays
- Uncertainty map visualization
- Side-by-side overlay comparison

### Expert / Contribute
- Login and role selection
- Campaign assignment and task queue
- Annotation submission and progress tracking

### Conversion & De-identification
- DICOM → NIfTI (via dcm2niix)
- NIfTI → MAT
- DICOM → MAT
- NIfTI slice → PNG export
- DICOM metadata sanitization (de-identification)
- Brain MRI de-facing via [pydeface](https://github.com/poldracklab/pydeface) (Linux only)

> **Note:** De-identification tools are supplied for research and internal institutional use. Always verify compliance with your data governance policies before processing patient data.

---

## Project Structure

```
koga-tool/
├── app.py                      # Main Gradio application entry point
├── download_models.py          # Master model download script
├── requirements.txt
│
├── ui/                         # Gradio UI tabs and event handlers
│   ├── viewer_tab.py
│   ├── custom_annotator_tab.py
│   ├── segmentation_handlers.py
│   ├── medgemma_handlers.py
│   ├── med_r1_handlers.py
│   ├── smolvlm_handlers.py
│   └── ...
│
├── models/                     # AI model implementations and download scripts
│   ├── medsam2/                # MedSAM2 interactive segmentation
│   │   ├── download_medsam2.py
│   │   ├── checkpoints/        # Model weights (gitignored, auto-downloaded)
│   │   └── ...
│   ├── medgemma/               # MedGemma-4B VLM (gated)
│   │   ├── download_medgemma.py
│   │   └── medgemma_service.py
│   ├── med-r1/                 # Med-R1 Qwen2-VL VLM
│   │   ├── download_med_r1.py
│   │   └── med_r1_service.py
│   └── smolvlm/                # SmolVLM-Instruct lightweight VLM
│       ├── download_smolvlm.py
│       └── smolvlm_service.py
│
├── xai/                        # Explainability methods
│   ├── segmentation/           # SAM2-based XAI
│   ├── vlm/                    # VLM visual grounding
│   └── visualization/          # Overlay rendering
│
├── utils/                      # Shared utilities
│   ├── conversion.py           # Format conversion
│   ├── dicom_utils.py
│   ├── nifti_utils.py
│   ├── annotation_manager.py
│   ├── deidentification.py     # DICOM de-identification / de-facing
│   └── ...
│
├── auth/                       # Authentication and user preferences
├── crowdsourcing/              # Campaign manager and task distribution
├── analytics/                  # Behavioral analytics storage
├── custom_components/          # Gradio custom component (forked annotator)
└── db/                         # Local annotation database (gitignored)
```
---

## Contributing

Contributions are welcome. Please:

1. Fork the repository and create a feature branch
2. Follow the existing code style
3. Open a pull request with a clear description

---

## Acknowledgements

KoGa Tool builds on the following open-source projects and models:

- [gradio_image_annotator](https://github.com/edgarGracia/gradio_image_annotator) — Edgar Gracia (forked and extended with polygon, eraser, undo/redo, zoom, and pan tools)
- [SAM2 / MedSAM2](https://github.com/bowang-lab/MedSAM2) — Meta AI & Wang Lab
- [MedGemma](https://huggingface.co/google/medgemma-4b-it) — Google DeepMind
- [Med-R1](https://huggingface.co/yuxianglai117/Med-R1) — Yuxiang Lai et al.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) — HuggingFace
- [Gradio](https://gradio.app) — HuggingFace
- [OpenAI Whisper](https://github.com/openai/whisper) — OpenAI
- [dcm2niix](https://github.com/rordenlab/dcm2niix) — Chris Rorden
- [pydeface](https://github.com/poldracklab/pydeface) — Poldrack Lab

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

