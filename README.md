# SegMed-Pro

A medical data annotation tool based on Gradio, inspired by ITK-SNAP.

## Features

- Support for multiple medical image formats:
  - DICOM series viewing and conversion
  - NIfTI file viewing and conversion
  - MAT file conversion
  - PNG export
- Interactive 3D volume viewing with multi-planar reconstruction
- Metadata inspection
- Crosshair navigation across slices
- Format conversion utilities

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/segpro-med.git
cd segpro-med
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Install dcm2niix for DICOM to NIfTI conversion (if needed):
   - On Linux: `sudo apt-get install dcm2niix`
   - On macOS: `brew install dcm2niix`
   - On Windows: Download from [GitHub](https://github.com/rordenlab/dcm2niix/releases)

## Usage

Run the application with:
```
python app.py
```

The web interface will be available at http://localhost:7860

### Viewer Tab

Use the Viewer tab to:
- Load DICOM series or NIfTI files
- Navigate through slices
- View in axial, sagittal, or coronal orientations
- Inspect metadata

### Conversion Tab

Use the Conversion tab to:
- Convert DICOM series to NIfTI format
- Convert NIfTI files to MAT format
- Convert DICOM series to MAT format
- Export NIfTI slices as PNG images

## Development Roadmap

### Phase 1: Core Viewing Infrastructure (Current)
- Basic viewing functionality
- DICOM and NIfTI support
- Crosshair navigation
- Basic metadata display

### Future Phases
- Enhanced annotation tools
- Segmentation support
- Integration with AI models (MedSAM)
- 3D visualization

## License

[MIT License](LICENSE)