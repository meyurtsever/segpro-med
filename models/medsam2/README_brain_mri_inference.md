# Brain MRI DICOM SAM2 Inference

This script performs automated brain MRI segmentation on DICOM files using the SAM2 (Segment Anything Model 2) architecture. It processes brain MRI DICOM volumes and generates high-quality segmentation masks for brain structures.

## Features

- **DICOM Support**: Direct processing of brain MRI DICOM files
- **Flexible Prompting**: Support for point prompts, box prompts, and automatic prompts
- **Window/Level Control**: Adjustable DICOM display windowing for optimal visualization
- **Batch Processing**: Process entire volumes or specific slice ranges
- **Multiple Output Formats**: Save masks as PNG images and NumPy arrays
- **Visualization**: Generate comprehensive visualization plots
- **CPU/GPU Support**: Automatic device detection with fallback to CPU

## Installation

Ensure you have the MedSAM2 environment set up with all dependencies:

```bash
# The script requires these main dependencies:
# - torch
# - numpy
# - pydicom
# - matplotlib
# - PIL (Pillow)
# - sam2 (MedSAM2 modules)
```

## Usage

### Basic Usage with Automatic Prompts

```bash
python brain_mri_dicom_inference.py \
    --dicom_folder /path/to/dicom/files \
    --output_dir results \
    --auto_prompts \
    --save_visualizations
```

### Usage with Manual Point Prompts

```bash
python brain_mri_dicom_inference.py \
    --dicom_folder /path/to/dicom/files \
    --output_dir results \
    --prompt_points prompts.json \
    --save_visualizations
```

### Usage with Box Prompts

```bash
python brain_mri_dicom_inference.py \
    --dicom_folder /path/to/dicom/files \
    --output_dir results \
    --prompt_boxes boxes.json \
    --save_visualizations
```

### Process Specific Slice Range

```bash
python brain_mri_dicom_inference.py \
    --dicom_folder /path/to/dicom/files \
    --output_dir results \
    --auto_prompts \
    --slice_range 10:20
```

## Command Line Arguments

- `--checkpoint`: Path to SAM2 checkpoint (default: checkpoints/MedSAM2_latest.pt)
- `--config`: Path to SAM2 config file (default: configs/sam2.1_hiera_t512.yaml)
- `--dicom_folder`: Path to folder containing DICOM files (required)
- `--output_dir`: Directory to save results (default: ./brain_mri_results)
- `--device`: Device to use - cuda/cpu (default: cuda)
- `--slice_range`: Slice range to process, e.g., "10:50" or "all" (default: all)
- `--window_center`: Window center for DICOM display in HU (default: 40)
- `--window_width`: Window width for DICOM display in HU (default: 80)
- `--prompt_points`: JSON file with point prompts for each slice
- `--prompt_boxes`: JSON file with box prompts for each slice
- `--auto_prompts`: Generate automatic prompts for brain structures
- `--save_visualizations`: Save visualization images

## Prompt File Formats

### Point Prompts (JSON)
```json
{
  "5": {
    "points": [[128, 128], [100, 120], [156, 120]],
    "labels": [1, 1, 1]
  },
  "10": {
    "points": [[130, 125], [110, 130], [150, 130]],
    "labels": [1, 1, 1]
  }
}
```

### Box Prompts (JSON)
```json
{
  "8": {
    "boxes": [[80, 80, 180, 180]]
  },
  "12": {
    "boxes": [[85, 85, 175, 175]]
  }
}
```

**Notes:**
- Point coordinates are in [x, y] format
- Labels: 1 = foreground, 0 = background
- Box format: [x1, y1, x2, y2] (top-left and bottom-right corners)
- Slice indices should be strings in the JSON

## Output Structure

```
output_dir/
├── summary.json                 # Processing summary and statistics
├── windowing_metadata.json     # Per-slice windowing parameters (NEW!)
├── masks/                       # Segmentation masks
│   ├── slice_0005_mask.png      # Best mask as PNG
│   ├── slice_0005_mask.npy      # Best mask as NumPy array
│   ├── slice_0005_all_masks.npy # All masks from SAM2
│   ├── slice_0005_scores.npy    # Quality scores for each mask
│   └── ...
└── visualizations/              # Visualization plots (if enabled)
    ├── slice_0005_visualization.png
    └── ...
```

## Example Workflows

### 1. Quick Test with Automatic Prompts
```bash
# Process a few slices for quick testing
python brain_mri_dicom_inference.py \
    --dicom_folder cvm48t1 \
    --output_dir quick_test \
    --auto_prompts \
    --slice_range 5:10 \
    --device cpu
```

### 2. Full Volume Processing
```bash
# Process entire brain volume
python brain_mri_dicom_inference.py \
    --dicom_folder cvm48t1 \
    --output_dir full_brain \
    --auto_prompts \
    --save_visualizations \
    --window_center 50 \
    --window_width 100
```

### 3. Targeted Segmentation with Manual Prompts
```bash
# Create prompt file first
python example_brain_mri_usage.py

# Run with manual prompts
python brain_mri_dicom_inference.py \
    --dicom_folder cvm48t1 \
    --output_dir targeted_seg \
    --prompt_points example_point_prompts.json \
    --save_visualizations
```

## DICOM Processing Details

### Automatic DICOM Metadata Windowing (NEW!)
The script now **automatically extracts and uses windowing parameters from each DICOM file's metadata**:
- **WindowCenter** and **WindowWidth** tags are read from each DICOM file
- **RescaleSlope** and **RescaleIntercept** are applied for proper value scaling
- Each slice uses its own optimal windowing parameters
- Fallback to default values (Center=40, Width=80) if metadata is missing

### Window/Level Settings
- **Automatic Mode (Recommended)**: Uses DICOM metadata windowing for each slice
- **Manual Override**: Use `--window_center` and `--window_width` to override all slices
- **Per-slice Metadata**: Windowing parameters are saved with results for reproducibility

### Benefits of DICOM Metadata Windowing
- **Optimal Contrast**: Each slice uses radiologist-intended window/level settings
- **Consistent Quality**: Maintains proper image preprocessing across the volume
- **Better Segmentation**: SAM2 receives optimally contrasted images
- **Preserves Intent**: Honors the original DICOM viewing parameters

### Automatic Prompts Strategy
When `--auto_prompts` is enabled, the script generates:
- Center point for brain tissue
- Left and right hemisphere points
- Upper and lower brain region points
- All points are marked as foreground (label=1)

### Image Preprocessing
1. Load DICOM pixel arrays
2. Apply window/level transformation
3. Normalize to 0-255 range
4. Convert to RGB format for SAM2
5. Automatic resizing handled by SAM2 transforms

## Performance Considerations

- **GPU vs CPU**: GPU is much faster but may have compatibility issues with newer hardware
- **Memory Usage**: Processing full volumes requires significant RAM
- **Slice Range**: Use `--slice_range` to process subsets for testing
- **Batch Size**: Currently processes one slice at a time for memory efficiency

## Troubleshooting

### GPU Compatibility Issues
If you encounter CUDA errors:
```bash
# Use CPU instead
python brain_mri_dicom_inference.py ... --device cpu
```

### Memory Issues
For large volumes:
```bash
# Process in smaller chunks
python brain_mri_dicom_inference.py ... --slice_range 0:10
python brain_mri_dicom_inference.py ... --slice_range 10:20
```

### DICOM Loading Issues
- Ensure DICOM files have `.dcm` extension
- Check that files are valid DICOM format
- Verify folder path is correct

## Integration with Existing Workflows

The script outputs can be easily integrated with:
- **Medical imaging software**: Load `.npy` files in Python
- **DICOM viewers**: Convert masks back to DICOM format
- **Analysis pipelines**: Use mask arrays for volume calculations
- **Visualization tools**: Use PNG outputs for quick viewing

## Example Applications

1. **Brain Tumor Segmentation**: Use targeted prompts around tumor regions
2. **Whole Brain Extraction**: Use automatic prompts for skull stripping
3. **Hemisphere Segmentation**: Use left/right hemisphere prompts
4. **Multi-structure Analysis**: Combine with different prompt strategies

## Future Enhancements

Potential improvements could include:
- Support for other medical imaging formats (NIfTI, etc.)
- 3D volume-based segmentation
- Integration with DICOM metadata
- Automatic quality assessment
- Multi-class segmentation support
