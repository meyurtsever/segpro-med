# SmolVLM Integration for Slice Captioning

## Overview

The SmolVLM (Small Visual Language Model) integration in SegMed-Pro provides automatic caption generation for medical image slices. This feature is available in the **Editor Tab** and allows users to get AI-generated descriptions of the currently displayed medical image slice.

## Location

The VLM feature is located in the **Editor Tab**, positioned below the "Slice" and "Crosshair" controls in the middle column.

## Components

- **"Run SmolVLM" Button**: Triggers the visual language model inference
- **"VLM Caption" Text Field**: Displays the AI-generated caption/description
- **Prompt Selection Checkboxes**: Choose between different analysis modes:
  - **"Identify Anomalies"**: Focuses on identifying abnormal regions and potential pathology (default)
  - **"Describe MRI Slice"**: Provides general description of anatomical structures

## How to Use

1. **Load Medical Data**: Use the data loading controls in the left column to load DICOM, NIfTI, or other medical images
2. **Navigate to Slice**: Use the slice navigation controls to select the desired slice
3. **Select Prompt Mode**: Choose between:
   - **Identify Anomalies**: For clinical analysis focusing on pathology detection
   - **Describe MRI Slice**: For general anatomical structure description
4. **Generate Caption**: Click the "Run SmolVLM" button to generate a description
5. **Read Results**: The generated caption will appear in the "VLM Caption" text field

## Prompt Modes

### Identify Anomalies (Default)
- **Purpose**: Clinical analysis and pathology detection
- **Prompt**: "Identify and label abnormal regions in this brain MRI. Highlight any suspicious areas and suggest their likely pathology."
- **Use Case**: Medical diagnosis, anomaly detection, clinical review

### Describe MRI Slice
- **Purpose**: General anatomical description
- **Prompt**: "Describe this medical image slice in detail, focusing on visible anatomical structures and any notable features."
- **Use Case**: Educational purposes, general image analysis, documentation

## Technical Details

### Model Location
- The SmolVLM model is located at: `/models/smolvlm/`
- CLI interface: `models/smolvlm/smolvlm_cli.py`

### Image Processing
- The current slice image is extracted from the `image_annotator` component
- Images are automatically converted from WebP (Gradio's default) to JPEG for model compatibility
- Grayscale medical images are converted to RGB format as required by the VLM

### Inference Parameters
- **Message Prompt**: Configurable based on selected prompt mode:
  - Anomalies: "Identify and label abnormal regions in this brain MRI. Highlight any suspicious areas and suggest their likely pathology."
  - Description: "Describe this medical image slice in detail, focusing on visible anatomical structures and any notable features."
- **Max Tokens**: 256
- **Device**: Auto-detection (CUDA if available, CPU otherwise)
- **Timeout**: 120 seconds

### Error Handling
- Graceful error handling for missing images, model failures, or timeouts
- Automatic cleanup of temporary image files
- Detailed error messages for troubleshooting

## Requirements

### Python Dependencies
- torch
- transformers
- PIL (Pillow)
- numpy

### Model Requirements
- HuggingFace Transformers library
- SmolVLM-Instruct model (automatically downloaded on first use)
- Sufficient GPU memory for CUDA acceleration (optional but recommended)

## Performance Notes

- **First Run**: May take longer due to model downloading and initialization
- **GPU Acceleration**: Significantly faster with CUDA-compatible GPU
- **Memory Usage**: Model requires ~2-4GB RAM/VRAM
- **Inference Time**: Typically 3-10 seconds depending on hardware

## Integration Architecture

```
Editor Tab UI
├── Navigation Controls (Slice, Crosshair)
├── VLM Integration Row
│   ├── "Run SmolVLM" Button → SmolVLMHandlers.run_vlm_inference()
│   └── "VLM Caption" TextField ← Generated caption
├── VLM Prompt Selection Row
│   ├── "Identify Anomalies" Checkbox (default: True)
│   └── "Describe MRI Slice" Checkbox (default: False)
└── Status/Metadata Sections

SmolVLM Workflow:
1. Extract image from image_annotator
2. Determine prompt based on checkbox selection
3. Convert numpy array to PIL Image
4. Save as temporary JPEG file
5. Execute smolvlm_cli.py subprocess with custom prompt
6. Parse and return generated caption
7. Cleanup temporary files
```

## Code Files

### New Files Added
- `ui/smolvlm_handlers.py`: Main VLM handler class
- `test_vlm_integration.py`: Integration test script

### Modified Files
- `ui/editor_tab.py`: Added VLM UI components
- `app.py`: Added VLM handler initialization and event connections

## Troubleshooting

### Common Issues

1. **"SmolVLM CLI not found"**
   - Ensure `models/smolvlm/smolvlm_cli.py` exists
   - Verify model installation

2. **"Model loading timeout"**
   - First run downloads large model files
   - Check internet connection
   - Consider using GPU acceleration

3. **"CUDA out of memory"**
   - Reduce max_tokens parameter
   - Use CPU inference instead
   - Close other GPU-intensive applications

4. **"No image available"**
   - Ensure medical data is loaded
   - Navigate to a valid slice
   - Check that image_annotator has content

### Debug Tips
- Check `medsam2_debug.log` for VLM-related errors
- Use the test script: `python test_vlm_integration.py`
- Monitor GPU memory usage during inference

## Future Enhancements

Potential improvements for future versions:
- Customizable prompt messages
- Multiple language support
- Integration with annotation workflow
- Batch processing for multiple slices
- Caption history and export functionality
- Fine-tuning for medical terminology

## Example Output

### Identify Anomalies Mode
```
VLM Caption: "Analysis of this axial brain MRI slice reveals a hypointense lesion in the left frontal white matter region, approximately 8mm in diameter. The lesion demonstrates clear borders and appears to be surrounded by minimal edema. Given the T1-weighted characteristics and location, differential diagnosis should include demyelinating plaque, small infarct, or possible metastatic lesion. Further contrast-enhanced imaging would be recommended for definitive characterization."
```

### Describe MRI Slice Mode  
```
VLM Caption: "This appears to be an axial T1-weighted MRI slice of the brain showing clear visualization of cerebral cortex, white matter structures, and ventricular system. The image demonstrates good contrast between gray and white matter, with visible sulci and gyri patterns typical of normal brain anatomy. The lateral ventricles appear symmetrical and of normal size. No obvious pathological abnormalities are evident in this particular slice."
```
