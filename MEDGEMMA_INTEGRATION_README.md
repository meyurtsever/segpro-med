# MedGemma-4B Integration for SegMed-Pro

## Overview

This document describes the integration of Google's MedGemma-4B visual language model into SegMed-Pro's Editor Tab, providing advanced medical image analysis capabilities alongside the existing SmolVLM and Med-R1 models.

## Features Added

### 1. Enhanced VLM UI Interface
- **VLM Model Selector**: Dropdown combobox allowing users to choose between:
  - SmolVLM
  - Med-R1  
  - MedGemma-4B (default selection)
- **Unified "Get Medical Analysis" Button**: Replaces the separate "Run SmolVLM" and "Run Med-R1 VLM" buttons
- **Improved Layout**: Cleaner, more intuitive interface with better organization

### 2. MedGemma-4B Backend Implementation

#### Service Layer (`models/medgemma/medgemma_service.py`)
- **Persistent Service**: Efficient model loading and inference with memory management
- **Automatic Device Detection**: Auto-selects GPU/CPU based on availability
- **4-bit Quantization**: Memory-efficient inference on GPU with BitsAndBytesConfig
- **Robust Error Handling**: Graceful fallbacks for various failure scenarios
- **Medical-Specific Prompts**: Pre-defined prompts optimized for medical imaging tasks

#### Handler Layer (`ui/medgemma_handlers.py`)
- **Unified Interface**: Consistent API matching existing SmolVLM and Med-R1 handlers
- **Medical Prompt Types**:
  - Anomaly identification focused prompts
  - General anatomical description prompts
  - Label suggestion prompts for annotation workflow
- **Image Processing**: Automatic preprocessing for optimal MedGemma performance

### 3. Model Integration

#### Hugging Face Model
- **Model**: `google/medgemma-4b-it` (4 billion parameter instruction-tuned model)
- **Storage**: Models stored in `models/medgemma/` directory
- **Auto-Download**: Automatic model download on first use

#### Dependencies Added
```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
protobuf>=3.20.0
```

## Implementation Details

### UI Changes

#### Before
```python
# Old separate buttons
vlm_btn = gr.Button("Run SmolVLM", variant="primary", size="lg", scale=1)
vlm_med_r1_btn = gr.Button("Run Med-R1 VLM", variant="secondary", size="lg", scale=1)
```

#### After
```python
# New unified interface
vlm_model_selector = gr.Dropdown(
    choices=["SmolVLM", "Med-R1", "MedGemma-4B"],
    value="MedGemma-4B",  # Default selected value
    label="VLM Model",
    scale=2
)
vlm_run_btn = gr.Button("🔍 Get Medical Analysis", variant="primary", size="lg", scale=2)
```

### Backend Integration

#### App Initialization
```python
# Added MedGemma handlers to main app
from ui.medgemma_handlers import MedGemmaHandlers
self.editor_medgemma_handlers = MedGemmaHandlers(self.state)
```

#### Unified Event Handler
```python
def handle_vlm_inference(vlm_model, image_display, identify_anomalies, describe_slice):
    """Handle VLM inference based on selected model"""
    if vlm_model == "MedGemma-4B":
        return self.editor_medgemma_handlers.run_vlm_inference(
            image_display, identify_anomalies, describe_slice
        )
    # ... other models
```

## Usage Instructions

### For Users

1. **Load Medical Image**: Use the data loading section to load DICOM or NIfTI files
2. **Select VLM Model**: Choose "MedGemma-4B" from the dropdown (default)
3. **Configure Analysis**:
   - Check "Identify Anomalies" for pathology detection
   - Check "Describe MRI Slice" for anatomical descriptions
4. **Run Analysis**: Click "Get Medical Analysis" button
5. **Review Results**: Analysis appears in the VLM Analysis text field

### For Developers

#### Adding New VLM Models
1. Create service file in `models/[model_name]/`
2. Implement handler in `ui/[model_name]_handlers.py`
3. Add to dropdown choices in `editor_tab.py`
4. Update unified handler in `app.py`

#### Customizing Medical Prompts
```python
# In medgemma_service.py
MEDICAL_PROMPTS = {
    "custom_prompt": "Your custom medical analysis prompt here...",
    # ... existing prompts
}
```

## Medical Prompts

### Pre-defined Prompt Types

1. **Anomaly Detection**
   ```
   "As a medical AI assistant, carefully analyze this brain MRI slice. 
   Identify any abnormal findings, anomalies, or pathological regions..."
   ```

2. **Anatomical Description**
   ```
   "As a medical AI assistant, provide a detailed description of this brain MRI slice. 
   Identify the anatomical structures visible, the imaging plane..."
   ```

3. **Label Suggestions** (Future Use)
   ```
   "This is a brain MRI slice. Please list the anatomical or pathological structures 
   that are visible. Return a comma-separated list of possible labels..."
   ```

## Technical Specifications

### Performance Optimizations
- **Model Quantization**: 4-bit quantization for GPU inference
- **Memory Management**: Automatic cleanup and garbage collection
- **Device Optimization**: Automatic GPU/CPU selection
- **Connection Pooling**: Persistent service to avoid reload overhead

### Error Handling
- **Service Availability**: Graceful degradation if model unavailable
- **Hardware Compatibility**: Automatic fallback to CPU if GPU fails
- **Input Validation**: Comprehensive validation of image inputs
- **Exception Recovery**: Detailed error messages for debugging

### Security Considerations
- **Model Integrity**: Downloaded from official Hugging Face repository
- **Input Sanitization**: Safe handling of medical image data
- **Resource Limits**: Memory and computation safeguards
- **Logging**: Comprehensive logging for audit trails

## Installation and Setup

### Automatic Setup
The MedGemma model will be automatically downloaded on first use. Ensure sufficient disk space (~8GB) for the model files.

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Pre-download model (optional)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/medgemma-4b-it')"
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU inference
- **Recommended**: 16GB RAM, CUDA-compatible GPU with 8GB+ VRAM
- **Storage**: 8GB free space for model files

## Future Enhancements

### Planned Features
1. **Custom Medical Prompts**: User-defined prompt templates
2. **Batch Processing**: Multiple slice analysis
3. **Model Fine-tuning**: Domain-specific model adaptation
4. **Multi-modal Input**: Integration with DICOM metadata

### Integration Opportunities
1. **Label Suggestion Workflow**: Automatic annotation labeling
2. **Quality Assessment**: Image quality scoring
3. **Report Generation**: Structured medical reports
4. **Teaching Mode**: Educational annotations and explanations

## Troubleshooting

### Common Issues

#### Model Download Fails
```
Error: Failed to download MedGemma model
Solution: Check internet connection and disk space
```

#### GPU Memory Error
```
Error: CUDA out of memory
Solution: Model automatically falls back to CPU
```

#### Import Errors
```
Error: MedGemma service not available
Solution: Ensure all dependencies are installed
```

### Debug Mode
Enable detailed logging by setting the log level:
```python
import logging
logging.getLogger('medgemma').setLevel(logging.DEBUG)
```

## Files Modified/Added

### New Files
- `models/medgemma/medgemma_service.py`: Core MedGemma service implementation
- `models/medgemma/__init__.py`: Package initialization
- `ui/medgemma_handlers.py`: MedGemma handler for UI integration

### Modified Files
- `ui/editor_tab.py`: Updated VLM UI interface
- `app.py`: Added MedGemma handlers and unified VLM event handling
- `requirements.txt`: Added MedGemma dependencies

## Testing

### Unit Tests
- Service initialization and cleanup
- Image preprocessing and inference
- Error handling and fallback scenarios
- Handler integration with UI components

### Integration Tests
- End-to-end VLM workflow
- Model switching functionality
- Memory usage and performance testing
- Multi-device compatibility testing

---

**Note**: This implementation follows Azure coding best practices with comprehensive error handling, security considerations, and performance optimizations as specified in the requirements.
