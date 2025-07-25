# MedGemma-4B Integration for SegMed-Pro

This directory contains the integration of Google's MedGemma-4B model into SegMed-Pro for medical image analysis.

## 🎯 Overview

MedGemma-4B is Google's specialized medical language model optimized for medical imaging tasks. It provides:
- **Medical Image Analysis**: Analyze brain MRI slices for anomalies and anatomical structures
- **Clinical Insights**: Generate detailed medical descriptions and findings
- **Label Suggestions**: Suggest anatomical/pathological labels for image regions

## 🔐 Important: Authentication Required

⚠️ **MedGemma-4B is a gated model** - you need to request access and authenticate with HuggingFace.

### Step-by-Step Authentication Setup

1. **Create HuggingFace Account**
   - Visit: https://huggingface.co
   - Create a free account

2. **Request Model Access**
   - Go to: https://huggingface.co/google/medgemma-4b-it
   - Click "Request access to this model"
   - Wait for approval (may take time)

3. **Create Access Token**
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Copy the token

4. **Set Up Authentication** (Choose one method)

   **Method 1: Quick Setup Script**
   ```bash
   cd models/medgemma
   python setup_auth.py
   ```

   **Method 2: Environment Variable**
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

   **Method 3: HuggingFace CLI**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

   **Method 4: Manual Token File**
   ```bash
   echo "your_token_here" > ~/.hf_token
   ```

## 🚀 Installation

### Option 1: Full Installation Script
```bash
cd models/medgemma
python install_medgemma.py
```

### Option 2: Manual Installation

1. **Install Dependencies**
   ```bash
   pip install torch>=2.0.0 transformers>=4.36.0 accelerate>=0.24.0 bitsandbytes>=0.41.0 huggingface_hub>=0.17.0
   ```

2. **Set Up Authentication** (see above)

3. **Download Model**
   ```bash
   python download_medgemma.py
   ```

## 📁 Files Description

- **`medgemma_service.py`** - Core service for MedGemma inference
- **`hf_auth.py`** - HuggingFace authentication helper
- **`download_medgemma.py`** - Model download script
- **`install_medgemma.py`** - Complete installation script
- **`setup_auth.py`** - Quick authentication setup
- **`__init__.py`** - Package initialization

## 🎮 Usage in SegMed-Pro

1. **Start SegMed-Pro**
   ```bash
   python app.py
   ```

2. **Navigate to Editor Tab**

3. **Load Medical Image**
   - Load DICOM or NIfTI file

4. **Select MedGemma-4B**
   - Choose "MedGemma-4B" from VLM Model dropdown

5. **Configure Analysis**
   - ✅ "Identify Anomalies" - Focus on abnormal findings
   - ✅ "Describe MRI Slice" - General anatomical description

6. **Run Analysis**
   - Click "🔍 Get Medical Analysis"

## 🔧 Features

### Medical Prompts
- **Anomaly Detection**: Identify pathological regions and abnormalities
- **Anatomical Description**: Describe visible structures and image quality
- **Label Suggestion**: Generate possible labels for image regions

### Performance Optimizations
- **4-bit Quantization**: Reduces memory usage on GPU
- **Persistent Service**: Keeps model loaded for fast inference
- **Automatic Fallback**: Falls back to CPU if GPU unavailable
- **Error Handling**: Robust error handling and recovery

### Memory Management
- **Smart Device Selection**: Automatically chooses best available device
- **Memory Cleanup**: Proper resource cleanup and garbage collection
- **Quantization Support**: Optional 4-bit quantization for memory efficiency

## 🛠️ Troubleshooting

### Authentication Issues
```
Error: Access to model google/medgemma-4b-it is restricted
```
**Solution**: Follow authentication setup steps above.

### Memory Issues
```
CUDA out of memory
```
**Solutions**:
- Model automatically falls back to CPU
- Uses 4-bit quantization on GPU to reduce memory
- Close other GPU applications

### Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Model Download Issues
```
Connection error during download
```
**Solutions**:
- Check internet connection
- Verify HuggingFace authentication
- Try downloading manually with the download script

## 📊 Technical Specifications

- **Model**: google/medgemma-4b-it
- **Parameters**: 4 billion
- **Input**: 512x512 RGB images
- **Output**: Medical text analysis
- **Memory**: ~8GB GPU memory (with quantization)
- **Device Support**: CUDA, CPU

## 🔄 Development

### Adding New Prompts
Edit `MEDICAL_PROMPTS` in `medgemma_service.py`:
```python
MEDICAL_PROMPTS = {
    "your_prompt_type": "Your medical prompt here...",
    # ...
}
```

### Modifying Service
The `MedGemmaService` class handles:
- Model initialization and loading
- Image preprocessing
- Text generation
- Resource management

## 📝 License

This integration follows the original MedGemma-4B license terms. Please review Google's model license before use.

## 🆘 Support

If you encounter issues:
1. Check this README for troubleshooting
2. Verify HuggingFace authentication
3. Ensure model access approval
4. Check system requirements

## 🔗 Links

- **MedGemma Model**: https://huggingface.co/google/medgemma-4b-it
- **HuggingFace Tokens**: https://huggingface.co/settings/tokens
- **Transformers Library**: https://huggingface.co/docs/transformers
