# Mammography Orientation Implementation

## Overview
This document describes the implementation of mammography (MG) orientation support in the Editor tab, allowing the system to automatically detect and handle MG-specific orientations (LCC, LMLO, RCC, RMLO) instead of standard anatomical views (Axial, Sagittal, Coronal).

## Changes Made

### 1. Editor Tab (`ui/editor_tab.py`)

#### New Helper Functions

1. **`detect_mg_orientation_from_metadata(metadata)`**
   - Detects MG orientation from DICOM metadata
   - Checks ViewPosition tag (0018,5101) and ImageLaterality (0020,0062)
   - Falls back to SeriesDescription or StudyDescription
   - Returns: 'LCC', 'LMLO', 'RCC', 'RMLO', or None

2. **`detect_mg_orientation_from_filename(filename)`**
   - Extracts orientation from filename
   - Useful when files are named like "LCC.dcm", "RMLO.dcm"
   - Returns: 'LCC', 'LMLO', 'RCC', 'RMLO', or None

3. **`get_mg_orientations_from_directory(file_list)`**
   - Scans a list of files to find available MG orientations
   - Returns: dict mapping orientation -> file_path
   - Example: `{'LCC': '/path/to/LCC.dcm', 'RMLO': '/path/to/RMLO.dcm'}`

4. **`update_view_selector_for_modality(metadata, file_list=None)`**
   - Updates view selector choices based on modality
   - For MG: returns ['LCC', 'LMLO', 'RCC', 'RMLO'] (filtered by available files)
   - For other modalities: returns ['Axial', 'Sagittal', 'Coronal']
   - Returns: (choices, default_value, visible)

5. **`get_file_for_mg_orientation(orientation, file_list, current_data_directory=None)`**
   - Gets the file path for a specific MG orientation
   - Returns: file_path or None

#### UI Changes

- Updated `view_selector` Radio component with info text:
  ```python
  info="Orientation changes based on modality (MG: LCC, LMLO, RCC, RMLO)"
  ```

- Exposed helper functions in the return dictionary under `'mg_orientation_functions'`

### 2. Data Handlers (`ui/handlers.py`)

#### Modified `load_data_for_annotator` Function

- Added automatic view selector update based on modality:
  ```python
  from ui.editor_tab import update_view_selector_for_modality
  view_choices, view_value, view_visible = update_view_selector_for_modality(
      self.state.current_metadata, 
      self.state.file_list
  )
  ```

- Added `view_selector` as 10th return value:
  ```python
  return (..., gr.Radio(choices=view_choices, value=view_value, visible=view_visible))
  ```

### 3. Main Application (`app.py`)

#### Updated Data Loading Event Handlers

- Added `view_selector` to outputs for:
  - `load_btn.click()`
  - `file_input.change()`
  - All sample data loading buttons (cvm, normal, hgg)

#### Enhanced `handle_view_change_with_labels` Function

Added MG orientation handling:

```python
if modality == 'MG' and view_value in ['LCC', 'LMLO', 'RCC', 'RMLO']:
    # Load the specific MG file for this orientation
    file_path = get_file_for_mg_orientation(view_value, ...)
    pixel_array, metadata = load_single_dicom(file_path, ...)
    # Update state and display
```

Key behaviors:
- Detects MG modality and orientation values
- Loads the appropriate DICOM file for the selected orientation
- Treats each orientation as a single 2D image
- Hides slice slider for MG images
- Updates slice text to show orientation (e.g., "0/0 (LCC)")

### 4. DICOM Utilities (`utils/dicom_utils.py`)

#### Enhanced Metadata Extraction

Added mammography-specific metadata fields:

```python
# Mammography-specific metadata
if hasattr(ds, 'ViewPosition'):
    metadata['ViewPosition'] = ds.ViewPosition
if hasattr(ds, 'ImageLaterality'):
    metadata['ImageLaterality'] = ds.ImageLaterality
if hasattr(ds, 'ViewCodeSequence'):
    # Extract ViewCodeSequence information
```

## Usage

### Loading MG Data from Directory

1. Enter directory path containing MG files (e.g., `/path/to/836163430/`)
2. Click "Load Data"
3. System automatically detects MG modality
4. View selector updates to show: LCC, LMLO, RCC, RMLO (based on available files)
5. Default view is set to first available orientation

### Switching Between Orientations

1. Select different orientation from View Orientation radio buttons
2. System automatically loads the corresponding DICOM file
3. Image updates to show the selected orientation
4. Slice slider is hidden (each orientation is a single image)

### File Naming Convention

For automatic detection from filenames, use:
- `LCC.dcm` - Left Cranio-Caudal
- `LMLO.dcm` - Left Medio-Lateral Oblique
- `RCC.dcm` - Right Cranio-Caudal
- `RMLO.dcm` - Right Medio-Lateral Oblique

Or include orientation in filename:
- `patient123_LCC_2024.dcm`
- `study_RMLO.dcm`

## Technical Details

### MG Orientation Detection Priority

1. **Metadata-based** (highest priority):
   - ViewPosition tag (0018,5101)
   - ImageLaterality tag (0020,0062)
   - ViewCodeSequence
   - SeriesDescription
   - StudyDescription

2. **Filename-based** (fallback):
   - Pattern matching in filename

### State Management

- `self.state.current_view` stores the selected orientation (e.g., "LCC")
- `self.state.current_data` contains the single 2D image as 3D array [1, H, W]
- `self.state.file_list` contains all MG files in directory
- `self.state.current_metadata` contains metadata from currently displayed file

### Compatibility

- Non-MG modalities (CT, MR, etc.) continue to use standard views
- System automatically switches between MG and standard views based on Modality tag
- Existing functionality for volumetric data remains unchanged

## Testing

### Test with Sample MG Data

```python
# Directory: /home/enes/segpro-med/836163430/
# Contains: LCC.dcm, LMLO.dcm, RCC.dcm, RMLO.dcm
```

1. Load the directory in Editor tab
2. Verify view selector shows MG orientations
3. Switch between orientations
4. Verify correct images are displayed
5. Test annotation tools on each orientation

## Future Enhancements

1. Add support for additional MG views (CC, MLO, ML, LM, etc.)
2. Implement orientation-specific annotation presets
3. Add MG-specific windowing presets
4. Support for tomosynthesis sequences
5. CAD (Computer-Aided Detection) integration for MG
