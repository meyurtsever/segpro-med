# Med-R1 VLM Label Suggestion Integration

## Overview

This implementation integrates Med-R1 VLM (Visual Language Model) for suggesting semantic labels based on user-drawn annotations in the SegMed-Pro application.

## Features Added

### 1. New Label Suggestion Method
- **File**: `ui/med_r1_handlers.py`
- **Method**: `suggest_labels_for_annotations()`
- **Purpose**: Analyzes annotated images and suggests semantic labels for highlighted regions

### 2. New UI Button
- **File**: `ui/editor_tab.py`
- **Button**: "🏷️ Suggest Labels (VLM)"
- **Location**: Below existing VLM buttons in the Editor tab
- **Functionality**: Triggers label suggestion analysis

### 3. Event Handler
- **File**: `app.py`
- **Handler**: Connects the new button to the Med-R1 label suggestion method
- **Inputs**: Current image with annotations from `image_annotator`
- **Outputs**: Label suggestions displayed in VLM caption textbox

## How It Works

### 1. Input Requirements
- The model receives the image as currently shown in `image_annotator`
- Includes all user-drawn annotations/overlays (polygons, boxes, brushes, etc.)
- Annotations are automatically detected and counted

### 2. Inference Logic
1. **Trigger**: User clicks "🏷️ Suggest Labels (VLM)" button
2. **Capture**: System captures current view from `image_annotator`
3. **Annotation Detection**: Analyzes annotation metadata to understand what was drawn
4. **Prompt Generation**: Creates specialized prompt for label suggestion based on annotations
5. **VLM Analysis**: Med-R1 analyzes the annotated image
6. **Output**: Returns specific semantic labels for each marked region

### 3. Prompt Strategy
The system creates a specialized prompt that:
- Informs Med-R1 about the number of annotations
- Describes the types of annotations (boxes, polygons, etc.)
- Requests specific semantic labels for each region
- Focuses on medical terminology and anatomical structures

### 4. Expected Behavior
- **Multi-region Support**: Handles multiple annotated regions in a single slice
- **Region-aware Analysis**: VLM suggestions are specific to highlighted areas
- **Medical Focus**: Uses brain MRI-specific terminology and anatomy knowledge
- **User Control**: Users can accept or discard suggested labels
- **Error Handling**: Graceful handling when no annotations are present

## Technical Implementation

### Med-R1 Handler Enhancement
```python
def suggest_labels_for_annotations(self, image_annotator_value: Optional[dict]) -> str:
    """
    Suggest semantic labels for user-drawn annotations using Med-R1 VLM
    """
    # Extract composite image with annotations
    # Create specialized prompt for label suggestion
    # Run Med-R1 inference
    # Format and return results
```

### UI Integration
```python
# New button in editor_tab.py
vlm_suggest_labels_btn = gr.Button(
    "🏷️ Suggest Labels (VLM)", 
    variant="secondary", 
    size="lg"
)

# Event handler in app.py
vlm_suggest_labels_btn.click(
    fn=self.med_r1_handlers.suggest_labels_for_annotations,
    inputs=[image_display],
    outputs=[vlm_caption]
)
```

## Usage Instructions

### For Users
1. Load a brain MRI image in the Editor tab
2. Draw annotations (polygons, boxes, etc.) on regions of interest
3. Click "🏷️ Suggest Labels (VLM)" button
4. Review the suggested labels in the VLM Analysis textbox
5. Apply or modify labels as needed

### For Developers
1. The label suggestion method is in `ui/med_r1_handlers.py`
2. UI components are in `ui/editor_tab.py`
3. Event handlers are in `app.py`
4. The method can be extended to support additional annotation types
5. Prompts can be customized for different medical imaging modalities

## Benefits

### 1. Improved Workflow
- Reduces manual label entry time
- Provides medical expertise through VLM knowledge
- Maintains annotation context awareness

### 2. Enhanced Accuracy
- Uses Med-R1's medical training for relevant suggestions
- Considers anatomical context and relationships
- Provides standard medical terminology

### 3. User Experience
- Seamless integration with existing annotation workflow
- Non-intrusive optional feature
- Clear visual feedback and error handling

## Future Enhancements

### Possible Improvements
1. **Interactive Label Selection**: Allow users to click on suggested labels to apply them
2. **Confidence Scores**: Display VLM confidence for each suggestion
3. **Multi-modal Support**: Extend to other imaging modalities (CT, X-ray, etc.)
4. **Batch Processing**: Process multiple annotated slices at once
5. **Custom Prompts**: Allow users to customize the analysis prompt
6. **Label History**: Track and suggest previously used labels

### Integration Opportunities
1. **Label Manager**: Connect with existing label management system
2. **Export Integration**: Include suggested labels in export workflows
3. **Quality Metrics**: Track suggestion accuracy and user feedback
4. **Training Loop**: Use user corrections to improve suggestions

## Files Modified

1. **`ui/med_r1_handlers.py`**
   - Added `suggest_labels_for_annotations()` method
   - Enhanced `_save_processed_image()` with suffix support
   - Added `_format_annotation_info()` helper method

2. **`ui/editor_tab.py`**
   - Added "🏷️ Suggest Labels (VLM)" button
   - Updated VLM caption placeholder text
   - Added button to component return tuple

3. **`app.py`**
   - Added event handler for new button
   - Updated component extraction to include new button

## Dependencies

- Med-R1 VLM service (existing)
- image_annotator component (existing)
- PIL/Pillow for image processing (existing)
- NumPy for array handling (existing)

## Testing Recommendations

1. Test with various annotation types (boxes, polygons, brushes)
2. Test with single and multiple annotations
3. Test error handling when no annotations present
4. Verify label suggestions are medically relevant
5. Test performance with different image sizes
6. Validate annotation metadata extraction

This implementation provides a robust foundation for VLM-powered label suggestion that can be further enhanced based on user feedback and specific workflow requirements.
