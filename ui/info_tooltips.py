"""
Material-UI Style Info Tooltip System for Medical Data Interface
Creates clean, simple tooltips without icons for enhanced user experience.
"""

import gradio as gr

def create_info_tooltip(tooltip_text="", position="top"):
    """
    Create a Material-UI style info tooltip with simple design
    
    Args:
        tooltip_text: Text to show on hover
        position: Tooltip position (top, bottom, left, right)
    
    Returns:
        gr.HTML component with styled info tooltip
    """
    
    # Generate unique ID for this tooltip
    import random
    tooltip_id = f"tooltip_{random.randint(1000, 9999)}"
    
    tooltip_html = f"""
    <div class="mui-tooltip-container" style="
        display: inline-flex; 
        position: relative; 
        margin-left: 8px; 
        align-items: center;
        vertical-align: middle;
        flex-shrink: 0;
    ">
            <div class="mui-info-trigger" style="
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                cursor: help;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s ease-in-out;
                user-select: none;
                position: relative;
                flex-shrink: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 10px;
                font-weight: 600;
                color: #6b7280;
            " 
            onmouseover="
                this.style.background='#fed7aa'; 
                this.style.borderColor='#fb923c'; 
                this.style.color='#ea580c';
                this.style.transform='scale(1.1)';
                var tooltip = document.getElementById('{tooltip_id}');
                var rect = this.getBoundingClientRect();
                tooltip.style.top = rect.top + 'px';
                tooltip.style.left = (rect.right + 10) + 'px';
                tooltip.style.opacity='1'; 
                tooltip.style.visibility='visible';
                tooltip.style.display='block';
            "
            onmouseout="
                this.style.background='#f3f4f6'; 
                this.style.borderColor='#d1d5db'; 
                this.style.color='#6b7280';
                this.style.transform='scale(1)';
                var tooltip = document.getElementById('{tooltip_id}');
                tooltip.style.opacity='0'; 
                tooltip.style.visibility='hidden';
                tooltip.style.display='none';
            ">i</div>        <div id="{tooltip_id}" class="mui-tooltip-content" style="
            position: fixed;
            top: 0;
            left: 0;
            transform: none;
            background: rgba(17, 24, 39, 0.98);
            color: #ffffff;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 400;
            line-height: 1.5;
            max-width: 280px;
            min-width: 150px;
            white-space: normal;
            box-shadow: 
                0 10px 15px -3px rgba(0, 0, 0, 0.3),
                0 4px 6px -2px rgba(0, 0, 0, 0.2);
            z-index: 10000;
            opacity: 0;
            visibility: hidden;
            display: none;
            transition: opacity 0.2s ease-in-out, visibility 0.2s ease-in-out;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            {tooltip_text}
        </div>
    </div>
    """
    
    return gr.HTML(tooltip_html)

def create_section_header_with_info(title, info_text, title_level="##"):
    """
    Create a section header with an info tooltip inline (single component)
    
    Args:
        title: The markdown title text
        info_text: Information to show in tooltip
        title_level: Markdown title level (##, ###, etc.)
    
    Returns:
        Single gr.HTML component with both title and tooltip inline
    """
    
    # Generate unique ID for this tooltip
    import random
    tooltip_id = f"tooltip_{random.randint(1000, 9999)}"
    
    # Convert markdown level to HTML
    if title_level == "##":
        html_tag = "h2"
        font_size = "1.5em"
        margin = "0.83em 0"
    elif title_level == "###":
        html_tag = "h3"
        font_size = "1.17em"
        margin = "1em 0"
    else:
        html_tag = "h2"
        font_size = "1.5em"
        margin = "0.83em 0"
    
    combined_html = f"""
    <div style="
        display: flex; 
        align-items: center; 
        gap: 8px; 
        margin: {margin};
        flex-wrap: nowrap;
    ">
        <{html_tag} style="
            font-size: {font_size};
            font-weight: bold;
            margin: 0;
            color: inherit;
            flex-shrink: 0;
        ">{title}</{html_tag}>
        
        <div class="mui-tooltip-container" style="
            display: inline-flex; 
            position: relative; 
            align-items: center;
            vertical-align: middle;
            flex-shrink: 0;
            margin-left: 4px;
        ">
            <div class="mui-info-trigger" style="
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                cursor: help;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s ease-in-out;
                user-select: none;
                position: relative;
                flex-shrink: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 10px;
                font-weight: 600;
                color: #6b7280;
            " 
            onmouseover="
                this.style.background='#fed7aa'; 
                this.style.borderColor='#fb923c'; 
                this.style.color='#ea580c';
                this.style.transform='scale(1.1)';
                var tooltip = document.getElementById('{tooltip_id}');
                var rect = this.getBoundingClientRect();
                tooltip.style.top = rect.top + 'px';
                tooltip.style.left = (rect.right + 10) + 'px';
                tooltip.style.opacity='1'; 
                tooltip.style.visibility='visible';
                tooltip.style.display='block';
            "
            onmouseout="
                this.style.background='#f3f4f6'; 
                this.style.borderColor='#d1d5db'; 
                this.style.color='#6b7280';
                this.style.transform='scale(1)';
                var tooltip = document.getElementById('{tooltip_id}');
                tooltip.style.opacity='0'; 
                tooltip.style.visibility='hidden';
                tooltip.style.display='none';
            ">i</div>
            
            <div id="{tooltip_id}" class="mui-tooltip-content" style="
                position: fixed;
                top: 0;
                left: 0;
                transform: none;
                background: rgba(17, 24, 39, 0.98);
                color: #ffffff;
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 400;
                line-height: 1.5;
                max-width: 300px;
                min-width: 160px;
                white-space: normal;
                box-shadow: 
                    0 10px 15px -3px rgba(0, 0, 0, 0.3),
                    0 4px 6px -2px rgba(0, 0, 0, 0.2);
                z-index: 10000;
                opacity: 0;
                visibility: hidden;
                display: none;
                transition: opacity 0.2s ease-in-out, visibility 0.2s ease-in-out;
                pointer-events: none;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                {info_text}
            </div>
        </div>
    </div>
    """
    
    return gr.HTML(combined_html)

def create_load_label_file_header():
    """
    Create the Load Label File header with Material-UI style tooltip
    """
    info_text = """
    <strong>Load Label File (.label)</strong><br><br>
    Upload a label file to define regions of interest for targeted segmentation. Label files contain 
    predefined annotations that guide the AI segmentation process for specific anatomical structures.
    <br><br>
    • <em>Format:</em> .label files with coordinate and structure definitions<br>
    • <em>Purpose:</em> Guided segmentation with predefined targets<br>
    • <em>Benefit:</em> Improved accuracy for specific anatomical regions
    """
    
    return create_section_header_with_info(
        title="Add Label File",
        info_text=info_text,
        title_level="##"
    )

def create_segmentation_settings_header():
    """
    Create the Segmentation Settings header with Material-UI style tooltip
    """
    info_text = """
    <strong>Segmentation Settings</strong><br><br>
    Configure AI-powered medical image segmentation parameters. Choose between different AI models, 
    processing modes, and output settings to optimize segmentation results for your specific use case.
    <br><br>
    • <em>Single Slice:</em> Process only the current slice<br>
    • <em>All Records:</em> Process entire 3D volume with quality filtering<br>
    • <em>Guided Segmentation:</em> Use point/box prompts for precise control
    """
    
    return create_section_header_with_info(
        title="Segmentation Settings",
        info_text=info_text,
        title_level="##"
    )

def create_load_medical_data_header():
    """
    Create the Load Medical Data header with Material-UI style tooltip
    """
    info_text = """
    <strong>Medical Data Loading</strong><br><br>
    Load single medical files (DICOM, NIFTI, MAT) or enter a directory path containing multiple DICOM files 
    for 3D volume processing. Supports automatic file detection and metadata extraction.
    <br><br>
    • <em>DICOM:</em> Single or multi-slice medical images<br>
    • <em>NIFTI:</em> Neuroimaging format (.nii, .nii.gz)<br>
    • <em>MAT:</em> MATLAB data files<br>
    • <em>Directory:</em> Bulk DICOM file processing
    """
    
    return create_section_header_with_info(
        title="Load Medical Data",
        info_text=info_text,
        title_level="##"
    )

def create_segmentation_with_ai_header():
    """
    Create the Segmentation with AI header with Material-UI style tooltip
    """
    info_text = """
    <strong>AI-Powered Segmentation</strong><br><br>
    Advanced medical image segmentation using state-of-the-art AI models. Choose between automatic 
    whole-area segmentation or guided segmentation with point/box prompts for precise control.
    <br><br>
    • <em>Automatic:</em> Detects all major structures automatically<br>
    • <em>Guided:</em> User-defined points or boxes for targeted segmentation<br>
    • <em>Quality Control:</em> Score thresholding for reliable results
    """
    
    return create_section_header_with_info(
        title="Segmentation with AI",
        info_text=info_text,
        title_level="##"
    )

def create_component_with_tooltip(component, info_text, label_override=None):
    """
    Create a component with an inline tooltip next to its label
    
    Args:
        component: The Gradio component 
        info_text: Text to show in tooltip
        label_override: Optional label override for the component
    
    Returns:
        gr.HTML component with enhanced label and tooltip
    """
    
    # Get the component's label
    if hasattr(component, 'label') and component.label:
        label_text = label_override or component.label
    else:
        label_text = label_override or "Component"
    
    # Generate unique ID for this tooltip
    import random
    tooltip_id = f"tooltip_{random.randint(1000, 9999)}"
    
    enhanced_label_html = f"""
    <div style="
        display: flex; 
        align-items: center; 
        gap: 6px; 
        margin-bottom: 4px;
        flex-wrap: nowrap;
    ">
        <label style="
            font-size: 14px;
            font-weight: 500;
            color: inherit;
            margin: 0;
            flex-shrink: 0;
        ">{label_text}</label>
        
        <div class="mui-tooltip-container" style="
            display: inline-flex; 
            position: relative; 
            align-items: center;
            vertical-align: middle;
            flex-shrink: 0;
        ">
            <div class="mui-info-trigger" style="
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                cursor: help;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s ease-in-out;
                user-select: none;
                position: relative;
                flex-shrink: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 9px;
                font-weight: 600;
                color: #6b7280;
            " 
            onmouseover="
                this.style.background='#fed7aa'; 
                this.style.borderColor='#fb923c'; 
                this.style.color='#ea580c';
                this.style.transform='scale(1.15)';
                var tooltip = document.getElementById('{tooltip_id}');
                var rect = this.getBoundingClientRect();
                tooltip.style.top = rect.top + 'px';
                tooltip.style.left = (rect.right + 10) + 'px';
                tooltip.style.opacity='1'; 
                tooltip.style.visibility='visible';
                tooltip.style.display='block';
            "
            onmouseout="
                this.style.background='#f3f4f6'; 
                this.style.borderColor='#d1d5db'; 
                this.style.color='#6b7280';
                this.style.transform='scale(1)';
                var tooltip = document.getElementById('{tooltip_id}');
                tooltip.style.opacity='0'; 
                tooltip.style.visibility='hidden';
                tooltip.style.display='none';
            ">i</div>
            
            <div id="{tooltip_id}" class="mui-tooltip-content" style="
                position: fixed;
                top: 0;
                left: 0;
                transform: none;
                background: rgba(17, 24, 39, 0.98);
                color: #ffffff;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 400;
                line-height: 1.5;
                max-width: 260px;
                min-width: 140px;
                white-space: normal;
                box-shadow: 
                    0 10px 15px -3px rgba(0, 0, 0, 0.3),
                    0 4px 6px -2px rgba(0, 0, 0, 0.2);
                z-index: 10000;
                opacity: 0;
                visibility: hidden;
                display: none;
                transition: opacity 0.2s ease-in-out, visibility 0.2s ease-in-out;
                pointer-events: none;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                {info_text}
            </div>
        </div>
    </div>
    """
    
    return gr.HTML(enhanced_label_html)

def create_info_enhanced_component(component_func, info_text, **component_kwargs):
    """
    Create any Gradio component enhanced with an info tooltip
    
    Args:
        component_func: Gradio component function (e.g., gr.Dropdown, gr.Slider)
        info_text: Text to show in tooltip
        **component_kwargs: Arguments to pass to the component function
    
    Returns:
        Tuple of (component, info_tooltip)
    """
    
    component = component_func(**component_kwargs)
    info_tooltip = create_info_tooltip(tooltip_text=info_text)
    
    return component, info_tooltip

# Predefined tooltips for common medical data interface elements
MEDICAL_TOOLTIPS = {
    "file_input": """
        <strong>File Upload</strong><br><br>
        Upload single medical image files. Supported formats include DICOM (.dcm), 
        NIFTI (.nii, .nii.gz), and MATLAB (.mat) files. 
        <br><br>
        <em>Tip:</em> For multiple files, use the directory path option below.
    """,
    
    "dir_input": """
        <strong>Directory Path</strong><br><br>
        Enter the full path to a directory containing DICOM files for batch processing. 
        The system will automatically detect and load all compatible files.
        <br><br>
        <em>Example:</em> C:\\Medical_Data\\Patient_001\\
    """,
    
    "view_selector": """
        <strong>View Orientation</strong><br><br>
        Select the anatomical view plane for medical image display:
        <br><br>
        • <em>Axial:</em> Horizontal cross-sections (top-down view)<br>
        • <em>Sagittal:</em> Vertical side-to-side sections<br>
        • <em>Coronal:</em> Vertical front-to-back sections
    """,
    
    "processing_mode": """
        <strong>Processing Mode</strong><br><br>
        Choose how the AI processes your medical data:
        <br><br>
        • <em>Single Slice:</em> Process only the currently visible slice - faster, focused analysis<br>
        • <em>All Records:</em> Process the entire 3D volume - comprehensive analysis with 3D visualization
    """,
    
    "score_threshold": """
        <strong>Quality Threshold</strong><br><br>
        Minimum confidence score (0.0 - 1.0) required for AI segmentation results. 
        Higher values show only high-confidence segments, lower values include more potential findings.
        <br><br>
        <em>Recommended:</em> 0.3 for general use, 0.5+ for high precision needs
    """,
    
    "window_level": """
        <strong>Window Level (Center)</strong><br><br>
        Adjusts the center point of the display window for optimal tissue contrast. 
        Different values highlight different tissue types in medical images.
        <br><br>
        <em>Common values:</em> Brain (40), Soft tissue (50), Bone (400)
    """,
    
    "window_width": """
        <strong>Window Width</strong><br><br>
        Controls the range of pixel values displayed. Narrower windows increase contrast 
        but may clip details, wider windows show more detail but with less contrast.
        <br><br>
        <em>Common values:</em> Brain (80), Soft tissue (350), Bone (1500)
    """
}

def get_medical_tooltip(key):
    """Get a predefined medical tooltip by key"""
    return MEDICAL_TOOLTIPS.get(key, "No tooltip available for this component.")

# CSS styles for Material-UI style tooltips - More robust and conflict-resistant
TOOLTIP_CSS = """
<style>
/* Material-UI style tooltip styles - scoped to avoid conflicts */
.mui-tooltip-container {
    position: relative !important;
    display: inline-flex !important;
}

.mui-info-trigger {
    position: relative !important;
    z-index: 1 !important;
}

.mui-info-trigger:hover {
    transform: scale(1.15) !important;
    transition: all 0.15s ease-in-out !important;
}

.mui-tooltip-content {
    position: fixed !important;
    z-index: 10000 !important;
    pointer-events: none !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    /* Break out of Gradio containers */
    transform: none !important;
}

/* Ensure tooltips work in different container contexts and break out of overflow constraints */
.gradio-container .mui-tooltip-content,
.gradio-app .mui-tooltip-content,
div[data-testid="block-container"] .mui-tooltip-content,
.mui-tooltip-content {
    position: fixed !important;
    z-index: 10000 !important;
    /* Override any container overflow settings */
    clip: auto !important;
    overflow: visible !important;
}

/* Prevent tooltip overflow issues */
.mui-tooltip-content {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    hyphens: auto !important;
}

/* Animation keyframes */
@keyframes mui-tooltip-fade-in {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes mui-tooltip-fade-out {
    from {
        opacity: 1;
    }
    to {
        opacity: 0;
    }
}

/* Ensure flex layouts work properly */
.header-with-info {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-wrap: nowrap !important;
}

.component-with-info-row {
    display: flex !important;
    align-items: flex-start !important;
    gap: 8px !important;
    flex-wrap: nowrap !important;
}

/* Force tooltips to be above everything else */
.mui-tooltip-content {
    position: fixed !important;
    z-index: 999999 !important;
}
</style>
"""
