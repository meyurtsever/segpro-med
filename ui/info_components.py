"""
SegMed-Pro Info Message Components
This module contains reusable info message components for providing
guidance and explanations throughout the application.
"""
import gradio as gr

def create_info_message(message: str, message_type: str = "info", visible: bool = True, open_state: bool = True):
    """
    Create an informative message component using gr.Accordion with preserved CSS styling
    
    Args:
        message: The message text to display
        message_type: Type of message - "info" (light yellow), "success" (light green), "warning" (light red)
        visible: Whether the message component is initially visible
        open_state: Whether the accordion is initially open
    
    Returns:
        The accordion component for external control
    """
    # Define colors based on message type
    colors = {
        "info": {
            "bg_color": "#fff8dc",  # Light yellow
            "border_color": "#f0d000",
            "text_color": "#4a3c28",  # Much darker brown for better readability
            "strong_color": "#2d1f0f",  # Very dark for <strong> tags
            "header_bg": "#f7f0b8"  # Slightly darker yellow for header
        },
        "success": {
            "bg_color": "#d4f8d4",  # Light green
            "border_color": "#4caf50",
            "text_color": "#1b5e20",  # Darker for better readability
            "strong_color": "#0d3818",
            "header_bg": "#c1e6c1"  # Slightly darker green for header
        },
        "warning": {
            "bg_color": "#ffe6e6",  # Light red
            "border_color": "#f44336", 
            "text_color": "#8b1538",  # Darker for readability
            "strong_color": "#5c0e24",
            "header_bg": "#f7d1d1"  # Slightly darker red for header
        }
    }
    
    color_scheme = colors.get(message_type, colors["info"])
    
    # Create accordion with custom styling that preserves all existing CSS
    with gr.Accordion(
        label="💡 Quick Tip", 
        open=open_state, 
        visible=visible,
        elem_classes=[f"info-accordion-{message_type}"]
    ) as info_accordion:
        
        # Apply the preserved CSS styling to the accordion
        gr.HTML(f"""
        <style>
            /* Custom styling for {message_type} info accordion */
            .info-accordion-{message_type} {{
                background-color: {color_scheme['bg_color']} !important;
                border: 2px solid {color_scheme['border_color']} !important;
                border-radius: 10px !important;
                margin: 2px 0 !important;
                box-shadow: 0 3px 8px rgba(0,0,0,0.12) !important;
                overflow: hidden !important;
                transition: all 0.3s ease !important;
            }}
            
            /* Style the accordion header */
            .info-accordion-{message_type} > .label-wrap {{
                background: linear-gradient(135deg, {color_scheme['header_bg']}, {color_scheme['bg_color']}) !important;
                padding: 8px 12px !important;
                border-bottom: 1px solid {color_scheme['border_color']} !important;
                cursor: pointer !important;
                font-weight: 600 !important;
                color: {color_scheme['text_color']} !important;
                font-size: 13px !important;
                user-select: none !important;
                transition: background 0.2s ease !important;
                border-radius: 8px 8px 0 0 !important;
                margin: 0 !important;
                line-height: 1.2 !important;
            }}
            
            .info-accordion-{message_type} > .label-wrap:hover {{
                opacity: 0.9 !important;
            }}
            
            /* Style the accordion content area */
            .info-accordion-{message_type} .accordion-content {{
                background-color: {color_scheme['bg_color']} !important;
                padding: 0px 15px !important;
                color: {color_scheme['text_color']} !important;
                font-size: 14px !important;
                line-height: 1.5 !important;
                transition: all 0.3s ease !important;
                border-radius: 0 0 8px 8px !important;
                margin: 0 !important;
            }}
            
            /* Text styling within content */
            .info-accordion-{message_type} .info-message-content {{
                color: {color_scheme['text_color']} !important;
            }}
            
            .info-accordion-{message_type} .info-message-content strong {{
                color: {color_scheme['strong_color']} !important;
                font-weight: bold !important;
            }}
        </style>
        """)
        
        # Content area with preserved styling
        gr.HTML(f"""
        <div class="accordion-content">
            <div class="info-message-content">
                {message}
            </div>
        </div>
        """)
    
    return info_accordion


def create_vlm_label_management_info():
    """
    Create specific info message for VLM Label Management section
    """
    message = """
    <strong>💡 VLM Label Management:</strong> This section uses Visual Language Models to automatically suggest relevant medical labels for your images. 
    <br><br>
    <strong>📝 How it works:</strong> Select a VLM model below and click <strong>"Suggest Labels"</strong> to get AI-powered label suggestions based on your current image.
    <br><br>
    <strong>🎯 Recommendation:</strong> For best medical imaging results, use <strong>MedGemma-4B</strong> which is specifically trained on medical data.
    """
    
    return create_info_message(message, "info", visible=True, open_state=False)


def create_vlm_tools_info():
    """
    Create specific info message for VLM Tools section
    """
    message = """
    <strong>🔍 VLM Tools:</strong> This section provides comprehensive Visual Language Model analysis for medical imaging interpretation.
    <br><br>
    <strong>📝 How it works:</strong> Select your preferred VLM model, optionally use voice input for prompts, choose analysis options, and click <strong>"Get Medical Analysis"</strong> for detailed insights.
    <br><br>
    <strong>🎯 Recommendation:</strong> Use <strong>MedGemma-4B</strong> with <strong>"Identify Anomalies"</strong> enabled for the most accurate medical image analysis.
    """
    
    return create_info_message(message, "info", visible=True, open_state=False)


def create_custom_prompts_info():
    """
    Create specific info message for Custom Prompts section
    """
    message = """
    <strong>✍️ Custom Prompts:</strong> Create personalized prompts using medical terminology for specialized analysis of anatomical structures.
    <br><br>
    <strong>📝 How it works:</strong> Use the text field above to describe specific anatomical regions or pathological findings you want to analyze. The VLM will provide targeted insights.
    <br><br>
    <strong>🎯 Examples:</strong> Try prompts like "Analyze the hippocampus for atrophy signs" or "Identify lesions in white matter regions" for focused medical analysis.
    """
    
    return create_info_message(message, "warning", visible=True)


def create_vlm_custom_prompt_info():
    """
    Create specific info message for VLM Custom Prompt section
    """
    message = """
    <strong>✏️ Custom Prompts:</strong> You don't have to choose between the pre-assigned prompt options below. You can create your own custom prompt for the AI to get tailored output.
    <br><br>
    <strong>📝 How to use:</strong> Either <strong>write your prompt</strong> in the text field or <strong>record your voice</strong> which will be automatically transcribed to text.
    <br><br>
    <strong>💡 Tip:</strong> Custom prompts allow you to ask specific questions about the medical image or request particular types of analysis.
    """
    
    return create_info_message(message, "success", visible=True, open_state=False)


def create_custom_info_message(title: str, description: str, recommendation: str = "", message_type: str = "info"):
    """
    Create a custom info message with structured content
    
    Args:
        title: Main title of the info message
        description: Description of functionality
        recommendation: Optional recommendation text
        message_type: Type of message styling
    """
    message = f"<strong>💡 {title}:</strong> {description}"
    
    if recommendation:
        message += f"<br><br><strong>🎯 Recommendation:</strong> {recommendation}"
    
    return create_info_message(message, message_type, visible=True)


# Predefined message templates for common use cases
INFO_MESSAGES = {
    "vlm_label_management": """
        <strong>Smart Labeling:</strong> Use the VLM tools to automatically generate precise labels for complex medical structures. The AI can identify and classify regions that might be challenging to segment manually.<br><br>
        <strong>Quality Control:</strong> Always review AI-generated labels carefully - they serve as intelligent starting points, not final annotations.
    """,
    
    "vlm_tools": """
        <strong>Voice Input:</strong> Record anatomical descriptions using natural language. The system converts speech to structured prompts for better AI understanding.<br><br>
        <strong>Smart Prompts:</strong> Use the custom prompt feature to describe exactly what you want to segment using medical terminology.
    """,
    
    "custom_prompts": """
        <strong>Medical Terminology:</strong> Use precise anatomical terms (e.g., "hippocampus", "cortical gray matter") for better segmentation accuracy.<br><br>
        <strong>Context Matters:</strong> Include imaging modality and plane information in your prompts when relevant.
    """
}
