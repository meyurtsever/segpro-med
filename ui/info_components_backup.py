"""
SegMed-Pro Info Message Components
This module contains reusable info message components for providing
guidance and explanations throughout the application.
"""
import gradio as gr

def create_info_message(message: str, message_type: str = "info", visible: bool = True):
    """
    Create an informative message component with integrated toggle show/hide functionality
    
    Args:
        message: The message text to display
        message_type: Type of message - "info" (light yellow), "success" (light green), "warning" (light red)
        visible: Whether the message is initially visible
    
    Returns:
        Tuple of (info_container, toggle_button, info_content) for external control
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
            "text_color": "#8b1538",  # Darker for better readability
            "strong_color": "#5c0e24",
            "header_bg": "#f7d1d1"  # Slightly darker red for header
        }
    }
    
    color_scheme = colors.get(message_type, colors["info"])
    
    # Create a professional-looking integrated toggle info component
    with gr.Column(visible=visible, elem_classes=["info-message-container"]) as info_container:
        # Create the integrated card with clickable header
        info_content = gr.HTML(
            value=f"""
            <div id="info-card" style="
                background-color: {color_scheme['bg_color']}; 
                border: 2px solid {color_scheme['border_color']}; 
                border-radius: 10px; 
                margin: 8px 0; 
                box-shadow: 0 3px 8px rgba(0,0,0,0.12);
                overflow: hidden;
                transition: all 0.3s ease;
            ">
                <!-- Clickable Header with Toggle -->
                <div onclick="toggleTipContent(this)" style="
                    background: linear-gradient(135deg, {color_scheme['header_bg']}, {color_scheme['bg_color']});
                    padding: 10px 15px;
                    border-bottom: 1px solid {color_scheme['border_color']};
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    font-weight: 600;
                    color: {color_scheme['text_color']};
                    font-size: 13px;
                    user-select: none;
                    transition: background 0.2s ease;
                " onmouseover="this.style.opacity='0.9'" onmouseout="this.style.opacity='1'">
                    <span style="display: flex; align-items: center;">
                        <span style="margin-right: 8px; font-size: 14px;">💡</span>
                        <span>Quick Tip</span>
                    </span>
                    <span class="toggle-icon" style="
                        font-size: 12px;
                        transition: transform 0.3s ease;
                        font-weight: bold;
                        color: {color_scheme['text_color']};
                    ">▼</span>
                </div>
                
                <!-- Content Area -->
                <div class="tip-content" style="
                    padding: 15px 18px;
                    color: {color_scheme['text_color']};
                    font-size: 14px;
                    line-height: 1.6;
                    transition: all 0.3s ease;
                    display: block;
                ">
                    <style>
                        .info-message-content {{
                            color: {color_scheme['text_color']} !important;
                        }}
                        .info-message-content strong {{
                            color: {color_scheme['strong_color']} !important;
                            font-weight: bold !important;
                        }}
                    </style>
                    <div class="info-message-content">
                        {message}
                    </div>
                </div>
            </div>
            
            <script>
            function toggleTipContent(header) {{
                const content = header.nextElementSibling;
                const icon = header.querySelector('.toggle-icon');
                const isVisible = content.style.display !== 'none';
                
                if (isVisible) {{
                    content.style.display = 'none';
                    icon.innerHTML = '▶';
                    icon.style.transform = 'rotate(0deg)';
                }} else {{
                    content.style.display = 'block';
                    icon.innerHTML = '▼';
                    icon.style.transform = 'rotate(0deg)';
                }}
            }}
            </script>
            """,
            visible=True,
            elem_classes=["info-content"]
        )
        
        # Hidden toggle button for backend state management (if needed)
        toggle_btn = gr.Button(
            value="expanded",  # Track state: "expanded" or "collapsed"
            visible=False
        )
    
    return info_container, toggle_btn, info_content

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
    
    return create_info_message(message, "info", visible=True)

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
    
    return create_info_message(message, "info", visible=True)

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
    
    return create_info_message(message, "success", visible=True)

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
