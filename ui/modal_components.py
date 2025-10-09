"""
SegMed-Pro Modal Components
This module provides modal-like fu            /* Modal container */
            .modal-container {
                position: fixed !important;
                top: 50% !important;
                left: 50% !important;
                transform: translate(-50%, -50%) !important;
                background: #27272A !important;
                border-radius: 16px !important;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8) !important;
                z-index: 1001 !important;
                width: 800px !important;
                max-width: 95vw !important;
                max-height: 80vh !important;
                overflow-y: auto !important;
                border: 1px solid #4b5563 !important;
                animation: modalSlideIn 0.3s ease-out !important;
            }    /* Modal header */
            .modal-header {{
                background: linear-gradient(135deg, #374151, #4b5563) !important;
                color: white !important;
                padding: 16px 20px !important;
                border-radius: 16px 16px 0 0 !important;
                border-bottom: 1px solid #4b5563 !important;
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                position: relative !important;
                gap: 16px !important;
            }}
            
            .modal-title {{
                font-size: 1.4rem !important;
                font-weight: 600 !important;
                margin: 0 !important;
                color: white !important;
                flex: 1 !important;
            }}ations by simulating
modal behavior with backdrop blur and layered components.
"""
import gradio as gr

def create_modal_backdrop():
    """
    Create a modal backdrop that simulates an overlay effect.
    This component starts visible to auto-show the modal.
    """
    return gr.HTML(
        value="",
        visible=True,  # Start visible for auto-opening
        elem_id="modal-backdrop",
        elem_classes=["modal-backdrop"]
    )

def create_editor_welcoming_modal():
    """
    Create a welcoming modal for the Editor tab with interactive quick start.
    This version uses actual Gradio buttons for proper event handling.
    """
    with gr.Column(
        visible=True,
        elem_id="modal-container",
        elem_classes=["modal-container"]
    ) as modal_container:
        
        # Modal styling (same as before but added sample button styles)
        gr.HTML(f"""
        <style>
            /* Modal backdrop - semi-transparent overlay */
            .modal-backdrop {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                background: rgba(39, 39, 42, 0.3) !important;
                z-index: 1000 !important;
                backdrop-filter: blur(1px) !important;
                -webkit-backdrop-filter: blur(1px) !important;
            }}
            
            /* Modal container */
            .modal-container {{
                position: fixed !important;
                top: 50% !important;
                left: 50% !important;
                transform: translate(-50%, -50%) !important;
                background: #27272A !important;
                border-radius: 16px !important;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8) !important;
                z-index: 1001 !important;
                width: 850px !important;
                max-width: 90vw !important;
                max-height: 80vh !important;
                overflow-y: auto !important;
                border: 1px solid #4b5563 !important;
                animation: modalSlideIn 0.3s ease-out !important;
            }}
            
            @keyframes modalSlideIn {{
                0% {{
                    opacity: 0;
                    transform: translate(-50%, -50%) scale(0.9) translateY(-20px);
                }}
                100% {{
                    opacity: 1;
                    transform: translate(-50%, -50%) scale(1) translateY(0);
                }}
            }}
            
            /* Modal header */
            .modal-header {{
                background: linear-gradient(135deg, #374151, #4b5563) !important;
                color: white !important;
                padding: 24px !important;
                border-radius: 8px 8px 0 0 !important;
                border-bottom: 1px solid #4b5563 !important;
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                position: relative !important;
            }}
            
            .modal-title {{
                font-size: 1.4rem !important;
                font-weight: 600 !important;
                margin: 0 !important;
                color: white !important;
                flex-grow: 1 !important;
            }}
            
            /* Modal content */
            .modal-content {{
                padding: 24px !important;
                color: #e5e7eb !important;
                line-height: 1.6 !important;
            }}
            
            .modal-content h2 {{
                color: #f9fafb !important;
                font-size: 1.3rem !important;
                margin-bottom: 16px !important;
                font-weight: 600 !important;
            }}
            
            .modal-content h3 {{
                color: #d1d5db !important;
                font-size: 1.1rem !important;
                margin-bottom: 12px !important;
                margin-top: 20px !important;
                font-weight: 500 !important;
            }}
            
            .modal-content p {{
                margin-bottom: 12px !important;
                color: #e5e7eb !important;
            }}
            
            .modal-content ul {{
                margin-bottom: 16px !important;
                padding-left: 20px !important;
            }}
            
            .modal-content li {{
                margin-bottom: 8px !important;
                color: #e5e7eb !important;
            }}
            
            .modal-content strong {{
                color: #f9fafb !important;
                font-weight: 600 !important;
            }}
            
            /* Close button styling - tiny and minimal */
            .modal-container .close-button {{
                background: rgba(55, 65, 81, 0.6) !important;
                color: #9ca3af !important;
                border: none !important;
                border-radius: 3px !important;
                padding: 1px 3px !important;
                cursor: pointer !important;
                font-size: 10px !important;
                font-weight: 300 !important;
                transition: all 0.2s ease !important;
                width: 40px !important;
                height: 40px !important;
                min-width: 20px !important;
                min-height: 20px !important;
                max-width: 20px !important;
                max-height: 20px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin-left: auto !important;
                flex-shrink: 0 !important;
                line-height: 1 !important;
                box-shadow: none !important;
            }}
            
            .modal-container .close-button:hover {{
                background: rgba(239, 68, 68, 0.7) !important;
                color: white !important;
                transform: scale(1.1) !important;
            }}
            
            .modal-container .close-button:active {{
                transform: scale(0.95) !important;
            }}
            
            /* Compact sample buttons row */
            .compact-sample-row {{
                gap: 12px !important;
                margin: 16px 0 !important;
            }}
            
            /* Compact sample buttons */
            .compact-sample-btn {{
                width: 100% !important;
                height: 40px !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
                padding: 8px 16px !important;
            }}
            
            .compact-sample-btn:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            }}
            
            .cavernoma-btn {{
                background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
                border-color: #2563eb !important;
            }}
            
            .healthy-btn {{
                background: linear-gradient(135deg, #10b981, #059669) !important;
                border-color: #047857 !important;
                color: white !important;
            }}
            
            .glioma-btn {{
                background: linear-gradient(135deg, #f59e0b, #d97706) !important;
                border-color: #b45309 !important;
                color: white !important;
            }}
            
            .cavernoma-card::before {{ background: linear-gradient(90deg, #f59e0b, #d97706) !important; }}
            .healthy-card::before {{ background: linear-gradient(90deg, #10b981, #059669) !important; }}
            .glioma-card::before {{ background: linear-gradient(90deg, #ef4444, #dc2626) !important; }}
            
            .card-header {{
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                margin-bottom: 12px !important;
            }}
            
            .card-icon {{
                font-size: 1.5rem !important;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)) !important;
            }}
            
            .card-badge {{
                background: rgba(59, 130, 246, 0.2) !important;
                color: #93c5fd !important;
                padding: 4px 8px !important;
                border-radius: 6px !important;
                font-size: 0.7rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            }}
            
            .card-content h4 {{
                color: #f9fafb !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                margin: 0 0 6px 0 !important;
            }}
            
            .card-content p {{
                color: #d1d5db !important;
                font-size: 0.85rem !important;
                margin: 0 0 8px 0 !important;
                line-height: 1.4 !important;
            }}
            
            .card-path {{
                color: #9ca3af !important;
                font-size: 0.75rem !important;
                font-family: 'Monaco', 'Menlo', monospace !important;
                background: rgba(0, 0, 0, 0.3) !important;
                padding: 4px 8px !important;
                border-radius: 4px !important;
                border-left: 3px solid #4b5563 !important;
            }}
            
            .cavernoma-btn:hover {{ box-shadow: 0 8px 20px rgba(245, 158, 11, 0.4) !important; }}
            .healthy-btn:hover {{ box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4) !important; }}
            .glioma-btn:hover {{ box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4) !important; }}
            
            /* Responsive design */
            @media (max-width: 768px) {{
                .modal-container {{
                    width: 95vw !important;
                    max-height: 85vh !important;
                    margin: 0 !important;
                }}
                
                .modal-content {{
                    padding: 16px !important;
                }}
                
                .modal-title {{
                    font-size: 1.2rem !important;
                }}
                
                .modal-header {{
                    padding: 12px 16px !important;
                }}
                
                .compact-sample-row {{
                    flex-direction: column !important;
                    gap: 8px !important;
                }}
                
                .compact-sample-btn {{
                    height: 36px !important;
                    font-size: 0.8rem !important;
                }}
            }}
        </style>
        """)
        
        # Modal header
        with gr.Row(elem_classes=["modal-header"]):
            with gr.Column(scale=30):
                gr.HTML(f'<h1 class="modal-title">Welcome to KoGa Medical Image Editor</h1>')
            
            with gr.Column(scale=1):
                close_button = gr.Button("✕", elem_classes=["close-button"], variant="secondary")

        # Modal content
        with gr.Column(elem_classes=["modal-content"]):
            gr.HTML("""
            <h2>🎉 Welcome to SegMed-Pro Editor!</h2>
            
            <p>Your comprehensive platform for <strong>AI-powered medical image segmentation</strong> and analysis.</p>
            
            <h3>🚀 Quick Start - Load Sample Data</h3>
            
            <p>Click any button below to automatically load sample medical imaging data:</p>
            """)
            
            # Compact sample data buttons
            with gr.Row(equal_height=True, elem_classes=["compact-sample-row"]):
                load_cvm_btn = gr.Button(
                    "Brain Sample - Malformation", 
                    variant="primary", 
                    size="sm",
                    elem_classes=["compact-sample-btn", "cavernoma-btn"]
                )
                
                load_normal_btn = gr.Button(
                    "Brain Sample - Benign", 
                    variant="secondary", 
                    size="sm",
                    elem_classes=["compact-sample-btn", "healthy-btn"]
                )
                
                load_hgg_btn = gr.Button(
                    "Brain Sample - Tumor", 
                    variant="stop", 
                    size="sm",
                    elem_classes=["compact-sample-btn", "glioma-btn"]
                )
            
        gr.HTML("""   
        <div style="background: #065f46; padding: 16px; border-radius: 8px; text-align: center; margin-top: 20px;">
            <p style="margin: 0; color: #d1fae5; font-size: 14px; font-weight: 500;">
                ✨ <strong>Ready to start?</strong> Load a dataset above and begin annotating!
            </p>
        </div>
        """)
    
    return modal_container, close_button, load_cvm_btn, load_normal_btn, load_hgg_btn

def show_modal(modal_container, backdrop):
    """
    Show the modal by making both backdrop and container visible.
    
    Args:
        modal_container: The modal container component
        backdrop: The backdrop component
    
    Returns:
        Updated components with visibility set to True
    """
    return (
        gr.update(visible=True),  # backdrop
        gr.update(visible=True)   # modal_container
    )

def hide_modal(modal_container, backdrop):
    """
    Hide the modal by making both backdrop and container invisible.
    
    Args:
        modal_container: The modal container component
        backdrop: The backdrop component
    
    Returns:
        Updated components with visibility set to False
    """
    return (
        gr.update(visible=False),  # backdrop
        gr.update(visible=False)   # modal_container
    )

# Backward compatibility aliases
create_welcome_modal = create_editor_welcoming_modal

def create_editor_welcoming_modal_system():
    """
    Create a modal system that automatically opens when the page loads.
    Uses Gradio's component lifecycle to trigger the modal.
    """
    # Create backdrop
    backdrop = create_modal_backdrop()
    
    # Create welcome modal - start as visible for auto-open
    welcome_modal, welcome_close, load_cvm_btn, load_normal_btn, load_hgg_btn = create_editor_welcoming_modal()
    
    # Set initial state to show modal automatically
    backdrop.visible = True
    welcome_modal.visible = True
    
    # Create a state component to track if modal has been shown
    modal_shown_state = gr.State(value=False)
    
    def show_welcome_modal():
        return gr.update(visible=True), gr.update(visible=True), True
    
    def hide_welcome_modal():
        return gr.update(visible=False), gr.update(visible=False), True
    
    # Connect close button
    welcome_close.click(
        fn=hide_welcome_modal,
        outputs=[backdrop, welcome_modal, modal_shown_state]
    )
    
    return {
        'backdrop': backdrop,
        'welcome_modal': welcome_modal,
        'welcome_close': welcome_close,
        'load_cvm_btn': load_cvm_btn,
        'load_normal_btn': load_normal_btn,
        'load_hgg_btn': load_hgg_btn,
        'modal_shown_state': modal_shown_state,
        'show_welcome': show_welcome_modal,
        'hide_welcome': hide_welcome_modal
    }

# Main function alias for backward compatibility
create_auto_opening_modal_system = create_editor_welcoming_modal_system

def create_segmentation_complete_modal():
    """
    Create a modal that appears after segmentation is completed to guide users
    on how to edit and refine their annotations.
    """
    with gr.Column(
        visible=False,  # Start hidden, will be shown after segmentation
        elem_id="segmentation-modal-container",
        elem_classes=["modal-container"]
    ) as modal_container:
        
        # Modal styling (focused on segmentation completion)
        gr.HTML(f"""
        <style>
            /* Segmentation modal specific styling */
            #segmentation-modal-container {{
                width: 650px !important;
                max-width: 90vw !important;
                padding: 0 !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                max-height: 80vh !important;
            }}
            
            .segmentation-modal-header {{
                background: linear-gradient(135deg, #059669, #047857) !important;
                color: white !important;
                padding: 16px 20px !important;
                border-radius: 16px 16px 0 0 !important;
                border-bottom: 1px solid #047857 !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                position: relative !important;
                min-height: 60px !important;
                margin: 0 !important;
            }}
            
            .segmentation-modal-title {{
                font-size: 1.3rem !important;
                font-weight: 600 !important;
                margin: 0 !important;
                color: white !important;
                text-align: center !important;
                flex: 1 !important;
            }}
            
            /* Position close button absolutely to avoid affecting title centering */
            .segmentation-modal-header .close-button {{
                position: absolute !important;
                right: 12px !important;
                top: 50% !important;
                transform: translateY(-50%) !important;
                background: rgba(255, 255, 255, 0.2) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 4px !important;
                padding: 4px !important;
                cursor: pointer !important;
                font-size: 12px !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
                width: 24px !important;
                height: 24px !important;
                min-width: 24px !important;
                min-height: 24px !important;
                max-width: 24px !important;
                max-height: 24px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                line-height: 1 !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
                flex-shrink: 0 !important;
            }}
            
            .segmentation-modal-header .close-button:hover {{
                background: rgba(239, 68, 68, 0.8) !important;
                color: white !important;
                transform: translateY(-50%) scale(1.05) !important;
                border-color: rgba(239, 68, 68, 0.8) !important;
            }}
            
            .segmentation-modal-header .close-button:active {{
                transform: translateY(-50%) scale(0.98) !important;
            }}
            
            .segmentation-modal-content {{
                padding: 24px !important;
                color: #e5e7eb !important;
                line-height: 1.6 !important;
                background: #27272A !important;
            }}
            
            .segmentation-modal-content h3 {{
                color: #10b981 !important;
                font-size: 1.1rem !important;
                margin-bottom: 12px !important;
                margin-top: 16px !important;
                font-weight: 600 !important;
            }}
            
            .segmentation-modal-content p {{
                margin-bottom: 14px !important;
                color: #d1d5db !important;
                font-size: 0.95rem !important;
            }}
            
            .segmentation-modal-content ul {{
                margin-bottom: 16px !important;
                padding-left: 20px !important;
            }}
            
            .segmentation-modal-content li {{
                margin-bottom: 8px !important;
                color: #d1d5db !important;
                font-size: 0.9rem !important;
            }}
            
            .segmentation-modal-content strong {{
                color: #f9fafb !important;
                font-weight: 600 !important;
            }}
            
            .completion-highlight {{
                background: rgba(16, 185, 129, 0.1) !important;
                border: 1px solid #10b981 !important;
                border-radius: 8px !important;
                padding: 16px !important;
                margin: 16px 0 !important;
                text-align: center !important;
            }}
            
            .completion-highlight h4 {{
                color: #10b981 !important;
                margin: 0 0 8px 0 !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
            }}
            
            .completion-highlight p {{
                color: #d1fae5 !important;
                margin: 0 !important;
                font-size: 0.9rem !important;
            }}
        </style>
        """)
        
        # Modal header
        with gr.Row(elem_classes=["segmentation-modal-header"]):
            gr.HTML(f'<h1 class="segmentation-modal-title">Segmentation Complete</h1>')
            close_button = gr.Button("✕", elem_classes=["close-button"], variant="secondary")

        # Modal content
        with gr.Column(elem_classes=["segmentation-modal-content"]):
            gr.HTML("""
            <div class="completion-highlight">
                <h4>AI Segmentation Successfully Generated</h4>
                <p>Your medical image has been processed and annotated</p>
            </div>
            
            <p>The AI has completed the initial segmentation. You can now refine and customize the results:</p>
            
            <h3>Edit Annotations</h3>
            <ul>
                <li><strong>Double-click any annotation</strong> to edit its properties</li>
                <li><strong>Rename labels</strong> to match your terminology</li>
                <li><strong>Change colors</strong> for better visual distinction</li>
                <li><strong>Drag annotations</strong> to reposition them as needed</li>
            </ul>
            
            <h3>Annotation Tools</h3>
            <ul>
                <li><strong>Select tool</strong> - Click and drag to move annotations</li>
                <li><strong>Rectangle tool</strong> - Draw new bounding boxes</li>
                <li><strong>Polygon tool</strong> - Create custom shaped annotations</li>
                <li><strong>Point tool</strong> - Add precise point markers</li>
            </ul>
            
            <p>All changes are automatically saved as you work. You can continue annotating or export your results when ready.</p>
            """)
    
    return modal_container, close_button

def create_segmentation_modal_system():
    """
    Create a modal system for post-segmentation guidance.
    """
    # Create backdrop for segmentation modal
    backdrop = create_modal_backdrop()
    
    # Create segmentation completion modal
    segmentation_modal, segmentation_close = create_segmentation_complete_modal()
    
    # Set initial state to hidden
    backdrop.visible = False
    segmentation_modal.visible = False
    
    def show_segmentation_modal():
        return gr.update(visible=True), gr.update(visible=True)
    
    def hide_segmentation_modal():
        return gr.update(visible=False), gr.update(visible=False)
    
    # Connect close button
    segmentation_close.click(
        fn=hide_segmentation_modal,
        outputs=[backdrop, segmentation_modal]
    )
    
    return {
        'backdrop': backdrop,
        'segmentation_modal': segmentation_modal,
        'segmentation_close': segmentation_close,
        'show_segmentation_modal': show_segmentation_modal,
        'hide_segmentation_modal': hide_segmentation_modal
    }