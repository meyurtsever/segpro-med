"""
PlotlyBridge component for bridging between Plotly visualizations and Python.
"""
import os
import gradio as gr
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PlotlyBridge(gr.Blocks):
    """
    A component that bridges between Plotly hover events and Python.
    This component allows updating crosshair position based on hover events in Plotly.
    """
    def __init__(
        self,
        plot_id=None,
        sync_crosshair=True,
        **kwargs
    ):
        """Initialize the PlotlyBridge component.
        
        Args:
            plot_id: ID of the plot to control (optional)
            sync_crosshair: Whether to sync hover data with crosshair (default: True)
        """        
        self.plot_id = plot_id
        self.sync_crosshair = sync_crosshair
        logger.info(f"Initializing PlotlyBridge with plot_id={plot_id}, sync_crosshair={sync_crosshair}")
        print(f"Initializing PlotlyBridge with plot_id={plot_id}, sync_crosshair={sync_crosshair}")
        
        super().__init__(**kwargs)
        with self:
            # Create a hidden textbox to capture hover events from JavaScript
            self.hover_data = gr.Textbox(
                value="", 
                visible=False, 
                elem_id="plotly-hover-data",
                label="Hover Data (Hidden)"
            )
            logger.info(f"Created hover_data textbox with elem_id=plotly-hover-data")
            
            # Add Svelte component for DOM interactions
            svelte_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PlotlyBridge.svelte")
            if os.path.exists(svelte_path):
                logger.info(f"Loading Svelte component from {svelte_path}")
                with open(svelte_path, "r") as f:
                    svelte_code = f.read()
                gr.HTML(f'<script type="module">{svelte_code}</script>')
            
            # Add JavaScript for Plotly event handling
            script = f"""
            <div id="plotly-bridge-container">
                <script>
                (function() {{
                    console.log("PlotlyBridge JS initialized with plotId={self.plot_id}, syncCrosshair={str(self.sync_crosshair).lower()}");
                    
                    // Configuration
                    const plotId = "{self.plot_id if self.plot_id else ''}";
                    const syncCrosshair = {str(self.sync_crosshair).lower()};
                    
                    // State
                    let plotDiv = null;
                    let hoverListenerActive = false;
                    let debounceTimeout = null;
                    let lastHoverTime = 0;
                    const debounceTime = 50; // ms
                    
                    // Find Plotly element
                    function findPlotlyElement() {{
                        if (plotId) {{
                            const elementById = document.getElementById(plotId);
                            if (elementById && (elementById._fullLayout || elementById.querySelector('.main-svg'))) {{
                                console.log(`PlotlyBridge: Found Plotly element by ID: ${{plotId}}`);
                                return elementById;
                            }} else {{
                                console.log(`PlotlyBridge: Element with ID ${{plotId}} not found or not a Plotly element`);
                            }}
                        }}
                        
                        const selectors = [
                            '.js-plotly-plot', 
                            '[data-testid="plotly"]',
                            '[data-testid="plotly-graph-div"]',
                            '.plotly-graph-div'
                        ];
                        
                        for (const selector of selectors) {{
                            const elements = document.querySelectorAll(selector);
                            for (const el of elements) {{
                                if (el._fullLayout) {{
                                    return el;
                                }}
                            }}
                        }}
                        
                        // Final attempt - any element with Plotly-specific children
                        const possiblePlots = document.querySelectorAll('div');
                        for (const div of possiblePlots) {{
                            if (div.querySelector('.main-svg') || 
                                div.querySelector('.plotly') || 
                                div._fullLayout) {{
                                return div;
                            }}
                        }}
                        return null;
                    }}
                    
                    // Send hover data to Python
                    function sendHoverData(x, y) {{
                        // Find the Gradio textbox component for hover data
                        const hoverDataElement = document.getElementById('plotly-hover-data');
                        if (!hoverDataElement) {{
                            console.error('PlotlyBridge: Could not find hover data element with ID plotly-hover-data');
                            return;
                        }}
                        
                        // Create JSON data
                        const hoverData = JSON.stringify({{
                            x: x,
                            y: y,
                            timestamp: Date.now()
                        }});
                        
                        console.log(`PlotlyBridge: Sending hover data: ${{hoverData}}`);
                        
                        // Set the value in the Gradio textbox
                        hoverDataElement.value = hoverData;
                        
                        // Trigger the change event to notify Gradio
                        const event = new Event('input', {{ bubbles: true }});
                        hoverDataElement.dispatchEvent(event);
                        
                        // Also trigger the change event
                        const changeEvent = new Event('change', {{ bubbles: true }});
                        hoverDataElement.dispatchEvent(changeEvent);
                        
                        console.log(`PlotlyBridge: Sent hover data to Python: x=${{x}}, y=${{y}}`);
                    }}
                    
                    // Handle Plotly hover events
                    function handleHover(event) {{
                        if (!event || !event.points || !event.points.length) {{
                            console.log('PlotlyBridge: Invalid hover event received', event);
                            return;
                        }}
                        
                        console.log('PlotlyBridge: Hover event detected!', event);
                        lastHoverTime = Date.now();
                        
                        // Skip if crosshair sync is disabled
                        if (!syncCrosshair) {{
                            console.log('PlotlyBridge: Crosshair sync disabled, ignoring hover event');
                            return;
                        }}
                        
                        // Debounce to prevent too many events
                        clearTimeout(debounceTimeout);
                        debounceTimeout = setTimeout(() => {{
                            const point = event.points[0];
                            
                            // Get coordinates - try different approaches since the format can vary
                            let x, y;
                            
                            if (point.hasOwnProperty('x') && point.hasOwnProperty('y')) {{
                                // Standard Plotly format
                                x = Math.round(point.x);
                                y = Math.round(point.y);
                            }} else if (point.hasOwnProperty('data')) {{
                                // Alternative format
                                x = Math.round(point.data.x);
                                y = Math.round(point.data.y);
                            }} else if (point.hasOwnProperty('pointNumber')) {{
                                // Use point index as fallback
                                x = point.pointNumber % 100;  // Assume 100x100 grid
                                y = Math.floor(point.pointNumber / 100);
                            }} else {{
                                // Last resort - use pixel coordinates
                                const rect = plotDiv.getBoundingClientRect();
                                x = Math.round(event.xpx - rect.left);
                                y = Math.round(event.ypx - rect.top);
                            }}
                            // Send data to Python
                            sendHoverData(x, y);
                        }}, debounceTime);
                    }}
                    
                    // Set up hover event listener
                    function setupHoverListener() {{
                        if (!plotDiv || hoverListenerActive) return;
                        try {{
                            // Only set up the hover listener if syncing is enabled
                            if (syncCrosshair) {{
                                // Primary method: use Plotly's built-in event system
                                plotDiv.on('plotly_hover', handleHover);
                                
                                // Backup method: add direct mousemove listener to catch all mouse movements
                                plotDiv.addEventListener('mousemove', function(e) {{
                                    // Only process if we haven't received a plotly_hover event recently
                                    if (Date.now() - lastHoverTime > 500) {{
                                        const rect = plotDiv.getBoundingClientRect();
                                        const x = Math.round(e.clientX - rect.left);
                                        const y = Math.round(e.clientY - rect.top);
                                        
                                        // Only send if within the plot area
                                        if (x >= 0 && y >= 0 && x <= rect.width && y <= rect.height) {{
                                            console.log(`PlotlyBridge: Direct mousemove at x=${{x}}, y=${{y}}`);
                                            sendHoverData(x, y);
                                        }}
                                    }}
                                }});
                                
                                hoverListenerActive = true;
                                console.log('Plotly hover listeners activated with crosshair sync enabled');
                            }} else {{
                                console.log('PlotlyBridge: Crosshair sync disabled, hover listeners not activated');
                            }}
                        }} catch (e) {{
                            console.error('Error setting up Plotly hover listeners:', e);
                        }}
                    }}
                    
                    // Update Plotly element reference
                    function updatePlotlyRef() {{
                        plotDiv = findPlotlyElement();
                        if (plotDiv) {{
                            console.log('Found Plotly element:', plotDiv);
                            setupHoverListener();
                        }} else {{
                            console.warn('Plotly element not found, will try again');
                        }}
                    }}
                    
                    // Initialize when DOM is loaded
                    function initialize() {{
                        console.log('Initializing PlotlyBridge');
                        
                        // Try to find Plotly element immediately
                        updatePlotlyRef();
                        
                        // Check again after a short delay
                        setTimeout(updatePlotlyRef, 500);
                        
                        // Check again after a longer delay (to ensure Plotly has fully loaded)
                        setTimeout(updatePlotlyRef, 2000);
                        
                        // Watch for Plotly element to appear
                        const observer = new MutationObserver((mutations) => {{
                            if (!plotDiv) {{
                                updatePlotlyRef();
                            }}
                        }});
                        
                        observer.observe(document.body, {{
                            childList: true,
                            subtree: true
                        }});
                    }}
                    // Initialize when DOM is loaded
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', initialize);
                    }} else {{
                        initialize();
                    }}
                }})();
                </script>
            </div>
            """
            gr.HTML(script, elem_id="plotly-bridge-container")
            
            # Add a debug display for troubleshooting
            with gr.Accordion("PlotlyBridge Debug", open=False):
                gr.Markdown(f"""
                **PlotlyBridge Configuration:**
                - Plot ID: `{self.plot_id}`
                - Sync Crosshair: `{self.sync_crosshair}`
                
                If hover/click events aren't working:
                1. Check browser console for errors
                2. Verify the Plotly element with ID `{self.plot_id}` exists
                3. Check if the hidden input `plotly-hover-data` is present
                """)
                debug_data = gr.JSON({"status": "initialized"})
                debug_btn = gr.Button("Debug PlotlyBridge")
                
                def debug_component():
                    return {
                        "status": "active", 
                        "plot_id": self.plot_id,
                        "sync_crosshair": self.sync_crosshair,
                        "hover_data_exists": True,
                        "timestamp": "now"
                    }
                
                debug_btn.click(fn=debug_component, outputs=[debug_data])

    def get_block_name(self):
        return "plotlybridge"

    # Method to connect event handlers
    def on_hover(self, fn):
        """Set up hover event handler function"""
        # Return a gradio event binding for the hover_data textbox
        logger.info(f"Setting up on_hover handler: {fn.__name__ if fn else 'None'}")
        return self.hover_data.change(fn=fn, inputs=self.hover_data, outputs=None)
    
    def on_click(self, fn):
        """Set up click event handler function (uses the same hidden input as hover)"""
        logger.info(f"Setting up on_click handler: {fn.__name__ if fn else 'None'}")
        return self.hover_data.change(fn=fn, inputs=self.hover_data, outputs=None)
