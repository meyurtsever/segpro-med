from pathlib import Path
import os
import gradio as gr

class PlotTools(gr.Blocks):
    """Plot Tools Component for changing Plotly dragmode without reloading the plot."""
    
    def __init__(
        self,
        plot_id=None,
        **kwargs
    ):
        """Initialize the Plot Tools component.
        
        Args:
            plot_id: ID of the plot to control (optional)
        """
        self.plot_id = plot_id
        super().__init__(**kwargs)
        
        with self:
            # Use custom Javascript-enabled HTML to interact with the Plotly plot
            script = """
            <div class="plot-tools-container">
                <h3>Plot Tools</h3>
                <div class="tools-buttons">
                    <button class="tool-button" data-mode="drawcircle">Draw Circle</button>
                    <button class="tool-button" data-mode="drawrect">Draw Rectangle</button>
                    <button class="tool-button" data-mode="drawline">Draw Line</button>
                    <button class="tool-button" data-mode="drawopenpath">Draw Open Path</button>
                    <button class="tool-button" data-mode="drawclosedpath">Draw Closed Path</button>
                    <button class="tool-button" data-mode="eraseshape">Erase Shape</button>
                    <button class="tool-button" data-mode="pan2d">Pan</button>
                    <button class="tool-button" data-mode="zoom2d">Zoom</button>
                    <button class="tool-button" data-mode="resetScale2d">Reset</button>
                </div>
            </div>
            <script>
            (function() {
                // Function to find Plotly plot and set dragmode
                function setPlotlyTool(mode) {
                    console.log('Setting Plotly tool to:', mode);
                    
                    // Try different selectors to find the plot
                    const plotDivs = [
                        ...document.querySelectorAll('.js-plotly-plot'),
                        ...document.querySelectorAll('[data-testid="plotly-graph-div"]')
                    ];
                    
                    for (const div of plotDivs) {
                        try {
                            if (window.Plotly && div) {
                                window.Plotly.relayout(div, {dragmode: mode})
                                    .then(() => console.log('Successfully set mode to:', mode))
                                    .catch(err => console.error('Error setting mode:', err));
                                return;
                            }
                        } catch (e) {
                            console.error('Error updating plot:', e);
                        }
                    }
                    
                    console.warn('No Plotly plot found');
                }
                
                // Set up click handlers for tool buttons
                function setupToolButtons() {
                    const buttons = document.querySelectorAll('.tool-button');
                    
                    buttons.forEach(btn => {
                        btn.addEventListener('click', function() {
                            const mode = this.dataset.mode;
                            setPlotlyTool(mode);
                        });
                    });
                    
                    console.log('Tool buttons initialized');
                }
                
                // Initialize when DOM is loaded
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', setupToolButtons);
                } else {
                    setupToolButtons();
                }
                
                // Setup observer to handle Gradio's dynamic updates
                const observer = new MutationObserver(() => {
                    setupToolButtons();
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            })();
            </script>
            <style>
            .plot-tools-container {
                display: flex;
                flex-direction: column;
                gap: 8px;
                padding: 10px;
                border-radius: 8px;
                background-color: #f5f5f5;
            }
            
            .tools-buttons {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .tool-button {
                padding: 8px 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .tool-button:hover {
                background-color: #e0e0e0;
            }
            </style>
            """
            gr.HTML(script)
            
    def get_block_name(self):
        return "plottools"