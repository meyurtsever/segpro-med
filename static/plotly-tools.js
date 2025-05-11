
// Plotly tools handler for SegMed-Pro
document.addEventListener('DOMContentLoaded', function() {
    console.log('Plotly tools handler loaded');
    
    // Set up tool button handlers
    function setupToolButtons() {
        console.log('Setting up tool buttons');
        setTimeout(function() {
            const toolButtons = {
                'draw_circle_btn': 'drawcircle',
                'draw_rect_btn': 'drawrect',
                'draw_line_btn': 'drawline',
                'draw_openpath_btn': 'drawopenpath',
                'draw_closedpath_btn': 'drawclosedpath',
                'erase_shape_btn': 'eraseshape',
                'pan_btn': 'pan2d',
                'zoom_btn': 'zoom2d',
                'reset_btn': 'resetScale2d'
            };
            
            for (const [btnId, toolMode] of Object.entries(toolButtons)) {
                const btn = document.getElementById(btnId);
                if (btn) {
                    console.log('Found button:', btnId);
                    btn.addEventListener('click', function() {
                        console.log('Button clicked:', btnId, 'Setting mode to:', toolMode);
                        updatePlotlyTool(toolMode);
                    });
                } else {
                    console.warn('Button not found:', btnId);
                }
            }
        }, 1000); // Wait 1 second for DOM to fully render
    }
    
    // Update Plotly tool
    function updatePlotlyTool(mode) {
        const plotDiv = document.querySelector('div[data-testid="plotly"]');
        if (!plotDiv) {
            console.warn('Plotly div not found, trying alternative selectors');
            const plotlyElements = document.querySelectorAll('div[class*="plotly"]');
            if (plotlyElements.length > 0) {
                console.log('Found possible plotly elements:', plotlyElements.length);
                for (const el of plotlyElements) {
                    if (el._fullLayout) {
                        console.log('Found plotly element with _fullLayout');
                        Plotly.relayout(el, {dragmode: mode});
                        return;
                    }
                }
            }
            
            // Final attempt - try with any plotly-graph-div
            const lastAttempt = document.querySelector('div[data-testid="plotly-graph-div"]');
            if (lastAttempt) {
                console.log('Found plotly-graph-div');
                Plotly.relayout(lastAttempt, {dragmode: mode});
                return;
            }
            
            console.error('No plotly element found after multiple attempts');
        } else {
            console.log('Found plotly div directly');
            Plotly.relayout(plotDiv, {dragmode: mode});
        }
    }
    
    // Set up the button handlers
    setupToolButtons();
    
    // Also set up handlers whenever Gradio refreshes the UI
    const observer = new MutationObserver(function(mutations) {
        console.log('DOM mutation detected, checking for new buttons');
        setupToolButtons();
    });
    
    observer.observe(document, {
        childList: true,
        subtree: true
    });
});
