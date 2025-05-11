<script>
  // Plot Tools component for interacting with Plotly
  import { onMount, onDestroy } from "svelte";

  // Props
  export let plotId = ""; // ID of the Plotly plot to control

  // State
  let plotDiv = null;
  let toolButtons = [
    { label: "Draw Circle", mode: "drawcircle" },
    { label: "Draw Rectangle", mode: "drawrect" },
    { label: "Draw Line", mode: "drawline" },
    { label: "Draw Open Path", mode: "drawopenpath" },
    { label: "Draw Closed Path", mode: "drawclosedpath" },
    { label: "Erase Shape", mode: "eraseshape" },
    { label: "Pan", mode: "pan2d" },
    { label: "Zoom", mode: "zoom2d" },
    { label: "Reset", mode: "resetScale2d" }
  ];

  // Function to set the Plotly dragmode
  function setPlotlyTool(mode) {
    console.log(`Setting Plotly tool to: ${mode}`);
    
    const findPlotly = () => {
      // Try different selectors to find the Plotly element
      const selectors = [
        '.js-plotly-plot', 
        '[data-testid="plotly"]',
        '[data-testid="plotly-graph-div"]'
      ];
      
      for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);
        for (const el of elements) {
          if (el._fullLayout) {
            return el;
          }
        }
      }
      
      // Final attempt - any element with an SVG child that looks like Plotly
      const possiblePlots = document.querySelectorAll('div');
      for (const div of possiblePlots) {
        if (div.querySelector('.main-svg') || div.querySelector('.plotly')) {
          return div;
        }
      }
      
      return null;
    };

    // Find the Plotly element
    plotDiv = findPlotly();
    
    if (plotDiv && window.Plotly) {
      try {
        window.Plotly.relayout(plotDiv, {dragmode: mode})
          .then(() => console.log(`Successfully set mode to ${mode}`))
          .catch(err => {
            console.error("Error setting Plotly mode:", err);
            // Fallback to direct property setting if relayout fails
            if (plotDiv._fullLayout) {
              plotDiv._fullLayout.dragmode = mode;
            }
          });
      } catch (e) {
        console.error("Error using Plotly.relayout:", e);
      }
    } else {
      console.error("Could not find Plotly element or Plotly is not loaded");
    }
  }

  // Set up monitoring for Plotly element when component mounts
  onMount(() => {
    // Try to find the Plotly element immediately
    const checkPlotly = () => {
      plotDiv = document.querySelector('.js-plotly-plot');
      if (!plotDiv) {
        setTimeout(checkPlotly, 500);
      } else {
        console.log("Found Plotly element:", plotDiv);
      }
    };
    
    checkPlotly();
    
    // Also set up observer to detect Plotly load
    const observer = new MutationObserver(() => {
      if (!plotDiv) {
        plotDiv = document.querySelector('.js-plotly-plot');
        if (plotDiv) {
          console.log("Plotly element detected by observer");
          observer.disconnect();
        }
      }
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    return () => {
      observer.disconnect();
    };
  });
</script>

<div class="plot-tools-container">
  <h3>Plot Tools</h3>
  <div class="tools-buttons">
    {#each toolButtons as button}
      <button on:click={() => setPlotlyTool(button.mode)} class="tool-button">
        {button.label}
      </button>
    {/each}
  </div>
</div>

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
