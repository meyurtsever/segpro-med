<script>
  // PlotlyBridge.svelte - A Svelte component to bridge Plotly with Python backend
  import { onMount, onDestroy } from "svelte";
  import { createEventDispatcher } from "svelte";

  // Props
  export let plotId = "";        // ID of the Plotly element to watch
  export let syncCrosshair = true; // Whether to sync hover data with crosshair

  // Local state
  let plotDiv = null;
  let hoverListenerActive = false;
  let debounceTimeout = null;
  const debounceTime = 50; // ms to debounce hover events
  
  // Create event dispatcher to communicate with Python
  const dispatch = createEventDispatcher();
    // Function to find the Plotly element in the DOM
  function findPlotlyElement() {
    // If plotId is provided, try to find element by ID first
    if (plotId) {
      const elementById = document.getElementById(plotId);
      if (elementById && (elementById._fullLayout || elementById.querySelector('.main-svg'))) {
        console.log(`PlotlyBridge: Found Plotly element by ID: ${plotId}`);
        return elementById;
      } else {
        console.log(`PlotlyBridge: Element with ID ${plotId} not found or not a Plotly element`);
      }
    }
    
    // Try different selectors that might match the Plotly container
    const selectors = [
      '.js-plotly-plot', 
      '[data-testid="plotly"]',
      '[data-testid="plotly-graph-div"]',
      '.plotly-graph-div'
    ];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      for (const el of elements) {
        if (el._fullLayout) {
          return el;
        }
      }
    }
    
    // Final attempt - any element with Plotly-specific children
    const possiblePlots = document.querySelectorAll('div');
    for (const div of possiblePlots) {
      if (div.querySelector('.main-svg') || 
          div.querySelector('.plotly') || 
          div._fullLayout) {
        return div;
      }
    }
    
    return null;
  }  // Handler for Plotly hover events
  function handleHover(event) {
    if (!event || !event.points || !event.points.length) {
      console.log('PlotlyBridge: Invalid hover event received', event);
      return;
    }
    console.log("PlotlyBridge: Hover event detected!", event);
    if (!syncCrosshair) {
      console.log('PlotlyBridge: Crosshair sync disabled, ignoring hover event');
      return;
    }
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      const point = event.points[0];
      let x, y;
      if (point.hasOwnProperty('x') && point.hasOwnProperty('y')) {
        x = Math.round(point.x);
        y = Math.round(point.y);
      } else if (point.hasOwnProperty('data')) {
        x = Math.round(point.data.x);
        y = Math.round(point.data.y);
      } else if (point.hasOwnProperty('pointNumber')) {
        x = point.pointNumber % 100;
        y = Math.floor(point.pointNumber / 100);
      } else {
        const rect = plotDiv.getBoundingClientRect();
        x = Math.round(event.xpx - rect.left);
        y = Math.round(event.ypx - rect.top);
      }
      console.log(`PlotlyBridge: Processing hover at x: ${x}, y: ${y}`);
      // Dispatch event to Python (Gradio expects hidden input update)
      sendHoverDataToPython(x, y);
      // Still dispatch Svelte event for local use
      dispatch('hover', {
        x: x,
        y: y,
        plotX: point.x,
        plotY: point.y,
        curveNumber: point.curveNumber,
        pointNumber: point.pointNumber
      });
      console.log(`PlotlyBridge: Dispatched hover event at x: ${x}, y: ${y}`);
    }, debounceTime);
  }

  // Send hover data to Python via hidden input
  function sendHoverDataToPython(x, y) {
    const hoverDataElement = document.getElementById('plotly-hover-data');
    if (!hoverDataElement) {
      console.error('PlotlyBridge: Could not find hover data element with ID plotly-hover-data');
      return;
    }
    
    // Forcefully set the value and trigger events
    const hoverData = JSON.stringify({ x, y, timestamp: Date.now() });
    console.log('DEBUGDATA: About to write to hidden input:', hoverData);
    
    // Set value and dispatch events with a small delay to ensure they're processed
    hoverDataElement.value = hoverData;
    
    // Dispatch events in this specific order for Gradio
    setTimeout(() => {
      console.log('DEBUGDATA: Dispatching input event');
      hoverDataElement.dispatchEvent(new Event('input', { bubbles: true }));
      
      setTimeout(() => {
        console.log('DEBUGDATA: Dispatching change event');
        hoverDataElement.dispatchEvent(new Event('change', { bubbles: true }));
        
        // Verify the value was set
        console.log('DEBUGDATA: Input value after events:', hoverDataElement.value);
      }, 10);
    }, 10);
  }
    // Handle Plotly click events
  function handleClick(event) {
    if (!event || !event.points || !event.points.length) {
      console.log('PlotlyBridge: Invalid click event received', event);
      return;
    }
    console.log("PlotlyBridge: Click event detected!", event);
    const point = event.points[0];
    let x, y;
    if (point.hasOwnProperty('x') && point.hasOwnProperty('y')) {
      x = Math.round(point.x);
      y = Math.round(point.y);
    } else if (point.hasOwnProperty('data')) {
      x = Math.round(point.data.x);
      y = Math.round(point.data.y);
    } else if (point.hasOwnProperty('pointNumber')) {
      x = point.pointNumber % 100;
      y = Math.floor(point.pointNumber / 100);
    } else {
      const rect = plotDiv.getBoundingClientRect();
      x = Math.round(event.xpx - rect.left);
      y = Math.round(event.ypx - rect.top);
    }
    console.log(`PlotlyBridge: Processing click at x: ${x}, y: ${y}`);
    sendHoverDataToPython(x, y);
    dispatch('click', {
      x: x,
      y: y,
      plotX: point.x,
      plotY: point.y,
      curveNumber: point.curveNumber,
      pointNumber: point.pointNumber
    });
    console.log(`PlotlyBridge: Dispatched click event at x: ${x}, y: ${y}`);
  }

  // Set up hover event listener
  function setupHoverListener() {
    if (!plotDiv || hoverListenerActive) return;
    
    try {
      // Only set up the hover listener if syncing is enabled
      if (syncCrosshair) {
        plotDiv.on('plotly_hover', handleHover);
        plotDiv.on('plotly_click', handleClick); // <-- Add click listener
        hoverListenerActive = true;
        console.log('PlotlyBridge: Hover and click listeners activated with crosshair sync enabled');
      } else {
        console.log('PlotlyBridge: Crosshair sync disabled, hover/click listeners not activated');
      }
    } catch (e) {
      console.error('PlotlyBridge: Error setting up Plotly hover/click listeners:', e);
    }
  }
  
  // Remove event listeners
  function removeListeners() {
    if (plotDiv && hoverListenerActive) {
      try {
        plotDiv.removeListener('plotly_hover', handleHover);
        plotDiv.removeListener('plotly_click', handleClick); // <-- Remove click listener
        hoverListenerActive = false;
        console.log('Plotly hover and click listeners removed');
      } catch (e) {
        console.error('Error removing Plotly hover/click listeners:', e);
      }
    }
  }
  
  // Update Plotly element reference
  function updatePlotlyRef() {
    console.log('Looking for Plotly element with ID:', plotId);
    plotDiv = findPlotlyElement();
    if (plotDiv) {
      console.log('Found Plotly element:', plotDiv);
      
      // Debug information about the element
      if (plotDiv._fullLayout) {
        console.log('Plotly element has _fullLayout, looks valid');
      } else {
        console.log('WARNING: Plotly element found but may not be fully initialized');
      }
      
      setupHoverListener();
    } else {
      console.warn('Plotly element not found, will try again');
      
      // Additional debugging
      if (plotId) {
        const anyElement = document.getElementById(plotId);
        if (anyElement) {
          console.log('Found element with matching ID but it may not be a Plotly element:', anyElement);
        } else {
          console.log('No element with ID', plotId, 'exists in the DOM');
        }
      }
      
      // List all potential plotly elements for debugging
      const potentialPlots = document.querySelectorAll('.js-plotly-plot, .plotly, .main-svg');
      console.log('Potential Plotly elements found:', potentialPlots.length);
    }
  }
  
  // Watch for Plotly element to appear in DOM
  function watchForPlotly() {
    const observer = new MutationObserver((mutations) => {
      if (!plotDiv) {
        updatePlotlyRef();
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
      return observer;
  }
  
  // React to changes in syncCrosshair property
  $: {
    // If plotDiv exists and the component is mounted, update listener based on syncCrosshair
    if (plotDiv) {
      if (syncCrosshair && !hoverListenerActive) {
        // Set up listener if syncCrosshair was enabled
        plotDiv.on('plotly_hover', handleHover);
        plotDiv.on('plotly_click', handleClick); // <-- Add click listener
        hoverListenerActive = true;
        console.log('PlotlyBridge: Hover listener activated (property changed)');
      } else if (!syncCrosshair && hoverListenerActive) {
        // Remove listener if syncCrosshair was disabled
        plotDiv.removeListener('plotly_hover', handleHover);
        plotDiv.removeListener('plotly_click', handleClick); // <-- Remove click listener
        hoverListenerActive = false;
        console.log('PlotlyBridge: Hover listener deactivated (property changed)');
      }
    }
  }
    // Component lifecycle
  onMount(() => {
    console.log('PlotlyBridge component mounted');
    
    // Add a 100ms delay to ensure the DOM is fully loaded
    setTimeout(() => {
      // Check if our hidden input exists
      const hiddenInput = document.getElementById('plotly-hover-data');
      console.log('Hidden input element exists:', hiddenInput !== null);
      
      // First attempt to find Plotly element
      updatePlotlyRef();
      
      // Set up periodic checks for the Plotly element
      const intervalId = setInterval(() => {
        if (!plotDiv) {
          console.log('Periodic check: Looking for Plotly element');
          updatePlotlyRef();
        } else {
          clearInterval(intervalId);
        }
      }, 2000);
    }, 100);
  });
  
  onDestroy(() => {
    console.log('PlotlyBridge component unmounting');
    removeListeners();
  });
</script>

<div class="plotly-bridge-container">
  <input id="plotly-hover-data" type="text" style="display:none" autocomplete="off" />
  <div class="debug-info">
    <!-- This will show if the hidden input is present -->
    <p style="font-size:10px; color:#999; margin:0; padding:2px;">
      PlotlyBridge active. Hidden input: {document.getElementById('plotly-hover-data') ? 'present' : 'missing'}
    </p>
  </div>
</div>

<style>
  /* Make container visible but hide input using inline style */
  .plotly-bridge-container {
    position: relative;
    width: 100%;
    min-height: 20px;
  }
  
  .debug-info {
    position: absolute;
    bottom: 0;
    right: 0;
    background: rgba(255,255,255,0.7);
    z-index: 1000;
  }
  
  .loading-message {
    font-size: 12px;
    color: #666;
    padding: 5px;
  }
</style>
