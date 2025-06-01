# filepath: c:\Users\Yurtsever\Downloads\segpro-med\fix_coordinates.py
"""
Script to fix the coordinate display in the PlotlyBridge component.
This script provides instructions to manually fix the coordinate display.
"""

print("========= COORDINATE DISPLAY FIX INSTRUCTIONS =========")
print("\nTo fix the coordinate display in your app, follow these steps:")
print("\n1. In your PlotlyBridge.svelte file:")
print("   - Make sure the sendHoverDataToPython function formats data correctly:")
print("     const hoverData = JSON.stringify({ x, y, timestamp: Date.now() });")

print("\n2. In your plotly_bridge_handler.py file:")
print("   - Update the handle_hover_event and handle_click_event methods to use")
print("     the original coordinates in the return text:")
print("     crosshair_text = f\"x: {x}, y: {y}, z: {self.state.current_slice_idx}\"")

print("\n3. In your app.py file:")
print("   - Use this improved _extract_coordinates_from_event method:")
print("""
def _extract_coordinates_from_event(self, text):
    \"\"\"Extract X and Y coordinates from event response text\"\"\"
    if not text or text == "No data":
        return "X: -- Y: --"
        
    try:
        # Try to extract coords from text like "x: 123, y: 456, z: 7"
        import re
        coords = re.search(r'x: (\\d+), y: (\\d+)', text)
        if coords:
            x, y = coords.groups()
            return f"X: {x} Y: {y}"
            
        # Last resort - try to find any numbers
        numbers = re.findall(r'\\d+', text)
        if len(numbers) >= 2:
            return f"X: {numbers[0]} Y: {numbers[1]}"
    except Exception as e:
        logger.error(f"Error extracting coordinates: {str(e)}")
        
    return "X: -- Y: --"
""")

print("\n4. Make sure the event chain in app.py properly processes coordinates:")
print("""
hover_chain = plotly_bridge.on_hover(
    fn=self.plotly_bridge_handler.handle_hover_event
).then(
    # Pass through the crosshair text
    fn=lambda text: text,
    outputs=[crosshair_info]
).then(
    # Extract coordinates from hover event for display
    fn=lambda text: self._extract_coordinates_from_event(text),
    outputs=[coords_display]
).then(
    # Then update the visualization
    fn=self.plotly_bridge_handler.update_crosshair_display,
    outputs=[image_plot]
)
""")

print("\n========= BEST PRACTICES FOR COORDINATE HANDLING =========")
print("\n1. Return consistent coordinate format from handlers")
print("2. Use JSON for data exchange when possible")
print("3. Extract coordinates as close to the source as possible")
print("4. Use a dedicated component for displaying coordinates")
print("5. Add debugging output to verify data flow")

print("\n==========================================================")