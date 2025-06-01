# filepath: c:\Users\Yurtsever\Downloads\segpro-med\utils\coordinate_helpers.py
"""
Helper functions for working with coordinates from Plotly events.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_coordinates_from_text(text):
    """
    Extract X and Y coordinates from event response text.
    
    Args:
        text: String containing coordinate information
        
    Returns:
        str: Formatted X and Y coordinates
    """
    if not text or text == "No data":
        return "X: -- Y: --"
        
    try:
        # First try to parse as JSON
        try:
            data = json.loads(text)
            x, y = data.get('x'), data.get('y')
            if x is not None and y is not None:
                return f"X: {x} Y: {y}"
        except json.JSONDecodeError:
            pass
            
        # If not JSON, try to extract from text format
        coords = re.search(r'x: (\d+), y: (\d+)', text)
        if coords:
            x, y = coords.groups()
            return f"X: {x} Y: {y}"
            
        # Try another common format
        coords = re.search(r'x=(\d+).*y=(\d+)', text)
        if coords:
            x, y = coords.groups()
            return f"X: {x} Y: {y}"
            
        # Last resort - try to find any numbers
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            return f"X: {numbers[0]} Y: {numbers[1]}"
    except Exception as e:
        logger.error(f"Error extracting coordinates: {str(e)}")
        
    return "X: -- Y: --"

def format_hover_data(hover_data_str):
    """
    Format hover data from PlotlyBridge for display.
    
    Args:
        hover_data_str: String containing hover data from PlotlyBridge
        
    Returns:
        str: Formatted coordinate string
    """
    try:
        data = json.loads(hover_data_str)
        x = data.get('x', '--')
        y = data.get('y', '--')
        return f"X: {x} Y: {y}"
    except Exception as e:
        logger.error(f"Error formatting hover data: {str(e)}")
        return extract_coordinates_from_text(hover_data_str)
        
def format_crosshair_position(position):
    """
    Format a crosshair position tuple for display.
    
    Args:
        position: Tuple of (x, y, z) coordinates
        
    Returns:
        str: Formatted coordinate string
    """
    try:
        x, y, _ = position
        return f"X: {x} Y: {y}"
    except Exception as e:
        logger.error(f"Error formatting crosshair position: {str(e)}")
        return "X: -- Y: --"