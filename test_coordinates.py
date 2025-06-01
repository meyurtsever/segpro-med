# Add a simple debug file to test coordinate handling directly
# filepath: c:\Users\Yurtsever\Downloads\segpro-med\test_coordinates.py

"""
Test script for coordinate extraction from Plotly events.
This script helps diagnose issues with coordinate handling.
"""

import json
import re

def extract_coordinates_from_text(text):
    """Extract X and Y coordinates from event response text"""
    print(f"Extracting coordinates from: {text}")
    
    if not text or text == "No data":
        return "X: -- Y: --"
        
    try:
        # Try to extract coords from text like "x: 123, y: 456, z: 7"
        coords = re.search(r'x: (\d+), y: (\d+)', text)
        if coords:
            x, y = coords.groups()
            return f"X: {x} Y: {y}"
        else:
            return "No match found"
    except Exception as e:
        return f"Error: {str(e)}"

# Test different coordinate formats
test_cases = [
    "x: 123, y: 456, z: 7",
    "Hover at x=42, y=73",
    "Invalid format",
    "x: 10, y: 20"
]

print("Testing coordinate extraction from various formats:")
for case in test_cases:
    result = extract_coordinates_from_text(case)
    print(f"Input: '{case}' → Output: '{result}'")

print("\nTesting coordinate extraction from JSON:")
json_data = '{"x": 50, "y": 60, "timestamp": 1633245600000}'
try:
    data = json.loads(json_data)
    x, y = data.get('x'), data.get('y')
    formatted = f"X: {x} Y: {y}"
    print(f"Parsed JSON: {data}")
    print(f"Formatted output: {formatted}")
except Exception as e:
    print(f"Error parsing JSON: {e}")

print("\nBest Practice for Coordinate Handling:")
print("1. Return coordinates in a consistent format from handlers (e.g., 'x: 123, y: 456, z: 7')")
print("2. Use a dedicated extraction function to parse the format")
print("3. Provide direct access to raw coordinates via the state object")
print("4. Add extensive logging throughout the coordinate handling flow")