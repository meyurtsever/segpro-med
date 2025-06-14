#!/usr/bin/env python3
"""
Test script to verify annotation persistence and fingerprinting logic.
"""

import sys
import os
import logging

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ui.state import AppState
from ui.image_handlers import ImagePlotToolHandlers

def test_annotation_fingerprinting():
    """Test the annotation fingerprinting logic"""
    print("Testing annotation fingerprinting logic...")
    
    # Create a mock state
    state = AppState()
    
    # Create the image handlers
    handlers = ImagePlotToolHandlers(state)
    
    # Test case 1: Empty annotations
    empty_value = None
    fingerprint1 = handlers._get_annotation_fingerprint(empty_value)
    print(f"Empty annotation fingerprint: '{fingerprint1}'")
    
    # Test case 2: Simple annotation
    simple_annotation = {
        "image": None,
        "boxes": [
            {
                "type": "rectangle",
                "label": "Test Annotation",
                "color": "#FF0000",
                "coordinates": [10, 10, 50, 50]
            }
        ]
    }
    fingerprint2 = handlers._get_annotation_fingerprint(simple_annotation)
    print(f"Simple annotation fingerprint: '{fingerprint2}'")
    
    # Test case 3: Same annotation (should have same fingerprint)
    same_annotation = {
        "image": None,
        "boxes": [
            {
                "type": "rectangle",
                "label": "Test Annotation",
                "color": "#FF0000",
                "coordinates": [10, 10, 50, 50]
            }
        ]
    }
    fingerprint3 = handlers._get_annotation_fingerprint(same_annotation)
    print(f"Same annotation fingerprint: '{fingerprint3}'")
    
    # Test case 4: Modified annotation (should have different fingerprint)
    modified_annotation = {
        "image": None,
        "boxes": [
            {
                "type": "rectangle",
                "label": "Modified Annotation",  # Changed label
                "color": "#FF0000",
                "coordinates": [10, 10, 50, 50]
            }
        ]
    }
    fingerprint4 = handlers._get_annotation_fingerprint(modified_annotation)
    print(f"Modified annotation fingerprint: '{fingerprint4}'")
    
    # Test fingerprint comparisons
    print(f"\nFingerprint comparison tests:")
    print(f"Simple == Same: {fingerprint2 == fingerprint3}")
    print(f"Simple == Modified: {fingerprint2 == fingerprint4}")
    print(f"Empty == Simple: {fingerprint1 == fingerprint2}")
    
    # Test save logic with fingerprints
    print(f"\nTesting save logic...")
    
    # Simulate loading an annotation
    handlers._loaded_fingerprints[0] = fingerprint2
    print(f"Stored fingerprint for slice 0: '{fingerprint2}'")
    
    # Test saving same annotation (should skip)
    print(f"Testing save of same annotation...")
    handlers.save_user_annotations(simple_annotation, 0)
    
    # Test saving modified annotation (should save)
    print(f"Testing save of modified annotation...")
    handlers.save_user_annotations(modified_annotation, 0)
    
    # Check user annotations
    print(f"User annotations stored: {len(handlers.user_annotations)}")
    if 0 in handlers.user_annotations:
        print(f"Slice 0 has {len(handlers.user_annotations[0])} user annotations")
    
    print("Test completed!")

if __name__ == "__main__":
    test_annotation_fingerprinting()
