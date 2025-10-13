#!/usr/bin/env python3
"""
Debug viewport aspect ratio calculation
"""
import sys
import os
import numpy as np

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.image_viewport import create_mammography_viewport

def debug_viewport_aspect():
    """Debug the viewport aspect ratio issue"""
    print("🔍 DEBUG VIEWPORT ASPECT RATIO CALCULATION")
    print("=" * 60)
    
    # Create mock mammography image with known aspect ratio
    mock_mammo = np.random.randint(0, 255, (3518, 2800, 3), dtype=np.uint8)
    original_height, original_width = mock_mammo.shape[:2]
    original_aspect = original_width / original_height  # Width/Height = 2800/3518 = 0.796
    
    print(f"Original image: {original_width}×{original_height}")
    print(f"Original aspect ratio: {original_aspect:.4f}")
    print()
    
    # Create viewport with current settings (1200x600)
    viewport = create_mammography_viewport(mock_mammo, viewport_width=1200, viewport_height=600)
    viewport_aspect = viewport.viewport_width / viewport.viewport_height
    print(f"Viewport dimensions: {viewport.viewport_width}×{viewport.viewport_height}")
    print(f"Viewport aspect ratio: {viewport_aspect:.4f}")
    print()
    
    # Get viewport info to see internal calculations
    info = viewport.get_viewport_info()
    print("Viewport info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Get the viewport image
    viewport_image = viewport.get_viewport_image()
    print(f"Viewport image shape: {viewport_image.shape}")
    
    # Find actual image content (non-black pixels)
    gray = np.mean(viewport_image, axis=2)
    non_black_mask = gray > 1
    
    if np.any(non_black_mask):
        # Find bounds of actual image content
        rows = np.any(non_black_mask, axis=1)
        cols = np.any(non_black_mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            displayed_width = cmax - cmin + 1
            displayed_height = rmax - rmin + 1
            displayed_aspect = displayed_width / displayed_height
            
            print(f"Displayed content bounds: ({cmin}, {rmin}) to ({cmax}, {rmax})")
            print(f"Displayed content size: {displayed_width}×{displayed_height}")
            print(f"Displayed aspect ratio: {displayed_aspect:.4f}")
            
            aspect_error = abs(displayed_aspect - original_aspect) / original_aspect
            print(f"Aspect error: {aspect_error:.4f} ({aspect_error*100:.2f}%)")
            
            if aspect_error < 0.01:
                print("✅ Aspect ratio preserved correctly")
            else:
                print("❌ Aspect ratio preservation failed")
                
                # Debug the aspect ratio calculation
                print("\nDebugging aspect ratio calculation:")
                region_aspect = original_aspect
                viewport_aspect = viewport.viewport_width / viewport.viewport_height
                print(f"  Region aspect: {region_aspect:.4f}")
                print(f"  Viewport aspect: {viewport_aspect:.4f}")
                
                if region_aspect > viewport_aspect:
                    print("  Logic: Region is wider - fit to viewport width")
                    expected_target_width = viewport.viewport_width
                    expected_target_height = int(viewport.viewport_width / region_aspect)
                    print(f"  Expected target: {expected_target_width}×{expected_target_height}")
                else:
                    print("  Logic: Region is taller - fit to viewport height")
                    expected_target_width = int(viewport.viewport_height * region_aspect)
                    expected_target_height = viewport.viewport_height
                    print(f"  Expected target: {expected_target_width}×{expected_target_height}")
                    expected_aspect = expected_target_width / expected_target_height
                    print(f"  Expected aspect: {expected_aspect:.4f}")
        else:
            print("❌ Could not find image content bounds")
    else:
        print("❌ No non-black pixels found in viewport image")

if __name__ == "__main__":
    debug_viewport_aspect()