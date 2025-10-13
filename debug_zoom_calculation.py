#!/usr/bin/env python3
"""
Debug zoom level calculations for lossless quality
"""

def calculate_zoom_levels():
    # Image dimensions (mammography example)
    original_width = 3518
    original_height = 2800
    
    # Viewport dimensions
    viewport_width = 1200
    viewport_height = 800
    
    print("🔍 ZOOM LEVEL ANALYSIS FOR LOSSLESS QUALITY")
    print("=" * 60)
    print(f"Original image: {original_width}×{original_height}")
    print(f"Viewport size: {viewport_width}×{viewport_height}")
    print()
    
    # Calculate minimum zoom (fit entire image)
    min_zoom_x = viewport_width / original_width
    min_zoom_y = viewport_height / original_height
    min_zoom = min(min_zoom_x, min_zoom_y)
    
    print(f"Minimum zoom (fit entire image): {min_zoom:.6f}")
    print(f"  - Zoom X: {min_zoom_x:.6f} (viewport width / original width)")
    print(f"  - Zoom Y: {min_zoom_y:.6f} (viewport height / original height)")
    print()
    
    # Calculate lossless zoom (1:1 pixel mapping)
    lossless_zoom_x = viewport_width / original_width
    lossless_zoom_y = viewport_height / original_height
    lossless_zoom = max(lossless_zoom_x, lossless_zoom_y)
    
    print(f"Lossless zoom (1:1 pixel mapping): {lossless_zoom:.6f}")
    print(f"  - When zoom >= {lossless_zoom:.6f}, we can show pixels at native resolution")
    print()
    
    # Test different zoom levels
    test_zooms = [0.227, 0.682, lossless_zoom, 1.0, 2.0]
    
    for zoom in test_zooms:
        visible_width = viewport_width / zoom
        visible_height = viewport_height / zoom
        
        # Calculate if we can achieve lossless quality at this zoom
        can_show_native = (visible_width <= viewport_width and visible_height <= viewport_height)
        
        # Calculate quality ratio
        scale_x = viewport_width / visible_width
        scale_y = viewport_height / visible_height
        scale = min(scale_x, scale_y)
        quality_ratio = 1.0 / scale if scale > 0 else float('inf')
        
        lossless = scale >= 1.0
        
        print(f"Zoom {zoom:.3f}:")
        print(f"  - Visible region: {visible_width:.0f}×{visible_height:.0f}")
        print(f"  - Scale factor: {scale:.3f}")
        print(f"  - Quality ratio: {quality_ratio:.3f}x")
        print(f"  - Lossless quality: {lossless}")
        print()

if __name__ == "__main__":
    calculate_zoom_levels()