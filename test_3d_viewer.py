"""
Test script for the 3D viewer functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ui.editor_tab import create_3d_visualization, create_empty_3d_plot

def test_3d_viewer():
    """Test the 3D viewer with existing annotation data"""
    print("Testing 3D viewer functionality...")
    
    # Test with empty plot
    print("Creating empty 3D plot...")
    empty_fig = create_empty_3d_plot("Test message")
    print(f"Empty plot created successfully: {type(empty_fig)}")
    
    # Test with actual data if available
    try:
        print("Testing with annotation data...")
        fig = create_3d_visualization("brain_target_results")
        print(f"3D visualization created successfully: {type(fig)}")
        
        # Save test HTML
        test_path = "test_3d_view.html"
        fig.write_html(test_path)
        print(f"Test 3D view saved to: {test_path}")
        
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")

if __name__ == "__main__":
    test_3d_viewer()
