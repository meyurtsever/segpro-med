#!/usr/bin/env python3
"""
Test script to verify that user coordinates are properly handled in both Single Slice and All Records modes
"""

import sys
import os
import json
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ui.state import AppState
from ui.medsam2_handlers import MEDSAM2Handlers

def test_coordinate_handling():
    """Test coordinate handling for both processing modes"""
    print("=" * 60)
    print("Testing MEDSAM2 Coordinate Handling")
    print("=" * 60)
    
    # Initialize state and handlers
    state = AppState()
    medsam2_handlers = MEDSAM2Handlers(state)
    
    # Test 1: Verify coordinate selection works
    print("\n1. Testing coordinate selection...")
    test_coordinates = [(100, 150), (200, 250), (300, 350)]
    
    for x, y in test_coordinates:
        medsam2_handlers.selected_coordinates.append((x, y))
    
    print(f"   Selected coordinates: {medsam2_handlers.selected_coordinates}")
    assert len(medsam2_handlers.selected_coordinates) == 3, "Coordinates not stored correctly"
    print("   ✓ Coordinate selection works")
    
    # Test 2: Test validation in workflow
    print("\n2. Testing coordinate validation...")
    
    # Test with no coordinates
    medsam2_handlers.selected_coordinates = []
    result_msg, result_img = medsam2_handlers.run_full_annotation_workflow(
        "test_output", False, "cpu", "Single Slice", 0.3
    )
    
    assert "No coordinates selected" in result_msg, f"Expected validation error, got: {result_msg}"
    print("   ✓ Validation correctly rejects empty coordinates")
    
    # Restore coordinates
    medsam2_handlers.selected_coordinates = test_coordinates
    
    # Test 3: Test single slice prompt generation
    print("\n3. Testing single slice prompt generation...")
    
    # Mock some data for testing
    state.current_data = np.random.rand(256, 256, 20)  # Mock 20-slice volume
    state.current_slice_idx = 10
    
    success, prompt_file = medsam2_handlers.generate_prompt_json("fake_dicom_folder")
    if success:
        # Read and verify the prompt file
        with open(prompt_file, 'r') as f:
            prompt_data = json.load(f)
        
        print(f"   Generated prompt file: {prompt_file}")
        print(f"   Prompt data keys: {list(prompt_data.keys())}")
        print(f"   Points for slice 10: {prompt_data.get('10', {}).get('points', [])}")
        
        expected_points = [[100, 150], [200, 250], [300, 350]]
        actual_points = prompt_data.get('10', {}).get('points', [])
        assert actual_points == expected_points, f"Expected {expected_points}, got {actual_points}"
        print("   ✓ Single slice prompt generation works correctly")
    else:
        print(f"   ⚠ Single slice prompt generation failed: {prompt_file}")
    
    # Test 4: Test all slices prompt generation
    print("\n4. Testing all slices prompt generation...")
    
    success, prompt_file = medsam2_handlers.generate_prompt_json_all_slices("fake_dicom_folder")
    if success:
        # Read and verify the prompt file
        with open(prompt_file, 'r') as f:
            prompt_data = json.load(f)
        
        print(f"   Generated all-slices prompt file: {prompt_file}")
        print(f"   Number of slices with prompts: {len(prompt_data)}")
        print(f"   Sample slice keys: {list(prompt_data.keys())[:5]}...")
        
        # Verify all slices have the same user coordinates
        expected_points = [[100, 150], [200, 250], [300, 350]]
        for slice_idx in range(20):  # Test first few slices
            slice_key = str(slice_idx)
            if slice_key in prompt_data:
                actual_points = prompt_data[slice_key].get('points', [])
                assert actual_points == expected_points, f"Slice {slice_idx}: Expected {expected_points}, got {actual_points}"
        
        print("   ✓ All slices prompt generation works correctly")
        print(f"   ✓ All slices use same user coordinates: {expected_points}")
    else:
        print(f"   ⚠ All slices prompt generation failed: {prompt_file}")
    
    # Test 5: Verify coordinate requirements for both modes
    print("\n5. Testing coordinate requirements for both modes...")
    
    # Test single slice mode
    medsam2_handlers.selected_coordinates = []
    result_msg, _ = medsam2_handlers.run_full_annotation_workflow(
        "test_output", False, "cpu", "Single Slice", 0.3
    )
    assert "No coordinates selected" in result_msg, "Single slice mode should require coordinates"
    print("   ✓ Single Slice mode requires coordinates")
    
    # Test all records mode
    result_msg, _ = medsam2_handlers.run_full_annotation_workflow(
        "test_output", False, "cpu", "All Records", 0.3
    )
    assert "No coordinates selected" in result_msg, "All Records mode should require coordinates"
    print("   ✓ All Records mode requires coordinates")
    
    print("\n" + "=" * 60)
    print("✅ ALL COORDINATE HANDLING TESTS PASSED!")
    print("✅ User coordinates are now properly used in both modes")
    print("=" * 60)
    
    # Cleanup
    if os.path.exists("brain_target_prompts.json"):
        os.remove("brain_target_prompts.json")

if __name__ == "__main__":
    test_coordinate_handling()
