#!/usr/bin/env python3
"""
Brain Segmentation Configuration Tester

This script allows easy testing of different SAM2 configurations
for brain structure segmentation tasks.

Usage:
    python test_brain_configs.py --config balanced
    python test_brain_configs.py --config small_structures --slices 5
    python test_brain_configs.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from brain_segmentation_configs import BRAIN_CONFIGS, CONFIG_DESCRIPTIONS, print_config_info

def modify_test_file(config_name, max_slices=None, save_viz=None):
    """
    Modify the test_comprehensive_mask_generator.py file to use specified configuration.
    
    Args:
        config_name (str): Configuration name to use
        max_slices (int, optional): Maximum number of slices to process
        save_viz (bool, optional): Whether to save visualizations
    """
    test_file_path = current_dir / "test_comprehensive_mask_generator.py"
    
    if not test_file_path.exists():
        print(f"❌ Test file not found: {test_file_path}")
        return False
    
    # Read the current file
    with open(test_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace configuration
    old_config_line = None
    new_config_line = f"BRAIN_CONFIG_MODE = '{config_name}'  # Change this to experiment with different configurations"
    
    # Find the line to replace
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith("BRAIN_CONFIG_MODE = "):
            old_config_line = i
            break
    
    if old_config_line is not None:
        lines[old_config_line] = new_config_line
        content = '\n'.join(lines)
    
    # Replace max slices if specified
    if max_slices is not None:
        for i, line in enumerate(lines):
            if line.strip().startswith("MAX_SLICES_TO_PROCESS = "):
                lines[i] = f"MAX_SLICES_TO_PROCESS = {max_slices}   # Limit number of slices for testing"
                break
        content = '\n'.join(lines)
    
    # Replace save visualizations if specified
    if save_viz is not None:
        for i, line in enumerate(lines):
            if line.strip().startswith("SAVE_VISUALIZATIONS = "):
                lines[i] = f"SAVE_VISUALIZATIONS = {save_viz}  # Set to True to save mask visualizations (slower)"
                break
        content = '\n'.join(lines)
    
    # Write the modified content
    try:
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated test file to use '{config_name}' configuration")
        if max_slices is not None:
            print(f"   • Max slices: {max_slices}")
        if save_viz is not None:
            print(f"   • Save visualizations: {save_viz}")
        return True
    except Exception as e:
        print(f"❌ Error updating test file: {e}")
        return False

def run_test():
    """Run the test with current configuration."""
    test_file_path = current_dir / "test_comprehensive_mask_generator.py"
    
    if not test_file_path.exists():
        print(f"❌ Test file not found: {test_file_path}")
        return False
    
    print("🚀 Running brain segmentation test...")
    print("=" * 50)
    
    # Import and run the test
    try:
        import subprocess
        result = subprocess.run([sys.executable, str(test_file_path)], 
                              capture_output=False, text=True, cwd=str(current_dir))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test different SAM2 brain segmentation configurations")
    parser.add_argument('--config', '-c', choices=list(BRAIN_CONFIGS.keys()), 
                       default='balanced', help='Configuration to use (default: balanced)')
    parser.add_argument('--slices', '-s', type=int, help='Maximum number of slices to process')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization saving (faster)')
    parser.add_argument('--viz', action='store_true', help='Enable visualization saving')
    parser.add_argument('--list', '-l', action='store_true', help='List available configurations')
    parser.add_argument('--run', '-r', action='store_true', help='Run the test after configuration')
    
    args = parser.parse_args()
    
    if args.list:
        print_config_info()
        return
    
    # Validate configuration
    if args.config not in BRAIN_CONFIGS:
        print(f"❌ Unknown configuration: {args.config}")
        print(f"Available configurations: {list(BRAIN_CONFIGS.keys())}")
        return
    
    # Determine visualization setting
    save_viz = None
    if args.no_viz:
        save_viz = False
    elif args.viz:
        save_viz = True
    
    # Show configuration info
    print(f"🧠 Selected configuration: {args.config}")
    print(f"   Description: {CONFIG_DESCRIPTIONS[args.config]}")
    config = BRAIN_CONFIGS[args.config]
    print(f"   Points per side: {config['points_per_side']}")
    print(f"   IoU threshold: {config['pred_iou_thresh']}")
    print(f"   Min area: {config['min_mask_region_area']}")
    print(f"   Use M2M: {config['use_m2m']}")
    print(f"   Crop layers: {config['crop_n_layers']}")
    print()
    
    # Update test file
    success = modify_test_file(args.config, args.slices, save_viz)
    
    if success and args.run:
        print()
        run_test()
    elif success:
        print("✅ Configuration updated. Run the test with:")
        print(f"   python test_comprehensive_mask_generator.py")
        print("   OR")
        print(f"   python {__file__} --config {args.config} --run")

if __name__ == "__main__":
    main()
