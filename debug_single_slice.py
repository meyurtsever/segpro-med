#!/usr/bin/env python3
"""
Debug script to test single slice mode
"""

import os
import json
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompts(prompts_file: str) -> dict:
    """Load prompts from JSON file"""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            return json.load(f)
    return {}

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Debug single slice mode')
    parser.add_argument('--prompt_points', type=str, default=None)
    parser.add_argument('--single_slice', action='store_true')
    return parser.parse_args()

def main():
    args = setup_args()
    
    logger.info(f"args.single_slice = {args.single_slice}")
    logger.info(f"args.prompt_points = {args.prompt_points}")
    
    # Load prompts first to determine which slices to load
    prompts_data = {}
    if args.prompt_points:
        logger.info(f"Loading prompts from: {args.prompt_points}")
        prompts_data.update(load_prompts(args.prompt_points))
        logger.info(f"Loaded prompts data: {prompts_data}")
    
    logger.info(f"prompts_data is not empty: {bool(prompts_data)}")
    
    # Determine which slice indices to load for single slice mode
    slice_indices_to_load = None
    if args.single_slice and prompts_data:
        # Get all slice indices that have prompts (convert to 0-based indexing)
        prompt_slice_keys = list(prompts_data.keys())
        logger.info(f"Prompt slice keys: {prompt_slice_keys}")
        if prompt_slice_keys:
            slice_indices_to_load = [int(s) - 1 for s in prompt_slice_keys]  # Convert to 0-based
            logger.info(f"Single slice mode: Will load only slices {[i+1 for i in slice_indices_to_load]} (1-based)")
    else:
        logger.info(f"Single slice mode NOT activated. single_slice={args.single_slice}, prompts_data_exists={bool(prompts_data)}")
    
    logger.info(f"slice_indices_to_load = {slice_indices_to_load}")

if __name__ == "__main__":
    main()
