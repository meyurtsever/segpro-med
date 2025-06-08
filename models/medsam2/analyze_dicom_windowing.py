#!/usr/bin/env python3
"""
Analyze DICOM Windowing Metadata
This script analyzes the windowing parameters extracted from DICOM files.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import glob
from pathlib import Path

def analyze_windowing_metadata(results_dir):
    """Analyze windowing metadata from inference results"""
    
    windowing_file = os.path.join(results_dir, 'windowing_metadata.json')
    summary_file = os.path.join(results_dir, 'summary.json')
    
    if not os.path.exists(windowing_file):
        print(f"Windowing metadata file not found: {windowing_file}")
        return
    
    # Load windowing metadata
    with open(windowing_file, 'r') as f:
        windowing_data = json.load(f)
    
    # Load summary
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
    else:
        summary_data = {}
    
    print(f"Analyzing windowing metadata for {len(windowing_data)} slices")
    print("=" * 60)
    
    # Extract windowing parameters
    window_centers = []
    window_widths = []
    rescale_slopes = []
    rescale_intercepts = []
    slice_indices = []
    
    for slice_idx_str, metadata in windowing_data.items():
        slice_idx = int(slice_idx_str)
        slice_indices.append(slice_idx)
        window_centers.append(metadata['window_center'])
        window_widths.append(metadata['window_width'])
        rescale_slopes.append(metadata['rescale_slope'])
        rescale_intercepts.append(metadata['rescale_intercept'])
    
    # Convert to numpy arrays
    slice_indices = np.array(slice_indices)
    window_centers = np.array(window_centers)
    window_widths = np.array(window_widths)
    rescale_slopes = np.array(rescale_slopes)
    rescale_intercepts = np.array(rescale_intercepts)
    
    # Print statistics
    print("Windowing Parameters Statistics:")
    print(f"Window Center - Mean: {np.mean(window_centers):.2f}, Std: {np.std(window_centers):.2f}, Range: [{np.min(window_centers):.2f}, {np.max(window_centers):.2f}]")
    print(f"Window Width  - Mean: {np.mean(window_widths):.2f}, Std: {np.std(window_widths):.2f}, Range: [{np.min(window_widths):.2f}, {np.max(window_widths):.2f}]")
    print(f"Rescale Slope - Mean: {np.mean(rescale_slopes):.4f}, Std: {np.std(rescale_slopes):.4f}, Range: [{np.min(rescale_slopes):.4f}, {np.max(rescale_slopes):.4f}]")
    print(f"Rescale Intercept - Mean: {np.mean(rescale_intercepts):.2f}, Std: {np.std(rescale_intercepts):.2f}, Range: [{np.min(rescale_intercepts):.2f}, {np.max(rescale_intercepts):.2f}]")
    print()
    
    # Check for variations
    center_variation = np.std(window_centers) > 1
    width_variation = np.std(window_widths) > 1
    
    if center_variation or width_variation:
        print("⚠️  Significant windowing variations detected across slices!")
        print("   This demonstrates why using per-slice DICOM metadata is important.")
    else:
        print("✓ Windowing parameters are consistent across slices.")
    print()
    
    # Show per-slice details
    print("Per-slice windowing details:")
    print("Slice | Center | Width | Slope  | Intercept | File")
    print("-" * 60)
    for i, (slice_idx, metadata) in enumerate(sorted(windowing_data.items(), key=lambda x: int(x[0]))):
        print(f"{slice_idx:5} | {metadata['window_center']:6.1f} | {metadata['window_width']:5.1f} | {metadata['rescale_slope']:6.3f} | {metadata['rescale_intercept']:9.2f} | {metadata['file_name']}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Window Center plot
    plt.subplot(2, 3, 1)
    plt.plot(slice_indices, window_centers, 'bo-', markersize=4)
    plt.title('Window Center vs Slice')
    plt.xlabel('Slice Index')
    plt.ylabel('Window Center')
    plt.grid(True, alpha=0.3)
    
    # Window Width plot
    plt.subplot(2, 3, 2)
    plt.plot(slice_indices, window_widths, 'ro-', markersize=4)
    plt.title('Window Width vs Slice')
    plt.xlabel('Slice Index')
    plt.ylabel('Window Width')
    plt.grid(True, alpha=0.3)
    
    # Rescale Slope plot
    plt.subplot(2, 3, 3)
    plt.plot(slice_indices, rescale_slopes, 'go-', markersize=4)
    plt.title('Rescale Slope vs Slice')
    plt.xlabel('Slice Index')
    plt.ylabel('Rescale Slope')
    plt.grid(True, alpha=0.3)
    
    # Rescale Intercept plot
    plt.subplot(2, 3, 4)
    plt.plot(slice_indices, rescale_intercepts, 'mo-', markersize=4)
    plt.title('Rescale Intercept vs Slice')
    plt.xlabel('Slice Index')
    plt.ylabel('Rescale Intercept')
    plt.grid(True, alpha=0.3)
    
    # Window Center distribution
    plt.subplot(2, 3, 5)
    plt.hist(window_centers, bins=20, alpha=0.7, color='blue')
    plt.title('Window Center Distribution')
    plt.xlabel('Window Center')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Window Width distribution
    plt.subplot(2, 3, 6)
    plt.hist(window_widths, bins=20, alpha=0.7, color='red')
    plt.title('Window Width Distribution')
    plt.xlabel('Window Width')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('DICOM Windowing Metadata Analysis')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'windowing_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nWindowing analysis plot saved to: {plot_path}")
    plt.show()
    
    return windowing_data

def compare_dicom_windowing_directly(dicom_folder):
    """Directly analyze DICOM files to show windowing variations"""
    
    dicom_files = sorted(glob.glob(os.path.join(dicom_folder, "*.dcm")))
    
    if not dicom_files:
        print(f"No DICOM files found in {dicom_folder}")
        return
    
    print(f"Direct DICOM Analysis: {len(dicom_files)} files")
    print("=" * 60)
    
    windowing_info = []
    
    for i, file_path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(file_path)
            
            # Extract windowing parameters
            window_center = getattr(ds, 'WindowCenter', 'N/A')
            window_width = getattr(ds, 'WindowWidth', 'N/A')
            rescale_slope = getattr(ds, 'RescaleSlope', 'N/A')
            rescale_intercept = getattr(ds, 'RescaleIntercept', 'N/A')
            
            # Handle multiple values
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]
            
            windowing_info.append({
                'file': os.path.basename(file_path),
                'slice_idx': i,
                'window_center': window_center,
                'window_width': window_width,
                'rescale_slope': rescale_slope,
                'rescale_intercept': rescale_intercept
            })
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Print results
    print("File | Center | Width | Slope | Intercept")
    print("-" * 50)
    for info in windowing_info[:10]:  # Show first 10
        print(f"{info['file'][:20]:20} | {info['window_center']:6} | {info['window_width']:5} | {info['rescale_slope']:5} | {info['rescale_intercept']:9}")
    
    if len(windowing_info) > 10:
        print(f"... and {len(windowing_info) - 10} more files")
    
    # Check for variations
    centers = [info['window_center'] for info in windowing_info if isinstance(info['window_center'], (int, float))]
    widths = [info['window_width'] for info in windowing_info if isinstance(info['window_width'], (int, float))]
    
    if centers and len(set(centers)) > 1:
        print(f"\n⚠️  Window Center varies: {set(centers)}")
    if widths and len(set(widths)) > 1:
        print(f"⚠️  Window Width varies: {set(widths)}")
    
    return windowing_info

def main():
    print("DICOM Windowing Metadata Analysis")
    print("=================================")
    
    # Analyze results if available
    results_dir = "brain_mri_metadata_test"
    if os.path.exists(results_dir):
        print(f"\n1. Analyzing results from: {results_dir}")
        analyze_windowing_metadata(results_dir)
    
    # Analyze DICOM files directly
    dicom_folder = "cvm48t1"
    if os.path.exists(dicom_folder):
        print(f"\n2. Direct DICOM analysis from: {dicom_folder}")
        compare_dicom_windowing_directly(dicom_folder)
    
    print("\n" + "="*60)
    print("Key Benefits of DICOM Metadata Windowing:")
    print("• Each slice uses its optimal window/level settings")
    print("• Preserves the original DICOM viewing parameters")
    print("• Ensures consistent image quality across the volume")
    print("• Maintains radiologist-intended contrast settings")
    print("• Improves segmentation accuracy by using proper preprocessing")

if __name__ == "__main__":
    main()