#!/usr/bin/env python3
"""
Analyze DICOM structure to understand differences between mammography and brain MRI data
"""
import pydicom
import os
import numpy as np

def analyze_dicom_file(file_path):
    """Analyze a single DICOM file and return its structure info"""
    try:
        # Read metadata without pixels first
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        
        info = {
            'filename': os.path.basename(file_path),
            'modality': getattr(ds, 'Modality', 'N/A'),
            'series_description': getattr(ds, 'SeriesDescription', 'N/A'),
            'number_of_frames': getattr(ds, 'NumberOfFrames', None),
            'rows': getattr(ds, 'Rows', 'N/A'),
            'columns': getattr(ds, 'Columns', 'N/A'),
            'photometric_interpretation': getattr(ds, 'PhotometricInterpretation', 'N/A'),
            'bits_allocated': getattr(ds, 'BitsAllocated', 'N/A'),
            'bits_stored': getattr(ds, 'BitsStored', 'N/A'),
        }
        
        # Try to get pixel array dimensions
        try:
            ds_full = pydicom.dcmread(file_path)
            if hasattr(ds_full, 'pixel_array'):
                pixel_shape = ds_full.pixel_array.shape
                info['pixel_shape'] = pixel_shape
                info['pixel_dtype'] = str(ds_full.pixel_array.dtype)
                
                # Check if it's multi-frame
                if len(pixel_shape) > 2:
                    info['is_multiframe'] = True
                    info['num_slices'] = pixel_shape[0] if len(pixel_shape) == 3 else 'Complex'
                else:
                    info['is_multiframe'] = False
                    info['num_slices'] = 1
            else:
                info['pixel_shape'] = 'No pixel data'
                info['is_multiframe'] = False
        except Exception as e:
            info['pixel_shape'] = f'Error loading pixels: {e}'
            info['is_multiframe'] = 'Unknown'
        
        return info
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

def analyze_directory(directory, name):
    """Analyze all DICOM files in a directory"""
    print(f"\n=== {name.upper()} ANALYSIS ===")
    print(f"Directory: {directory}")
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return
    
    files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    
    if not files:
        print("No DICOM files found")
        return
    
    print(f"Found {len(files)} DICOM files")
    
    for file in sorted(files):
        file_path = os.path.join(directory, file)
        info = analyze_dicom_file(file_path)
        
        print(f"\n📁 {info['filename']}")
        
        if 'error' in info:
            print(f"  ❌ Error: {info['error']}")
            continue
        
        print(f"  🏥 Modality: {info['modality']}")
        print(f"  📋 Series: {info['series_description']}")
        print(f"  📐 Dimensions: {info['rows']} x {info['columns']}")
        print(f"  🖼️  Pixel Shape: {info['pixel_shape']}")
        
        if info.get('is_multiframe'):
            print(f"  📚 Multi-frame: YES ({info['num_slices']} slices)")
        elif info.get('is_multiframe') is False:
            print(f"  📚 Multi-frame: NO (single slice)")
        
        if 'number_of_frames' in info and info['number_of_frames']:
            print(f"  🎞️  Number of Frames: {info['number_of_frames']}")
        
        print(f"  💾 Photometric: {info['photometric_interpretation']}")
        print(f"  🔢 Bits: {info['bits_allocated']}/{info['bits_stored']}")

if __name__ == "__main__":
    # Analyze mammography data
    mammo_dir = r"c:\Users\Yurtsever\Downloads\segpro-med\836163459"
    analyze_directory(mammo_dir, "Mammography (836163459)")
    
    # Analyze brain MRI data
    brain_dir = r"c:\Users\Yurtsever\Downloads\segpro-med\cvm_48_t1"
    analyze_directory(brain_dir, "Brain MRI (cvm_48_t1)")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("- Mammography: Likely multi-slice per file")
    print("- Brain MRI: Single slice per file")
    print("- Need to support both patterns in Load Data functionality")