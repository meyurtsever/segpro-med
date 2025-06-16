#!/usr/bin/env python3
"""
Enhanced test for automatic_mask_generator.py with MEDSAM2
This version processes multiple DICOM slices with timing and optional visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time
import torch
import pydicom

# Configuration flags
SAVE_VISUALIZATIONS = True  # Set to True to save mask visualizations (slower)
MAX_SLICES_TO_PROCESS = 20   # Limit number of slices for testing

# Add the current directory and models directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models", "medsam2")

sys.path.insert(0, current_dir)
sys.path.insert(0, models_dir)

# Clear any existing Hydra initialization more aggressively
def clear_hydra_completely():
    """Clear Hydra initialization completely"""
    try:
        # First clear modules
        modules_to_remove = [m for m in sys.modules.keys() if 'hydra' in m.lower()]
        for module in modules_to_remove:
            if module != 'hydra.core.global_hydra':  # Keep this for clearing
                try:
                    del sys.modules[module]
                except:
                    pass
        
        # Then clear GlobalHydra
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            return True
    except Exception:
        pass
    return False

# Clear Hydra at module load time
clear_hydra_completely()

from models.medsam2.build_sam import build_sam2
from models.medsam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def setup_output_directory():
    """Create output directory for test results"""
    output_dir = os.path.join(current_dir, "automatic_mask_test_results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def preprocess_dicom_for_sam(pixel_array):
    """Preprocess DICOM pixel array for SAM input"""
    try:
        # Convert to float32 for processing
        if pixel_array.dtype != np.float32:
            pixel_array = pixel_array.astype(np.float32)
        
        # Normalize to 0-255 range for SAM input
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)
        
        if pixel_max > pixel_min:
            normalized = (pixel_array - pixel_min) / (pixel_max - pixel_min)
            scaled = (normalized * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Convert grayscale to RGB (SAM expects 3-channel input)
        if len(scaled.shape) == 2:
            rgb_image = np.stack([scaled, scaled, scaled], axis=2)
        else:
            rgb_image = scaled
        
        return rgb_image
        
    except Exception as e:
        print(f"Error preprocessing DICOM: {e}")
        raise

def save_masks_visualization(image, masks, output_path, slice_name):
    """Save comprehensive visualization of ALL generated masks"""
    try:
        if not masks:
            print(f"No masks generated for {slice_name}")
            return
        
        num_masks = len(masks)
        print(f"Creating visualization for {num_masks} masks...")
        
        # Sort masks by predicted IoU for better organization
        sorted_masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
          # Calculate grid dimensions to fit all masks
        # We'll have: original image + all masks overlay + individual masks
        total_plots = 2 + num_masks  # original + overlay + all individual masks
        
        # Calculate optimal grid dimensions to ensure ALL masks fit
        if total_plots <= 6:
            cols = 3
            rows = 2
        elif total_plots <= 12:
            cols = 4
            rows = 3
        elif total_plots <= 20:
            cols = 5
            rows = 4
        elif total_plots <= 30:
            cols = 6
            rows = 5
        elif total_plots <= 42:
            cols = 7
            rows = 6
        elif total_plots <= 56:
            cols = 8
            rows = 7
        else:
            # For very large numbers of masks, use square-ish grid
            cols = int(np.ceil(np.sqrt(total_plots)))
            rows = int(np.ceil(total_plots / cols))
            
        # Ensure we have enough slots for all plots
        while (rows * cols) < total_plots:
            if cols <= rows:
                cols += 1
            else:
                rows += 1
        
        # Create figure with appropriate size
        fig_width = cols * 4
        fig_height = rows * 3
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        plot_idx = 1
        
        # Plot 1: Original image
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.title(f'Original Image\n{slice_name}', fontsize=10)
        plt.axis('off')
        plot_idx += 1
        
        # Plot 2: All masks overlay
        plt.subplot(rows, cols, plot_idx)
        display_image = image[:, :, 0].copy()
        all_masks = np.zeros_like(display_image, dtype=bool)
        
        # Create colored overlay with all masks
        colors = plt.cm.Set3(np.linspace(0, 1, min(num_masks, 12)))  # Use different colors
        overlay_image = np.stack([display_image, display_image, display_image], axis=2)
        
        for i, mask_data in enumerate(sorted_masks):
            mask = mask_data['segmentation']
            all_masks |= mask
            
            # Add colored mask to overlay (for first 12 masks to avoid color confusion)
            if i < 12:
                color = colors[i][:3]  # RGB only
                mask_colored = np.zeros_like(overlay_image)
                mask_colored[mask] = [c * 255 for c in color]
                overlay_image = np.where(mask[..., np.newaxis], 
                                       0.7 * overlay_image + 0.3 * mask_colored, 
                                       overlay_image)
        
        plt.imshow(overlay_image.astype(np.uint8))
        plt.title(f'All {num_masks} Masks Overlay\nSorted by IoU', fontsize=10)
        plt.axis('off')
        plot_idx += 1
          # Plot individual masks
        for i, mask_data in enumerate(sorted_masks):
            plt.subplot(rows, cols, plot_idx)
            mask = mask_data['segmentation']
            predicted_iou = mask_data['predicted_iou']
            stability_score = mask_data['stability_score']
            area = mask_data['area']
            
            # Create colored mask visualization
            mask_display = np.stack([display_image, display_image, display_image], axis=2)
            
            # Use different colors for different quality ranges
            if predicted_iou >= 0.8:
                color = [0, 255, 0]  # Green for high quality
            elif predicted_iou >= 0.6:
                color = [255, 255, 0]  # Yellow for medium quality  
            else:
                color = [255, 0, 0]  # Red for low quality
            
            mask_colored = np.zeros_like(mask_display)
            mask_colored[mask] = color
            
            # Blend mask with original image
            alpha = 0.4
            blended = np.where(mask[..., np.newaxis], 
                             (1 - alpha) * mask_display + alpha * mask_colored,
                             mask_display)
            
            plt.imshow(blended.astype(np.uint8))
            plt.title(f'Mask {i+1} (Rank #{i+1})\nIoU: {predicted_iou:.3f}\nStab: {stability_score:.3f}\nArea: {area:,}', 
                     fontsize=8)
            plt.axis('off')
            plot_idx += 1
        
        # Fill remaining subplots if any
        while plot_idx <= rows * cols:
            plt.subplot(rows, cols, plot_idx)
            plt.axis('off')
            plot_idx += 1        
        plt.tight_layout(pad=1.0)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved COMPLETE visualization with ALL {num_masks} masks: {output_path}")
        print(f"  📊 Grid: {rows}x{cols} ({rows*cols} slots for {total_plots} plots)")
        print(f"  🖼️  Image size: {fig_width}x{fig_height} pixels")
        print(f"  📈 Includes: Original + Overlay + {num_masks} individual masks")
        
    except Exception as e:
        print(f"Error saving visualization: {e}")
        import traceback
        traceback.print_exc()

def save_mask_statistics_plot(masks, output_path, slice_name):
    """Save a separate plot with mask statistics"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract statistics
        ious = [m['predicted_iou'] for m in masks]
        stabilities = [m['stability_score'] for m in masks]
        areas = [m['area'] for m in masks]
        
        # IoU distribution
        axes[0, 0].hist(ious, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('IoU Distribution')
        axes[0, 0].set_xlabel('Predicted IoU')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Stability distribution
        axes[0, 1].hist(stabilities, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Stability Score Distribution')
        axes[0, 1].set_xlabel('Stability Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Area distribution (log scale)
        axes[1, 0].hist(areas, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Mask Area Distribution')
        axes[1, 0].set_xlabel('Area (pixels)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # IoU vs Stability scatter
        scatter = axes[1, 1].scatter(ious, stabilities, c=areas, cmap='viridis', alpha=0.6)
        axes[1, 1].set_title('IoU vs Stability (colored by area)')
        axes[1, 1].set_xlabel('Predicted IoU')
        axes[1, 1].set_ylabel('Stability Score')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Area (pixels)')
        
        plt.suptitle(f'Mask Statistics Summary - {slice_name}\nTotal Masks: {len(masks)}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved mask statistics plot: {output_path}")
        
    except Exception as e:
        print(f"Error saving statistics plot: {e}")

def process_dicom_slice(mask_generator, dicom_path, output_dir, slice_idx, save_viz=False):
    """Process a single DICOM slice and generate masks with detailed timing"""
    timing_info = {}
    
    try:
        print(f"\n{'='*50}")
        print(f"Processing slice {slice_idx}: {os.path.basename(dicom_path)}")
        print(f"{'='*50}")
        
        # Load DICOM - TIMED
        start_time = time.time()
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array
        timing_info['load_dicom'] = time.time() - start_time
        print(f"✓ DICOM loaded in {timing_info['load_dicom']:.3f}s - Shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")
        
        # Preprocess for SAM - TIMED
        start_time = time.time()
        rgb_image = preprocess_dicom_for_sam(pixel_array)
        timing_info['preprocess'] = time.time() - start_time
        print(f"✓ Preprocessing completed in {timing_info['preprocess']:.3f}s - Output shape: {rgb_image.shape}")
        
        # Generate masks - TIMED (main operation)
        print(f"🔄 Starting mask generation...")
        start_time = time.time()
        masks = mask_generator.generate(rgb_image)
        timing_info['mask_generation'] = time.time() - start_time
        print(f"✅ MASK GENERATION completed in {timing_info['mask_generation']:.3f}s")
        print(f"   Generated {len(masks)} masks")
        
        # Calculate quality metrics
        if masks:
            avg_iou = np.mean([mask['predicted_iou'] for mask in masks])
            avg_stability = np.mean([mask['stability_score'] for mask in masks])
            max_iou = max([mask['predicted_iou'] for mask in masks])
            print(f"   Quality: Avg IoU={avg_iou:.3f}, Max IoU={max_iou:.3f}, Avg Stability={avg_stability:.3f}")
        
        # Sort masks by predicted IoU for better analysis
        masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
        
        # Save JSON data - TIMED
        start_time = time.time()
        slice_name = f"slice_{slice_idx:03d}"
        mask_data_path = os.path.join(output_dir, f"{slice_name}_masks.json")
        
        masks_serializable = []
        for i, mask in enumerate(masks):
            mask_info = {
                'mask_id': i,
                'area': int(mask['area']),
                'bbox': [float(x) for x in mask['bbox']],
                'predicted_iou': float(mask['predicted_iou']),
                'stability_score': float(mask['stability_score']),
                'point_coords': [[float(x), float(y)] for x, y in mask['point_coords']]
            }
            masks_serializable.append(mask_info)
        
        result_data = {
            'slice_name': slice_name,
            'dicom_file': os.path.basename(dicom_path),
            'num_masks': len(masks),
            'timing_info': timing_info,
            'quality_metrics': {
                'avg_iou': avg_iou if masks else 0,
                'avg_stability': avg_stability if masks else 0,
                'max_iou': max_iou if masks else 0
            },
            'masks': masks_serializable,
            'image_shape': pixel_array.shape,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(mask_data_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        timing_info['save_json'] = time.time() - start_time
        print(f"✓ JSON data saved in {timing_info['save_json']:.3f}s")
        
        # Save visualization (optional) - TIMED
        if save_viz:
            start_time = time.time()
            viz_path = os.path.join(output_dir, f"{slice_name}_comprehensive_viz.png")
            save_masks_visualization(rgb_image, masks, viz_path, slice_name)
            timing_info['save_visualization'] = time.time() - start_time
            print(f"✓ Visualization saved in {timing_info['save_visualization']:.3f}s")
        else:
            timing_info['save_visualization'] = 0
            print("⏭️  Skipping visualization (save_viz=False)")
        
        # Calculate total time
        timing_info['total'] = sum(timing_info.values())
        
        print(f"📊 SLICE {slice_idx} TIMING BREAKDOWN:")
        print(f"   Load DICOM:        {timing_info['load_dicom']:.3f}s")
        print(f"   Preprocess:        {timing_info['preprocess']:.3f}s")
        print(f"   🎯 Mask Generation: {timing_info['mask_generation']:.3f}s ⭐")
        print(f"   Save JSON:         {timing_info['save_json']:.3f}s")
        print(f"   Save Visualization: {timing_info['save_visualization']:.3f}s")
        print(f"   ⏱️  TOTAL TIME:      {timing_info['total']:.3f}s")
        
        return {
            'slice_idx': slice_idx,
            'dicom_path': dicom_path,
            'num_masks': len(masks),
            'timing_info': timing_info,
            'quality_metrics': result_data['quality_metrics'],
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error processing DICOM slice {slice_idx}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'slice_idx': slice_idx,
            'dicom_path': dicom_path,
            'error': str(e),
            'success': False
        }
        
        return {
            'slice_idx': slice_idx,
            'dicom_path': dicom_path,
            'num_masks': len(masks),
            'masks': masks,
            'image_shape': pixel_array.shape
        }
        
    except Exception as e:
        print(f"Error processing DICOM slice {slice_idx}: {e}")
        return None

def main():
    # Start overall timer
    overall_start_time = time.time()
    
    print("=" * 80)
    print("MEDSAM2 Enhanced Automatic Mask Generator Test")
    print(f"Configuration: Save Visualizations = {SAVE_VISUALIZATIONS}")
    print(f"Configuration: Max Slices = {MAX_SLICES_TO_PROCESS}")
    print("=" * 80)
    
    try:        # Clear any existing Hydra initialization before loading model
        setup_start_time = time.time()
        
        # Aggressive Hydra clearing
        def force_clear_hydra():
            """Aggressively clear Hydra state"""
            try:
                # Clear from sys.modules if loaded
                modules_to_remove = [m for m in sys.modules.keys() if 'hydra' in m.lower()]
                for module in modules_to_remove:
                    if module != 'hydra.core.global_hydra':  # Keep this one for clearing
                        try:
                            del sys.modules[module]
                        except:
                            pass
                
                # Clear GlobalHydra instance
                from hydra.core.global_hydra import GlobalHydra
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                    print("✓ Cleared existing Hydra initialization")
                return True
            except Exception as e:
                print(f"Note: Hydra clearing attempt: {e}")
                return False
        
        # Try multiple times if needed
        for attempt in range(3):
            if force_clear_hydra():
                break
            if attempt < 2:
                print(f"Hydra clear attempt {attempt + 1} failed, retrying...")
                time.sleep(0.1)
        
        # Setup output directory
        output_dir = setup_output_directory()
        print(f"Output directory: {output_dir}")
        
        # Define paths
        config_path = os.path.join(models_dir, "configs", "sam2.1_hiera_b+.yaml")
        checkpoint_path = os.path.join(models_dir, "checkpoints", "sam2.1_hiera_base_plus.pt")
        setup_time = time.time() - setup_start_time
        
        # Check CUDA availability and test basic operations
        cuda_test_start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            try:
                # Test basic CUDA operations first
                print("Testing CUDA compatibility...")
                test_tensor = torch.randn(10, 10).cuda()
                result = torch.mm(test_tensor, test_tensor.t())
                print("✓ Basic CUDA test passed")
                
                # Test convolution operations that SAM2 uses heavily
                conv_test = torch.nn.Conv2d(3, 16, 3).cuda()
                input_test = torch.randn(1, 3, 32, 32).cuda()
                conv_result = conv_test(input_test)
                print("✓ Convolution CUDA test passed")
                
            except Exception as e:
                print(f"⚠ CUDA compatibility test failed: {e}")
                print("Attempting to continue, but SAM2 may fail...")
        
        cuda_test_time = time.time() - cuda_test_start_time
        print(f"Using device: {device}")
        
        # Load SAM2 model with enhanced error handling
        print("\n🔄 Loading SAM2 model...")
        model_load_start_time = time.time()
        try:
            sam2_model = build_sam2(config_path, checkpoint_path, device=device)
            model_load_time = time.time() - model_load_start_time
            print(f"✅ SAM2 model loaded successfully in {model_load_time:.2f}s!")
        except RuntimeError as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
        # Create mask generator with optimized parameters
        print("\n🔄 Creating automatic mask generator...")
        generator_start_time = time.time()
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=24,          # Balance between quality and speed
            points_per_batch=48,         # Reasonable batch size
            pred_iou_thresh=0.75,        # Higher threshold for better quality
            stability_score_thresh=0.85, # Good stability threshold
            min_mask_region_area=100,    # Filter out very small regions
            output_mode="binary_mask",
            use_m2m=False,               # Disable for initial test
            multimask_output=True
        )
        generator_time = time.time() - generator_start_time
        print(f"✅ Mask generator created in {generator_time:.3f}s!")
        
        # Find DICOM files
        print("\n🔍 Finding DICOM files...")
        dicom_dir = os.path.join(current_dir, "cvm_48_t1")
        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        
        if not dicom_files:
            print("❌ No DICOM files found!")
            return
        
        dicom_files.sort()
        print(f"✅ Found {len(dicom_files)} DICOM files")
        
        # Process multiple slices
        max_slices = min(MAX_SLICES_TO_PROCESS, len(dicom_files))
        print(f"\n🚀 Processing first {max_slices} slices...")
        print(f"📊 Visualizations will be {'SAVED' if SAVE_VISUALIZATIONS else 'SKIPPED'}")
        
        # Start processing timer
        processing_start_time = time.time()
        results = []
        
        for i in range(max_slices):
            dicom_path = os.path.join(dicom_dir, dicom_files[i])
            result = process_dicom_slice(mask_generator, dicom_path, output_dir, i, SAVE_VISUALIZATIONS)
            if result and result.get('success', False):
                results.append(result)
        
        processing_time = time.time() - processing_start_time
        overall_time = time.time() - overall_start_time        
        processing_time = time.time() - processing_start_time
        overall_time = time.time() - overall_start_time
        
        # Calculate detailed timing statistics
        if results:
            mask_generation_times = []
            total_times = []
            for r in results:
                if 'timing_info' in r:
                    mask_generation_times.append(r['timing_info']['mask_generation'])
                    total_times.append(r['timing_info']['total'])
            
            avg_mask_time = np.mean(mask_generation_times) if mask_generation_times else 0
            total_mask_time = sum(mask_generation_times) if mask_generation_times else 0
        
        # Create comprehensive summary report
        summary_path = os.path.join(output_dir, "comprehensive_test_summary.json")
        
        # Calculate statistics
        total_masks = sum(r['num_masks'] for r in results if r.get('success', False))
        avg_masks = total_masks / len(results) if results else 0
        
        all_ious = []
        all_stability = []
        all_areas = []
        
        for r in results:
            if r.get('success', False) and 'quality_metrics' in r:
                if r['quality_metrics']['avg_iou'] > 0:
                    all_ious.append(r['quality_metrics']['avg_iou'])
                if r['quality_metrics']['avg_stability'] > 0:
                    all_stability.append(r['quality_metrics']['avg_stability'])
        
        # Enhanced summary with timing information
        summary = {
            'test_info': {
                'test_date': datetime.now().isoformat(),
                'model_config': 'sam2.1_hiera_b+.yaml',
                'model_checkpoint': 'sam2.1_hiera_base_plus.pt',
                'device': device,
                'total_dicom_files': len(dicom_files),
                'processed_slices': len(results),
                'save_visualizations': SAVE_VISUALIZATIONS
            },
            'timing_summary': {
                'setup_time': setup_time,
                'cuda_test_time': cuda_test_time,
                'model_load_time': model_load_time,
                'generator_creation_time': generator_time,
                'total_processing_time': processing_time,
                'overall_time': overall_time,
                'avg_mask_generation_time_per_slice': avg_mask_time,
                'total_mask_generation_time': total_mask_time,
                'masks_per_second': total_masks / total_mask_time if total_mask_time > 0 else 0
            },
            'mask_statistics': {
                'total_masks': total_masks,
                'avg_masks_per_slice': avg_masks,
                'avg_iou': np.mean(all_ious) if all_ious else 0,
                'avg_stability': np.mean(all_stability) if all_stability else 0,
                'iou_range': [min(all_ious), max(all_ious)] if all_ious else [0, 0],
                'stability_range': [min(all_stability), max(all_stability)] if all_stability else [0, 0]
            },
            'detailed_results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also create a readable text summary with timing
        text_summary_path = os.path.join(output_dir, "comprehensive_test_summary.txt")
        with open(text_summary_path, 'w') as f:
            f.write("MEDSAM2 Automatic Mask Generator - Comprehensive Test Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Model: sam2.1_hiera_base_plus.pt\n")
            f.write(f"Config: sam2.1_hiera_b+.yaml\n")
            f.write(f"Save Visualizations: {SAVE_VISUALIZATIONS}\n\n")
            
            f.write("TIMING BREAKDOWN:\n")
            f.write(f"Setup & Initialization:    {setup_time:.3f}s\n")
            f.write(f"CUDA Compatibility Test:   {cuda_test_time:.3f}s\n")
            f.write(f"Model Loading:             {model_load_time:.2f}s\n")
            f.write(f"Generator Creation:        {generator_time:.3f}s\n")
            f.write(f"MASK GENERATION TOTAL:   {total_mask_time:.2f}s\n")
            f.write(f"Other Processing:          {processing_time - total_mask_time:.2f}s\n")
            f.write(f"OVERALL TOTAL TIME:      {overall_time:.2f}s\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Average mask generation per slice: {avg_mask_time:.3f}s\n")
            f.write(f"Masks generated per second: {total_masks / total_mask_time if total_mask_time > 0 else 0:.1f}\n")
            f.write(f"Total slices per minute: {60 / avg_mask_time if avg_mask_time > 0 else 0:.1f}\n\n")
            
            f.write("PROCESSING SUMMARY:\n")
            f.write(f"Total DICOM files available: {len(dicom_files)}\n")
            f.write(f"Slices processed: {len(results)}\n")
            f.write(f"Total masks generated: {total_masks}\n")
            f.write(f"Average masks per slice: {avg_masks:.2f}\n\n")
            
            f.write("MASK QUALITY STATISTICS:\n")
            if all_ious:
                f.write(f"IoU - Min: {min(all_ious):.3f}, Max: {max(all_ious):.3f}, Mean: {np.mean(all_ious):.3f}\n")
                f.write(f"Stability - Min: {min(all_stability):.3f}, Max: {max(all_stability):.3f}, Mean: {np.mean(all_stability):.3f}\n\n")
            
            f.write("PER-SLICE RESULTS:\n")
            for r in results:
                if r.get('success', False):
                    timing = r.get('timing_info', {})
                    mask_time = timing.get('mask_generation', 0)
                    total_time = timing.get('total', 0)
                    f.write(f"Slice {r['slice_idx']:02d}: {os.path.basename(r['dicom_path'])} -> "
                           f"{r['num_masks']} masks (Gen: {mask_time:.2f}s, Total: {total_time:.2f}s)\n")
        
        print("\n" + "🎉" * 27)
        print("🎉 COMPREHENSIVE TEST COMPLETED! 🎉")
        print("🎉" * 27)
        
        print(f"\n⏱️  TIMING SUMMARY:")
        print("=" * 50)
        print(f"Overall time:              {overall_time:.2f}s")
        print(f"Model loading:             {model_load_time:.2f}s")
        print(f"🎯 Total mask generation:   {total_mask_time:.2f}s ⭐")
        print(f"Average per slice:         {avg_mask_time:.3f}s")
        
        print(f"\n📊 PERFORMANCE:")
        print("=" * 50)
        print(f"Processed {len(results)} slices")
        print(f"Generated {total_masks} masks total")
        print(f"Average {avg_masks:.1f} masks per slice")
        print(f"🚀 Speed: {total_masks / total_mask_time if total_mask_time > 0 else 0:.1f} masks/second")
        
        if all_ious:
            print(f"\n🎯 QUALITY:")
            print("=" * 50)
            print(f"IoU range: {min(all_ious):.3f} - {max(all_ious):.3f} (avg: {np.mean(all_ious):.3f})")
            print(f"Stability range: {min(all_stability):.3f} - {max(all_stability):.3f} (avg: {np.mean(all_stability):.3f})")
        
        # Also create a readable text summary  
        text_summary_path = os.path.join(output_dir, "comprehensive_test_summary.txt")
        with open(text_summary_path, 'w') as f:
            f.write("MEDSAM2 Automatic Mask Generator - Comprehensive Test Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Model: sam2.1_hiera_base_plus.pt\n")
            f.write(f"Config: sam2.1_hiera_b+.yaml\n\n")
            
            f.write("PROCESSING SUMMARY:\n")
            f.write(f"Total DICOM files available: {len(dicom_files)}\n")
            f.write(f"Slices processed: {len(results)}\n")
            f.write(f"Total masks generated: {total_masks}\n")
            f.write(f"Average masks per slice: {avg_masks:.2f}\n\n")
            f.write("MASK QUALITY STATISTICS:\n")
            if all_ious:
                f.write(f"IoU - Min: {min(all_ious):.3f}, Max: {max(all_ious):.3f}, Mean: {np.mean(all_ious):.3f}\n")
                f.write(f"Stability - Min: {min(all_stability):.3f}, Max: {max(all_stability):.3f}, Mean: {np.mean(all_stability):.3f}\n")
                f.write(f"Area - Min: {min(all_areas):,}, Max: {max(all_areas):,}, Mean: {int(np.mean(all_areas)):,}\n\n")
            
            f.write("PER-SLICE RESULTS:\n")
            for r in results:
                f.write(f"Slice {r['slice_idx']:02d}: {os.path.basename(r['dicom_path'])} -> {r['num_masks']} masks\n")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📄 JSON Summary: {summary_path}")
        print(f"📄 Text Summary: {text_summary_path}")
        
        if not SAVE_VISUALIZATIONS:
            print(f"\n💡 Tip: Set SAVE_VISUALIZATIONS = True at the top of the script to save ALL mask visualizations")
            print(f"       📊 Enhanced features: Shows ALL masks (not just top 6), quality-colored overlays,")
            print(f"       📈 and statistical analysis plots for comprehensive mask analysis!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
