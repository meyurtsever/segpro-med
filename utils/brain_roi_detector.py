"""
Brain ROI Detector for Automated Annotation Pipeline

This module provides automatic detection of brain regions and eyes from medical images
to generate bounding box prompts for MEDSAM2 segmentation.

Uses anatomical priors, DICOM metadata, and image processing techniques to identify
likely regions of interest in brain MRI/CT scans.
"""

import numpy as np
import cv2
import pydicom
from typing import List, Tuple, Dict, Optional, Union
import logging
import os
from pathlib import Path
import json
from scipy import ndimage
from skimage import measure, morphology, filters

logger = logging.getLogger(__name__)


class BrainROIDetector:
    """Automatic detector for brain regions and eye structures in medical images"""
    
    def __init__(self):
        """Initialize the brain ROI detector"""
        self.brain_tissue_threshold_percentile = 30  # Percentile for brain tissue detection
        self.eye_detection_enabled = True
        self.min_brain_area = 1000  # Minimum area for brain region
        self.min_eye_area = 100     # Minimum area for eye region
        
    def extract_anatomical_info(self, dicom_dataset) -> Dict:
        """Extract anatomical information from DICOM metadata"""
        anatomical_info = {
            'image_orientation': 'unknown',
            'slice_thickness': 1.0,
            'pixel_spacing': [1.0, 1.0],
            'image_position': [0.0, 0.0, 0.0],
            'body_part': 'unknown',
            'view_position': 'unknown'
        }
        
        try:
            # Get image orientation
            if hasattr(dicom_dataset, 'ImageOrientationPatient'):
                orientation = dicom_dataset.ImageOrientationPatient
                # Determine if this is axial, sagittal, or coronal
                if abs(orientation[2]) > 0.8 or abs(orientation[5]) > 0.8:
                    anatomical_info['image_orientation'] = 'axial'
                elif abs(orientation[0]) > 0.8 or abs(orientation[3]) > 0.8:
                    anatomical_info['image_orientation'] = 'sagittal'
                else:
                    anatomical_info['image_orientation'] = 'coronal'
            
            # Get slice thickness
            if hasattr(dicom_dataset, 'SliceThickness'):
                anatomical_info['slice_thickness'] = float(dicom_dataset.SliceThickness)
            
            # Get pixel spacing
            if hasattr(dicom_dataset, 'PixelSpacing'):
                anatomical_info['pixel_spacing'] = [float(x) for x in dicom_dataset.PixelSpacing]
            
            # Get image position
            if hasattr(dicom_dataset, 'ImagePositionPatient'):
                anatomical_info['image_position'] = [float(x) for x in dicom_dataset.ImagePositionPatient]
            
            # Get body part examined
            if hasattr(dicom_dataset, 'BodyPartExamined'):
                anatomical_info['body_part'] = str(dicom_dataset.BodyPartExamined).lower()
            
            # Get view position
            if hasattr(dicom_dataset, 'ViewPosition'):
                anatomical_info['view_position'] = str(dicom_dataset.ViewPosition)
                
        except Exception as e:
            logger.warning(f"Error extracting anatomical info: {e}")
        
        return anatomical_info
    
    def detect_brain_bounds(self, image: np.ndarray, anatomical_info: Dict) -> List[Tuple[int, int, int, int]]:
        """
        Detect brain tissue bounds using image processing techniques
        
        Args:
            image: Grayscale image (H, W) or RGB image (H, W, 3)
            anatomical_info: Dictionary with anatomical metadata
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        try:
            # Convert to grayscale if needed 
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            h, w = gray.shape
            bounding_boxes = []
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding to find brain tissue
            # Brain tissue typically has moderate intensity values
            threshold_value = np.percentile(blurred[blurred > 0], self.brain_tissue_threshold_percentile)
            
            # Create binary mask for potential brain tissue
            brain_mask = blurred > threshold_value
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            brain_mask = cv2.morphologyEx(brain_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            labeled_image = measure.label(brain_mask)
            regions = measure.regionprops(labeled_image)
            
            # Sort regions by area (largest first)
            regions = sorted(regions, key=lambda r: r.area, reverse=True)
            
            # Generate bounding boxes based on image orientation
            if anatomical_info.get('image_orientation') == 'axial':
                bounding_boxes.extend(self._generate_axial_brain_boxes(regions, h, w))
            elif anatomical_info.get('image_orientation') == 'sagittal':
                bounding_boxes.extend(self._generate_sagittal_brain_boxes(regions, h, w))
            elif anatomical_info.get('image_orientation') == 'coronal':
                bounding_boxes.extend(self._generate_coronal_brain_boxes(regions, h, w))
            else:
                # Default: treat as axial
                bounding_boxes.extend(self._generate_axial_brain_boxes(regions, h, w))
            
            # If no brain regions detected, generate default center box
            if not bounding_boxes:
                center_box = self._generate_default_brain_box(h, w)
                if center_box:
                    bounding_boxes.append(center_box)
            
            logger.info(f"Detected {len(bounding_boxes)} brain bounding boxes")
            return bounding_boxes
            
        except Exception as e:
            logger.error(f"Error in brain bounds detection: {e}")
            # Return default center box as fallback
            return [self._generate_default_brain_box(h, w)] if 'h' in locals() and 'w' in locals() else []
    
    def detect_eye_bounds(self, image: np.ndarray, anatomical_info: Dict) -> List[Tuple[int, int, int, int]]:
        """
        Detect eye regions using anatomical priors and image processing
        
        Args:
            image: Grayscale or RGB image
            anatomical_info: Dictionary with anatomical metadata
            
        Returns:
            List of eye bounding boxes as (x1, y1, x2, y2) tuples
        """
        if not self.eye_detection_enabled:
            return []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            h, w = gray.shape
            eye_boxes = []
            
            # Eyes are typically located in the upper portion of axial brain images
            # and appear as darker (low intensity) circular/oval regions
            if anatomical_info.get('image_orientation') == 'axial':
                eye_boxes.extend(self._detect_axial_eyes(gray, h, w))
            elif anatomical_info.get('image_orientation') == 'coronal':
                eye_boxes.extend(self._detect_coronal_eyes(gray, h, w))
            # Sagittal view typically shows only one eye, so we skip it for bilateral detection
            
            logger.info(f"Detected {len(eye_boxes)} eye bounding boxes")
            return eye_boxes
            
        except Exception as e:
            logger.error(f"Error in eye detection: {e}")
            return []
    
    def _generate_axial_brain_boxes(self, regions: List, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes for axial brain images"""
        boxes = []
        
        for region in regions[:3]:  # Take up to 3 largest regions
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Add some padding around the detected region
            padding_h = int(h * 0.05)  # 5% padding
            padding_w = int(w * 0.05)
            
            x1 = max(0, minc - padding_w)
            y1 = max(0, minr - padding_h)
            x2 = min(w, maxc + padding_w)
            y2 = min(h, maxr + padding_h)
            
            boxes.append((x1, y1, x2, y2))
        
        # If we have a large central region, also add smaller focused boxes
        if boxes and regions:
            main_region = regions[0]
            if main_region.area > h * w * 0.3:  # If main region is >30% of image
                # Add focused boxes for specific brain structures
                minr, minc, maxr, maxc = main_region.bbox
                center_x = (minc + maxc) // 2
                center_y = (minr + maxr) // 2
                
                # Add smaller boxes for specific regions
                box_size_x = (maxc - minc) // 4
                box_size_y = (maxr - minr) // 4
                
                # Left hemisphere
                left_box = (max(0, center_x - box_size_x), 
                           max(0, center_y - box_size_y//2),
                           min(w, center_x), 
                           min(h, center_y + box_size_y//2))
                boxes.append(left_box)
                
                # Right hemisphere  
                right_box = (max(0, center_x), 
                            max(0, center_y - box_size_y//2),
                            min(w, center_x + box_size_x), 
                            min(h, center_y + box_size_y//2))
                boxes.append(right_box)
        
        return boxes
    
    def _generate_sagittal_brain_boxes(self, regions: List, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes for sagittal brain images"""
        boxes = []
        
        for region in regions[:2]:  # Take up to 2 largest regions
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Add padding
            padding_h = int(h * 0.08)
            padding_w = int(w * 0.08)
            
            x1 = max(0, minc - padding_w)
            y1 = max(0, minr - padding_h)
            x2 = min(w, maxc + padding_w)
            y2 = min(h, maxr + padding_h)
            
            boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _generate_coronal_brain_boxes(self, regions: List, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes for coronal brain images"""
        boxes = []
        
        for region in regions[:2]:  # Take up to 2 largest regions
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Add padding
            padding_h = int(h * 0.08)
            padding_w = int(w * 0.08)
            
            x1 = max(0, minc - padding_w)
            y1 = max(0, minr - padding_h)
            x2 = min(w, maxc + padding_w)
            y2 = min(h, maxr + padding_h)
            
            boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _generate_default_brain_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Generate a default brain bounding box when detection fails"""
        # Center box covering ~60% of the image
        margin_h = int(h * 0.2)
        margin_w = int(w * 0.2)
        
        return (margin_w, margin_h, w - margin_w, h - margin_h)
    
    def _detect_axial_eyes(self, gray: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect eyes in axial brain images"""
        eyes = []
        
        # Eyes are typically in the upper 1/3 of axial brain images
        upper_region = gray[:h//3, :]
        
        # Apply threshold to find dark regions (eyes appear dark)
        threshold = np.percentile(upper_region[upper_region > 0], 20)  # Bottom 20% intensity
        eye_mask = upper_region < threshold
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eye_mask = cv2.morphologyEx(eye_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        labeled = measure.label(eye_mask)
        regions = measure.regionprops(labeled)
        
        # Filter regions by size and shape for eye candidates
        eye_candidates = []
        for region in regions:
            if (self.min_eye_area < region.area < self.min_eye_area * 20 and  # Size constraint
                0.3 < region.eccentricity < 0.9):  # Shape constraint (somewhat oval)
                eye_candidates.append(region)
        
        # Sort by area and take the two largest (left and right eyes)
        eye_candidates = sorted(eye_candidates, key=lambda r: r.area, reverse=True)[:2]
        
        for region in eye_candidates:
            minr, minc, maxr, maxc = region.bbox
            
            # Add small padding around detected eye
            padding = 5
            x1 = max(0, minc - padding)
            y1 = max(0, minr - padding)
            x2 = min(w, maxc + padding)
            y2 = min(h//3, maxr + padding)  # Keep within upper region
            
            eyes.append((x1, y1, x2, y2))
        
        return eyes
    
    def _detect_coronal_eyes(self, gray: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect eyes in coronal brain images"""
        eyes = []
        
        # In coronal view, eyes appear in the lower portion of the image
        lower_region = gray[h//2:, :]
        
        # Similar approach as axial but adjusted for coronal anatomy
        threshold = np.percentile(lower_region[lower_region > 0], 25)
        eye_mask = lower_region < threshold
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eye_mask = cv2.morphologyEx(eye_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        labeled = measure.label(eye_mask)
        regions = measure.regionprops(labeled)
        
        eye_candidates = []
        for region in regions:
            if (self.min_eye_area < region.area < self.min_eye_area * 15 and
                region.eccentricity < 0.8):
                eye_candidates.append(region)
        
        eye_candidates = sorted(eye_candidates, key=lambda r: r.area, reverse=True)[:2]
        
        for region in eye_candidates:
            minr, minc, maxr, maxc = region.bbox
            
            padding = 5
            x1 = max(0, minc - padding)
            y1 = max(0, minr + h//2 - padding)  # Adjust for lower region offset
            x2 = min(w, maxc + padding)
            y2 = min(h, maxr + h//2 + padding)
            
            eyes.append((x1, y1, x2, y2))
        
        return eyes
    
    def generate_prompt_boxes(
        self, 
        image: np.ndarray, 
        dicom_dataset: Optional[pydicom.Dataset] = None,
        include_eyes: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate bounding box prompts for brain structures
        
        Args:
            image: Input medical image (grayscale or RGB)
            dicom_dataset: Optional DICOM dataset with metadata
            include_eyes: Whether to include eye detection
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        all_boxes = []
        
        try:
            # Extract anatomical information
            anatomical_info = {}
            if dicom_dataset:
                anatomical_info = self.extract_anatomical_info(dicom_dataset)
            
            # Detect brain regions
            brain_boxes = self.detect_brain_bounds(image, anatomical_info)
            all_boxes.extend(brain_boxes)
            
            # Detect eyes if requested
            if include_eyes:
                eye_boxes = self.detect_eye_bounds(image, anatomical_info)
                all_boxes.extend(eye_boxes)
            
            # If no boxes detected, add default center box
            if not all_boxes:
                h, w = image.shape[:2]
                default_box = self._generate_default_brain_box(h, w)
                all_boxes.append(default_box)
                logger.warning("No brain regions detected, using default center box")
            
            logger.info(f"Generated {len(all_boxes)} total bounding box prompts")
            return all_boxes
            
        except Exception as e:
            logger.error(f"Error generating prompt boxes: {e}")
            # Fallback: return center box
            try:
                h, w = image.shape[:2]
                default_box = self._generate_default_brain_box(h, w)
                logger.warning("Error in detection, using fallback center box")
                return [default_box]
            except:
                logger.error("Failed to generate any prompts")
                return []
    
    def generate_prompts_for_volume(
        self,
        volume_data: List[Tuple[np.ndarray, pydicom.Dataset]],
        output_file: Optional[str] = None,
        slice_indices: Optional[List[int]] = None
    ) -> Dict[str, Dict]:
        """
        Generate prompts for an entire volume of brain images
        
        Args:
            volume_data: List of (image, dicom_dataset) tuples
            output_file: Optional path to save prompts JSON file
            slice_indices: Optional list of slice indices to process (0-based)
            
        Returns:
            Dictionary with prompts for each slice
        """
        all_prompts = {}
        
        # Determine which slices to process
        if slice_indices is None:
            process_indices = range(len(volume_data))
        else:
            process_indices = [i for i in slice_indices if 0 <= i < len(volume_data)]
        
        for i in process_indices:
            if i >= len(volume_data):
                continue
                
            image, dicom_ds = volume_data[i]
            
            # Generate bounding boxes for this slice
            boxes = self.generate_prompt_boxes(image, dicom_ds)
            
            if boxes:
                # Convert to MEDSAM2 format (UI slice numbers are 1-based)
                slice_key = str(i + 1)  # Convert to 1-based for UI compatibility
                all_prompts[slice_key] = {
                    "boxes": boxes
                }
                
                logger.info(f"Generated {len(boxes)} prompts for slice {slice_key}")
          # Save to file if requested
        if output_file:
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(all_prompts, f, indent=2)
                logger.info(f"Saved prompts to {output_file}")
            except Exception as e:
                logger.error(f"Error saving prompts to {output_file}: {e}")
                # Try saving to current directory with just filename
                try:
                    filename = os.path.basename(output_file)
                    with open(filename, 'w') as f:
                        json.dump(all_prompts, f, indent=2)
                    logger.info(f"Saved prompts to {filename} in current directory")
                except Exception as e2:
                    logger.error(f"Failed to save prompts: {e2}")
        
        return all_prompts


def create_brain_prompts_for_directory(
    dicom_dir: str,
    output_file: str,
    include_eyes: bool = True,
    slice_range: Optional[Tuple[int, int]] = None
) -> Dict[str, Dict]:
    """
    Convenience function to create brain prompts for a DICOM directory
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        output_file: Path to save prompts JSON file
        include_eyes: Whether to include eye detection
        slice_range: Optional (start, end) slice range to process
        
    Returns:
        Dictionary with generated prompts
    """
    detector = BrainROIDetector()
    detector.eye_detection_enabled = include_eyes
    
    # Load DICOM files (simplified version)
    import glob
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    # Load and sort DICOM files
    volume_data = []
    for file_path in sorted(dicom_files):
        try:
            dicom_ds = pydicom.dcmread(file_path)
            pixel_array = dicom_ds.pixel_array.astype(np.float32)
            
            # Basic windowing
            if hasattr(dicom_ds, 'WindowCenter') and hasattr(dicom_ds, 'WindowWidth'):
                center = float(dicom_ds.WindowCenter[0] if isinstance(dicom_ds.WindowCenter, list) else dicom_ds.WindowCenter)
                width = float(dicom_ds.WindowWidth[0] if isinstance(dicom_ds.WindowWidth, list) else dicom_ds.WindowWidth)
                
                windowed = np.clip(pixel_array, center - width/2, center + width/2)
                windowed = ((windowed - (center - width/2)) / width * 255).astype(np.uint8)
            else:
                # Normalize to 0-255
                windowed = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            volume_data.append((windowed, dicom_ds))
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not volume_data:
        raise ValueError(f"No valid DICOM files could be loaded from {dicom_dir}")
    
    # Apply slice range if specified
    slice_indices = None
    if slice_range:
        start, end = slice_range
        slice_indices = list(range(max(0, start), min(len(volume_data), end)))
    
    # Generate prompts
    prompts = detector.generate_prompts_for_volume(
        volume_data, 
        output_file, 
        slice_indices
    )
    
    return prompts
