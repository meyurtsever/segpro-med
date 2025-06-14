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
        self.max_detected_regions = 12  # Increase to 12 boxes per slice for better coverage
        self.padding_percentage = 0.02  # Reduce padding to 2% to keep boxes tighter
        
        # Dynamic box sizing parameters based on slice position
        self.slice_position_cache = {}  # Cache slice characteristics
        
    def determine_slice_characteristics(self, image: np.ndarray, slice_index: Optional[int] = None) -> Dict:
        """
        Determine slice characteristics for dynamic box sizing
        
        Args:
            image: Input medical image
            slice_index: Optional slice index for caching
            
        Returns:
            Dictionary with slice characteristics
        """
        cache_key = f"slice_{slice_index}" if slice_index is not None else "unknown"
        
        if cache_key in self.slice_position_cache:
            return self.slice_position_cache[cache_key]
        
        h, w = image.shape[:2]
        
        # Analyze brain content density
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Calculate brain tissue intensity
        non_zero_pixels = gray_image[gray_image > 0]
        if len(non_zero_pixels) > 0:
            brain_intensity = np.percentile(non_zero_pixels, 75)
            brain_area_ratio = np.sum(gray_image > brain_intensity * 0.3) / (h * w)
        else:
            brain_intensity = np.mean(gray_image)
            brain_area_ratio = 0.1
        
        # Detect anatomical features
        edges = cv2.Canny((gray_image * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Classify slice type based on brain content
        if brain_area_ratio < 0.05:
            slice_type = "minimal"  # Very top/bottom of head
            box_scale = 0.6  # Smaller boxes
            grid_density = 2  # Fewer boxes (2x2 grid)
        elif brain_area_ratio < 0.15:
            slice_type = "transitional"  # Beginning/end of brain
            box_scale = 0.75  # Medium boxes
            grid_density = 3  # Medium coverage (3x3 grid)
        elif brain_area_ratio > 0.35:
            slice_type = "dense"  # Core brain regions
            box_scale = 1.0  # Full-size boxes
            grid_density = 4  # Maximum coverage (4x3 grid)
        else:
            slice_type = "standard"  # Regular brain tissue
            box_scale = 0.85  # Standard boxes
            grid_density = 3  # Standard coverage (3x3 grid)
        
        characteristics = {
            'slice_type': slice_type,
            'brain_area_ratio': brain_area_ratio,
            'edge_density': edge_density,
            'brain_intensity': brain_intensity,
            'box_scale': box_scale,
            'grid_density': grid_density,
            'recommended_boxes': min(12, grid_density * grid_density + 3)  # Grid + 3 additional
        }
        
        # Cache the result
        self.slice_position_cache[cache_key] = characteristics
        
        logger.info(f"Slice characteristics: {slice_type} (brain_ratio={brain_area_ratio:.3f}, "
                   f"scale={box_scale}, density={grid_density}, boxes={characteristics['recommended_boxes']})")
        
        return characteristics
    
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
              # Use multiple thresholds to better detect brain tissue
            # Brain tissue typically has moderate intensity values
            non_zero_pixels = blurred[blurred > 0]
            if len(non_zero_pixels) == 0:
                return [self._generate_default_brain_box(h, w)]
              # Use adaptive thresholding based on image histogram
            threshold_low = np.percentile(non_zero_pixels, self.brain_tissue_threshold_percentile)
            threshold_high = np.percentile(non_zero_pixels, 75)  # Lower upper threshold to exclude skull (was 85)
            
            # If thresholds are too close (uniform intensity), use a wider range
            if threshold_high - threshold_low < 10:
                threshold_low = max(0, threshold_low - 20)
                threshold_high = min(255, threshold_high + 20)
            
            # Create binary mask for potential brain tissue (moderate intensity)
            # Exclude very high intensities that are likely skull
            brain_mask = (blurred > threshold_low) & (blurred < threshold_high)
            
            # Additional filtering to remove noise and improve brain detection
            # Remove very small bright spots that might be noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            brain_mask = cv2.morphologyEx(brain_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
            
            # Fill holes and connect nearby regions
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel_large)
            
            # Remove regions that are too close to the image border (likely skull or artifacts)
            border_margin = min(h, w) // 20  # 5% margin from edges
            brain_mask[:border_margin, :] = 0
            brain_mask[-border_margin:, :] = 0
            brain_mask[:, :border_margin] = 0
            brain_mask[:, -border_margin:] = 0
            
            # Find connected components
            labeled_image = measure.label(brain_mask)
            regions = measure.regionprops(labeled_image)
              # Filter regions by area and shape constraints
            valid_regions = []
            for region in regions:
                # Area constraint: not too small, not too large
                # Reduce max area to exclude skull and focus on brain tissue
                if (self.min_brain_area < region.area < h * w * 0.4):  # Max 40% of image (down from 60%)
                    # Shape constraint: not too elongated
                    if region.eccentricity < 0.95:  # Not extremely elongated
                        # Additional constraint: exclude regions that are too close to image borders
                        minr, minc, maxr, maxc = region.bbox
                        border_threshold = min(h, w) * 0.1  # 10% from edge
                        if (minc > border_threshold and minr > border_threshold and 
                            maxc < w - border_threshold and maxr < h - border_threshold):
                            valid_regions.append(region)
                        else:
                            logger.debug(f"Region excluded: too close to border")
                    else:
                        logger.debug(f"Region excluded: too elongated (eccentricity: {region.eccentricity})")
                else:
                    logger.debug(f"Region excluded: area {region.area} not in range [{self.min_brain_area}, {h * w * 0.4}]")
            
            # Sort regions by area (largest first)
            valid_regions = sorted(valid_regions, key=lambda r: r.area, reverse=True)
            
            # Generate bounding boxes based on image orientation
            if anatomical_info.get('image_orientation') == 'axial':
                bounding_boxes.extend(self._generate_axial_brain_boxes(valid_regions, h, w))
            elif anatomical_info.get('image_orientation') == 'sagittal':
                bounding_boxes.extend(self._generate_sagittal_brain_boxes(valid_regions, h, w))
            elif anatomical_info.get('image_orientation') == 'coronal':
                bounding_boxes.extend(self._generate_coronal_brain_boxes(valid_regions, h, w))
            else:
                # Default: treat as axial
                bounding_boxes.extend(self._generate_axial_brain_boxes(valid_regions, h, w))
            
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
        
        # Use configurable max regions instead of hardcoded 3
        for region in regions[:self.max_detected_regions]:
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Use smaller, more precise padding
            padding_h = int(h * self.padding_percentage)
            padding_w = int(w * self.padding_percentage)
            
            x1 = max(0, minc - padding_w)
            y1 = max(0, minr - padding_h)
            x2 = min(w, maxc + padding_w)
            y2 = min(h, maxr + padding_h)            
            boxes.append((x1, y1, x2, y2))
          # If we have a large central region, add multiple smaller, focused boxes within it
        if boxes and regions:
            main_region = regions[0]
            if main_region.area > h * w * 0.05:  # If main region is >5% of image (much lower threshold)
                # Add focused boxes for specific brain structures within the detected region
                minr, minc, maxr, maxc = main_region.bbox
                
                # Calculate smaller focused boxes within the brain region
                region_width = maxc - minc
                region_height = maxr - minr
                
                # Create grid of smaller boxes within the brain region
                # Make boxes even smaller for better precision
                box_size_x = max(20, region_width // 4)  # Minimum 20 pixels, divide by 4 instead of 3
                box_size_y = max(20, region_height // 4)
                
                # Create a 3x3 grid of focused boxes within the brain region
                for i in range(3):  # 3 rows
                    for j in range(3):  # 3 columns
                        start_x = minc + j * (region_width // 3)
                        start_y = minr + i * (region_height // 3)
                        end_x = min(maxc, start_x + box_size_x)
                        end_y = min(maxr, start_y + box_size_y)
                        
                        # Only add if box is reasonably sized and within brain bounds
                        if ((end_x - start_x) > 15 and (end_y - start_y) > 15 and
                            start_x >= minc and start_y >= minr and
                            end_x <= maxc and end_y <= maxr):
                            focused_box = (start_x, start_y, end_x, end_y)
                            boxes.append(focused_box)
        
        return boxes
    
    def _generate_sagittal_brain_boxes(self, regions: List, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes for sagittal brain images"""
        boxes = []
        
        # Use configurable max regions instead of hardcoded 2
        for region in regions[:min(self.max_detected_regions, 4)]:  # Limit to 4 for sagittal
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Use smaller padding
            padding_h = int(h * self.padding_percentage * 1.5)  # Slightly more padding for sagittal
            padding_w = int(w * self.padding_percentage * 1.5)
            
            x1 = max(0, minc - padding_w)
            y1 = max(0, minr - padding_h)
            x2 = min(w, maxc + padding_w)
            y2 = min(h, maxr + padding_h)            
            boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _generate_coronal_brain_boxes(self, regions: List, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes for coronal brain images"""
        boxes = []
        
        # Use configurable max regions instead of hardcoded 2
        for region in regions[:min(self.max_detected_regions, 4)]:  # Limit to 4 for coronal
            if region.area < self.min_brain_area:
                continue
                
            minr, minc, maxr, maxc = region.bbox
            
            # Use smaller padding
            padding_h = int(h * self.padding_percentage * 1.5)  # Slightly more padding for coronal
            padding_w = int(w * self.padding_percentage * 1.5)
            
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
        include_eyes: bool = True,
        slice_index: Optional[int] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate bounding box prompts for brain structures with brain-constrained positioning
        Enhanced version that detects brain boundary and places all boxes within brain tissue
        
        Args:
            image: Input medical image (grayscale or RGB)
            dicom_dataset: Optional DICOM dataset with metadata
            include_eyes: Whether to include eye detection
            slice_index: Optional slice index for caching slice characteristics
            
        Returns:
            List of 12 bounding boxes as (x1, y1, x2, y2) tuples, all positioned within brain
        """
        try:
            # Convert to grayscale if needed for brain boundary detection
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # Step 1: Detect actual brain boundary (excluding skull and background)
            logger.info(f"Detecting brain boundary for slice {slice_index if slice_index else 'unknown'}")
            brain_boundary = self.detect_brain_boundary(gray_image)
            
            # Step 2: Generate 12 boxes positioned ONLY inside the brain boundary
            logger.info("Generating brain-constrained boxes")
            brain_boxes = self.generate_brain_constrained_boxes(
                gray_image, 
                brain_boundary, 
                target_boxes=12
            )
            
            # Step 3: Add eyes if requested and if brain region is large enough for orbital structures
            all_boxes = brain_boxes.copy()
            if include_eyes and brain_boundary['brain_area_ratio'] > 0.15:  # Only if substantial brain tissue
                anatomical_info = {}
                if dicom_dataset:
                    anatomical_info = self.extract_anatomical_info(dicom_dataset)
                
                eye_boxes = self.detect_eye_bounds(gray_image, anatomical_info)
                # Add up to 2 eye boxes if they don't overlap significantly with brain boxes
                for eye_box in eye_boxes[:2]:
                    overlaps = False
                    for brain_box in brain_boxes:
                        if self._boxes_overlap_significantly(eye_box, brain_box, overlap_threshold=0.3):
                            overlaps = True
                            break
                    if not overlaps:
                        all_boxes.append(eye_box)
            
            # Ensure exactly 12 boxes by taking the first 12
            final_boxes = all_boxes[:12]
            
            # If we have fewer than 12, pad with smaller boxes in remaining brain areas
            if len(final_boxes) < 12:
                needed = 12 - len(final_boxes)
                additional_boxes = self._generate_small_brain_boxes(
                    gray_image, brain_boundary, needed, existing_boxes=final_boxes
                )
                final_boxes.extend(additional_boxes[:needed])
            
            # Final validation: ensure all boxes are within image bounds
            h, w = gray_image.shape
            validated_boxes = []
            for box in final_boxes:
                x1, y1, x2, y2 = box
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(x1+1, min(w, x2))
                y2 = max(y1+1, min(h, y2))
                validated_boxes.append((x1, y1, x2, y2))
            
            # Ensure we have exactly 12 boxes
            while len(validated_boxes) < 12:
                # Add small fallback box at brain center
                brain_center_row, brain_center_col = brain_boundary['brain_centroid']
                center_x, center_y = int(brain_center_col), int(brain_center_row)
                fallback_box = (
                    max(0, center_x - 10),
                    max(0, center_y - 10), 
                    min(w, center_x + 10),
                    min(h, center_y + 10)
                )
                validated_boxes.append(fallback_box)
            
            final_boxes = validated_boxes[:12]
            
            logger.info(f"Generated {len(final_boxes)} brain-constrained bounding boxes")
            return final_boxes
            
        except Exception as e:
            logger.error(f"Error generating brain-constrained prompt boxes: {e}")
            # Fallback: return 12 standard grid boxes
            try:
                h, w = image.shape[:2]
                fallback_boxes = self._generate_standard_grid_boxes(h, w, 12)
                logger.warning(f"Error in brain detection, using fallback with {len(fallback_boxes)} boxes")
                return fallback_boxes
            except Exception as fallback_error:
                logger.error(f"Failed to generate any prompts: {fallback_error}")
                return []
    
    def generate_anatomical_boxes(
        self, 
        image: np.ndarray, 
        slice_characteristics: Dict, 
        anatomical_info: Dict
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate anatomically-informed boxes based on brain regions
        
        Args:
            image: Input medical image
            slice_characteristics: Slice analysis results
            anatomical_info: Anatomical information from DICOM
            
        Returns:
            List of anatomically-placed bounding boxes
        """
        h, w = image.shape[:2]
        boxes = []
        
        # Get slice characteristics
        slice_type = slice_characteristics['slice_type']
        box_scale = slice_characteristics['box_scale']
        
        # Define anatomical regions based on slice type
        if slice_type in ['dense', 'standard']:
            # Core brain regions - focus on important areas
            center_x, center_y = w // 2, h // 2
            
            # Central brain region (larger)
            central_size = int(min(h, w) * 0.3 * box_scale)
            boxes.append((
                max(0, center_x - central_size // 2),
                max(0, center_y - central_size // 2),
                min(w, center_x + central_size // 2),
                min(h, center_y + central_size // 2)
            ))
            
            # Left and right hemisphere regions
            hemi_size = int(min(h, w) * 0.25 * box_scale)
            # Left hemisphere
            left_x = w // 4
            boxes.append((
                max(0, left_x - hemi_size // 2),
                max(0, center_y - hemi_size // 2),
                min(w, left_x + hemi_size // 2),
                min(h, center_y + hemi_size // 2)
            ))
            
            # Right hemisphere
            right_x = 3 * w // 4
            boxes.append((
                max(0, right_x - hemi_size // 2),
                max(0, center_y - hemi_size // 2),
                min(w, right_x + hemi_size // 2),
                min(h, center_y + hemi_size // 2)
            ))
            
        elif slice_type == 'transitional':
            # Transitional slices - focus on central areas
            center_x, center_y = w // 2, h // 2
            region_size = int(min(h, w) * 0.2 * box_scale)
            
            # Central region only
            boxes.append((
                max(0, center_x - region_size // 2),
                max(0, center_y - region_size // 2),
                min(w, center_x + region_size // 2),
                min(h, center_y + region_size // 2)
            ))
        
        return boxes
    
    def _boxes_overlap_significantly(
        self, 
        box1: Tuple[int, int, int, int], 
        box2: Tuple[int, int, int, int], 
        overlap_threshold: float = 0.5
    ) -> bool:
        """Check if two boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return False  # No intersection
        
        # Calculate areas
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio
        min_area = min(box1_area, box2_area)
        overlap_ratio = intersection_area / min_area if min_area > 0 else 0
        
        return overlap_ratio > overlap_threshold
    
    def _generate_fallback_boxes(
        self, 
        h: int, 
        w: int, 
        existing_boxes: List[Tuple[int, int, int, int]], 
        count: int
    ) -> List[Tuple[int, int, int, int]]:
        """Generate fallback boxes to reach target count"""
        boxes = []
        
        # Simple grid approach for fallback
        box_size = min(h, w) // 5
        positions = [
            (w//4, h//4), (3*w//4, h//4),      # Top row
            (w//4, h//2), (3*w//4, h//2),      # Middle row  
            (w//4, 3*h//4), (3*w//4, 3*h//4),  # Bottom row
            (w//2, h//4), (w//2, 3*h//4),      # Center column
        ]
        
        for i, (cx, cy) in enumerate(positions):
            if len(boxes) >= count:
                break
                
            x1 = max(0, cx - box_size // 2)
            y1 = max(0, cy - box_size // 2)
            x2 = min(w, cx + box_size // 2)
            y2 = min(h, cy + box_size // 2)
            
            new_box = (x1, y1, x2, y2)
            
            # Check if this box overlaps significantly with existing boxes
            overlaps = False
            for existing_box in existing_boxes:
                if self._boxes_overlap_significantly(new_box, existing_box, 0.6):
                    overlaps = True
                    break
            
            if not overlaps:
                boxes.append(new_box)
        
        return boxes
    
    def _select_best_boxes(
        self, 
        boxes: List[Tuple[int, int, int, int]], 
        target_count: int, 
        slice_characteristics: Dict
    ) -> List[Tuple[int, int, int, int]]:
        """Select the best boxes based on position and characteristics"""
        if len(boxes) <= target_count:
            return boxes
        
        # Score boxes based on position and size
        scored_boxes = []
        h, w = slice_characteristics.get('image_shape', (512, 512))
        center_x, center_y = w // 2, h // 2
        
        for box in boxes:
            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            
            # Distance from image center (closer = better)
            center_dist = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
            center_score = 1.0 / (1.0 + center_dist / min(h, w))
            
            # Box size score (medium size preferred)
            box_area = (x2 - x1) * (y2 - y1)
            ideal_area = (h * w) * 0.05  # 5% of image
            size_score = 1.0 / (1.0 + abs(box_area - ideal_area) / ideal_area)
            
            # Combined score
            total_score = center_score * 0.7 + size_score * 0.3
            scored_boxes.append((total_score, box))
        
        # Sort by score (descending) and take top boxes
        scored_boxes.sort(key=lambda x: x[0], reverse=True)
        return [box for score, box in scored_boxes[:target_count]]
    
    def _generate_standard_grid_boxes(self, h: int, w: int, count: int) -> List[Tuple[int, int, int, int]]:
        """Generate a standard grid of boxes as fallback"""
        boxes = []
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(count)))
        box_h = h // grid_size
        box_w = w // grid_size
        
        for row in range(grid_size):
            for col in range(grid_size):
                if len(boxes) >= count:
                    break
                    
                x1 = col * box_w
                y1 = row * box_h
                x2 = min(w, (col + 1) * box_w)
                y2 = min(h, (row + 1) * box_h)
                
                boxes.append((x1, y1, x2, y2))
        
        return boxes[:count]
    
    def detect_brain_boundary(self, image: np.ndarray) -> Dict:
        """
        Detect the actual brain boundary, excluding skull and background
        
        Args:
            image: Input medical image
            
        Returns:
            Dictionary with brain boundary information
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Normalize to 0-255 range
        if gray_image.max() <= 1.0:
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            gray_image = gray_image.astype(np.uint8)
        
        # Step 1: Remove background (solid black pixels)
        # Background is typically very dark (close to 0)
        background_threshold = np.percentile(gray_image[gray_image > 0], 5) if np.any(gray_image > 0) else 10
        background_mask = gray_image > background_threshold
        
        # Step 2: Detect skull boundary
        # Skull appears as bright white/gray boundary
        skull_threshold = np.percentile(gray_image[gray_image > 0], 85) if np.any(gray_image > 0) else 200
        
        # Create brain tissue mask (between background and skull)
        brain_min = background_threshold
        brain_max = skull_threshold * 0.8  # Brain tissue is darker than skull
        brain_mask = (gray_image >= brain_min) & (gray_image <= brain_max)
        
        # Step 3: Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        brain_mask = cv2.morphologyEx(brain_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
        
        # Step 4: Find the largest connected component (main brain region)
        from skimage import measure
        labeled_regions = measure.label(brain_mask)
        if labeled_regions.max() > 0:
            regions = measure.regionprops(labeled_regions)
            largest_region = max(regions, key=lambda r: r.area)
            
            # Create mask for largest region only
            main_brain_mask = labeled_regions == largest_region.label
            
            # Get bounding box of brain region
            brain_bbox = largest_region.bbox  # (min_row, min_col, max_row, max_col)
            brain_centroid = largest_region.centroid
            brain_area = largest_region.area
            
        else:
            # Fallback if no brain region detected
            main_brain_mask = np.ones((h, w), dtype=bool)
            brain_bbox = (h//4, w//4, 3*h//4, 3*w//4)
            brain_centroid = (h//2, w//2)
            brain_area = h * w // 4
        
        # Step 5: Calculate brain region properties
        brain_height = brain_bbox[2] - brain_bbox[0]
        brain_width = brain_bbox[3] - brain_bbox[1]
        
        # Step 6: Analyze brain shape for intelligent box placement
        # Find brain contour for more precise boundary
        contours, _ = cv2.findContours(main_brain_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            brain_contour = max(contours, key=cv2.contourArea)
            # Create a more precise mask using the contour
            precise_brain_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(precise_brain_mask, [brain_contour], 255)
            precise_brain_mask = precise_brain_mask > 0
        else:
            precise_brain_mask = main_brain_mask
        
        boundary_info = {
            'brain_mask': precise_brain_mask,
            'brain_bbox': brain_bbox  # (min_row, min_col, max_row, max_col)
        }
        
        logger.info(f"Brain boundary detected: bbox={brain_bbox}")
        
        return boundary_info
    
    def generate_brain_constrained_boxes(
        self, 
        image: np.ndarray, 
        brain_boundary: Dict, 
        target_boxes: int = 12
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate exactly 12 boxes positioned INSIDE the brain boundary
        
        Args:
            image: Input medical image
            brain_boundary: Brain boundary detection results
            target_boxes: Number of boxes to generate (default 12)
            
        Returns:
            List of 12 bounding boxes positioned within brain tissue
        """
        h, w = image.shape[:2]
        brain_mask = brain_boundary['brain_mask']
        brain_bbox = brain_boundary['brain_bbox']  # (min_row, min_col, max_row, max_col)
        
        # Extract brain region coordinates
        min_row, min_col, max_row, max_col = brain_bbox
        brain_height = max_row - min_row
        brain_width = max_col - min_col
        
        # Calculate box size based on brain dimensions (not image dimensions)
        box_height = max(20, brain_height // 4)  # Ensure minimum box size
        box_width = max(20, brain_width // 4)
        
        boxes = []
        
        # Strategy 1: Grid-based placement within brain boundary
        # Use 3x4 grid for 12 boxes
        grid_rows, grid_cols = 3, 4
        
        # Calculate grid step sizes with overlap
        step_row = max(box_height // 2, (brain_height - box_height) // (grid_rows - 1)) if grid_rows > 1 else 0
        step_col = max(box_width // 2, (brain_width - box_width) // (grid_cols - 1)) if grid_cols > 1 else 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate candidate position within brain bbox
                candidate_row = min_row + row * step_row
                candidate_col = min_col + col * step_col
                
                # Ensure box stays within brain bbox
                box_min_row = max(min_row, candidate_row)
                box_min_col = max(min_col, candidate_col)
                box_max_row = min(max_row, box_min_row + box_height)
                box_max_col = min(max_col, box_min_col + box_width)
                
                # Adjust if box extends beyond brain boundary
                if box_max_row - box_min_row < 20 or box_max_col - box_min_col < 20:
                    continue  # Skip too small boxes
                
                # Check if box center is within brain mask
                box_center_row = (box_min_row + box_max_row) // 2
                box_center_col = (box_min_col + box_max_col) // 2
                
                if (0 <= box_center_row < h and 0 <= box_center_col < w and 
                    brain_mask[box_center_row, box_center_col]):
                    
                    # Verify significant overlap with brain tissue
                    box_brain_overlap = np.sum(brain_mask[box_min_row:box_max_row, box_min_col:box_max_col])
                    box_area = (box_max_row - box_min_row) * (box_max_col - box_min_col)
                    overlap_ratio = box_brain_overlap / box_area if box_area > 0 else 0
                    
                    if overlap_ratio > 0.5:  # At least 50% overlap with brain tissue
                        boxes.append((box_min_col, box_min_row, box_max_col, box_max_row))
        
        # Strategy 2: If we don't have enough boxes, add strategic positions
        while len(boxes) < target_boxes:
            # Add boxes at strategic brain locations
            brain_center_row, brain_center_col = int(brain_boundary['brain_centroid'][0]), int(brain_boundary['brain_centroid'][1])
            
            # Try different offsets from brain center
            offsets = [
                (0, 0),                                      # Center
                (-brain_height//4, -brain_width//4),         # Top-left quadrant
                (-brain_height//4, brain_width//4),          # Top-right quadrant
                (brain_height//4, -brain_width//4),          # Bottom-left quadrant
                (brain_height//4, brain_width//4),           # Bottom-right quadrant
                (-brain_height//6, 0),                       # Top center
                (brain_height//6, 0),                        # Bottom center
                (0, -brain_width//6),                        # Left center
                (0, brain_width//6),                         # Right center
            ]
            
            added_box = False
            for offset_row, offset_col in offsets:
                if len(boxes) >= target_boxes:
                    break
                
                center_row = max(min_row, min(max_row, brain_center_row + offset_row))
                center_col = max(min_col, min(max_col, brain_center_col + offset_col))
                
                # Create box around this position
                box_min_row = max(min_row, center_row - box_height // 2)
                box_min_col = max(min_col, center_col - box_width // 2)
                box_max_row = min(max_row, box_min_row + box_height)
                box_max_col = min(max_col, box_min_col + box_width)
                
                candidate_box = (box_min_col, box_min_row, box_max_col, box_max_row)
                
                # Check if this box overlaps significantly with existing boxes
                overlaps = False
                for existing_box in boxes:
                    if self._boxes_overlap_significantly(candidate_box, existing_box, 0.5):
                        overlaps = True
                        break
                
                if not overlaps and brain_mask[center_row, center_col]:
                    boxes.append(candidate_box)
                    added_box = True
            
            if not added_box:
                break  # Avoid infinite loop
        
        # Ensure we have exactly target_boxes
        final_boxes = boxes[:target_boxes]
        
        # If still not enough, fill with smaller boxes
        if len(final_boxes) < target_boxes:
            smaller_box_size = min(box_height // 2, box_width // 2, 15)
            for _ in range(target_boxes - len(final_boxes)):
                # Add small boxes in remaining brain areas
                center_row = min_row + brain_height // 2
                center_col = min_col + brain_width // 2
                small_box = (
                    max(min_col, center_col - smaller_box_size),
                    max(min_row, center_row - smaller_box_size),
                    min(max_col, center_col + smaller_box_size),
                    min(max_row, center_row + smaller_box_size)
                )
                final_boxes.append(small_box)
        
        logger.info(f"Generated {len(final_boxes)} brain-constrained boxes (box size: {box_height}x{box_width})")
        
        return final_boxes[:target_boxes]
    
    def _generate_small_brain_boxes(
        self, 
        image: np.ndarray, 
        brain_boundary: Dict, 
        needed_boxes: int, 
        existing_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate small boxes within brain boundary to pad the list when needed
        
        Args:
            image: Input image
            brain_boundary: Brain boundary information
            needed_boxes: Number of additional boxes needed
            existing_boxes: List of existing boxes to avoid overlap
            
        Returns:
            List of small boxes within brain tissue
        """
        h, w = image.shape[:2]
        brain_mask = brain_boundary['brain_mask']
        brain_bbox = brain_boundary['brain_bbox']
        min_row, min_col, max_row, max_col = brain_bbox
        
        small_boxes = []
        small_box_size = 15  # Small boxes
        
        # Generate candidate positions within brain
        attempts = 0
        max_attempts = 50
        
        while len(small_boxes) < needed_boxes and attempts < max_attempts:
            attempts += 1
            
            # Random position within brain bounding box
            center_row = np.random.randint(min_row + small_box_size, max_row - small_box_size)
            center_col = np.random.randint(min_col + small_box_size, max_col - small_box_size)
            
            # Check if position is within brain tissue
            if not brain_mask[center_row, center_col]:
                continue
            
            # Create small box
            box_x1 = max(min_col, center_col - small_box_size)
            box_y1 = max(min_row, center_row - small_box_size)
            box_x2 = min(max_col, center_col + small_box_size)
            box_y2 = min(max_row, center_row + small_box_size)
            
            candidate_box = (box_x1, box_y1, box_x2, box_y2)
            
            # Check overlap with existing boxes
            overlaps = False
            for existing_box in existing_boxes + small_boxes:
                if self._boxes_overlap_significantly(candidate_box, existing_box, 0.3):
                    overlaps = True
                    break
            
            if not overlaps:
                small_boxes.append(candidate_box)
        
        return small_boxes

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
