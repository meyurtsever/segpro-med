"""
Annotation helper functions for extracting and calculating annotation properties
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def extract_coordinates(ann_data: Dict[str, Any]) -> List:
    """Extract coordinates from annotation data"""
    try:
        shape_type = ann_data.get('type', '')
        
        # Handle polygon, freehand, polyline and similar point-based shapes
        if shape_type in ('polygon', 'freehand', 'polyline') and 'points' in ann_data:
            points = ann_data['points']
            if isinstance(points, list) and points:
                coords = []
                for p in points:
                    if isinstance(p, dict):
                        # Dict with x, y keys
                        coords.append([p.get('x', 0), p.get('y', 0)])
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        # List/tuple like [x, y]
                        coords.append([p[0], p[1]])
                    else:
                        logger.warning(f"Unexpected point format: {type(p)}")
                        continue
                return coords
        elif shape_type in ('box', 'rect', 'rectangle'):
            if 'xmin' in ann_data and 'ymin' in ann_data:
                return [
                    [ann_data['xmin'], ann_data['ymin']],
                    [ann_data['xmax'], ann_data['ymin']],
                    [ann_data['xmax'], ann_data['ymax']],
                    [ann_data['xmin'], ann_data['ymax']]
                ]
        
        return []
    except Exception as e:
        logger.error(f"Error extracting coordinates from shape type '{shape_type}': {e}", exc_info=True)
        return []


def extract_bbox(ann_data: Dict[str, Any]) -> List:
    """Extract bounding box from annotation data"""
    try:
        if 'xmin' in ann_data and 'ymin' in ann_data:
            return [
                ann_data.get('xmin', 0),
                ann_data.get('ymin', 0),
                ann_data.get('xmax', 0),
                ann_data.get('ymax', 0)
            ]
        elif 'points' in ann_data:
            points = ann_data['points']
            if isinstance(points, list) and points:
                xs = []
                ys = []
                for p in points:
                    if isinstance(p, dict):
                        xs.append(p.get('x', 0))
                        ys.append(p.get('y', 0))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        xs.append(p[0])
                        ys.append(p[1])
                
                if xs and ys:
                    return [min(xs), min(ys), max(xs), max(ys)]
        
        return [0, 0, 0, 0]
    except Exception as e:
        logger.error(f"Error extracting bbox: {e}", exc_info=True)
        return [0, 0, 0, 0]


def calculate_area(ann_data: Dict[str, Any]) -> float:
    """Calculate area of annotation"""
    try:
        shape_type = ann_data.get('type', '')
        
        # Handle polygon, freehand, polyline
        if shape_type in ('polygon', 'freehand', 'polyline') and 'points' in ann_data:
            points = ann_data['points']
            if isinstance(points, list) and len(points) >= 3:
                # Shoelace formula for polygon area
                area = 0
                n = len(points)
                for i in range(n):
                    j = (i + 1) % n
                    
                    # Extract x1, y1
                    if isinstance(points[i], dict):
                        x1 = points[i].get('x', 0)
                        y1 = points[i].get('y', 0)
                    elif isinstance(points[i], (list, tuple)) and len(points[i]) >= 2:
                        x1 = points[i][0]
                        y1 = points[i][1]
                    else:
                        continue
                    
                    # Extract x2, y2
                    if isinstance(points[j], dict):
                        x2 = points[j].get('x', 0)
                        y2 = points[j].get('y', 0)
                    elif isinstance(points[j], (list, tuple)) and len(points[j]) >= 2:
                        x2 = points[j][0]
                        y2 = points[j][1]
                    else:
                        continue
                    
                    area += x1 * y2 - x2 * y1
                return abs(area) / 2.0
        elif shape_type in ('box', 'rect', 'rectangle'):
            if 'xmin' in ann_data and 'ymin' in ann_data:
                width = ann_data.get('xmax', 0) - ann_data.get('xmin', 0)
                height = ann_data.get('ymax', 0) - ann_data.get('ymin', 0)
                return abs(width * height)
        
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating area for shape type '{shape_type}': {e}", exc_info=True)
        return 0.0
