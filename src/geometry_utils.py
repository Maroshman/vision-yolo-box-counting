"""
Geometry Utilities for Bounding Box Operations

Handles spatial relationships between detected boxes and labels.
"""

from typing import List, Tuple, Dict
import numpy as np


def get_bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def is_point_inside_bbox(point: Tuple[float, float], bbox: List[int]) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: Point as (x, y)
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        True if point is inside bbox, False otherwise
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def is_label_inside_box(label_bbox: List[int], box_bbox: List[int]) -> bool:
    """
    Check if a label is inside a box by testing if the label's center point
    is within the box's bounding box.
    
    Args:
        label_bbox: Label bounding box as [x1, y1, x2, y2]
        box_bbox: Box bounding box as [x1, y1, x2, y2]
        
    Returns:
        True if label center is inside box, False otherwise
    """
    label_center = get_bbox_center(label_bbox)
    return is_point_inside_bbox(label_center, box_bbox)


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x1, y1, x2, y2]
        bbox2: Second bounding box as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def match_labels_to_boxes(
    boxes: List[Dict],
    labels: List[Dict],
    method: str = 'center'
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Match labels to boxes based on spatial containment.
    
    Args:
        boxes: List of box dictionaries with 'bbox' key
        labels: List of label dictionaries with 'bbox' key
        method: Matching method ('center' or 'iou')
                'center': label center must be inside box
                'iou': use IoU threshold for matching
        
    Returns:
        Tuple of:
        - Dictionary mapping box index to list of label indices
        - List of orphan label indices (not matched to any box)
    """
    box_to_labels = {i: [] for i in range(len(boxes))}
    matched_labels = set()
    
    for label_idx, label in enumerate(labels):
        label_bbox = label['bbox']
        matched = False
        
        for box_idx, box in enumerate(boxes):
            box_bbox = box['bbox']
            
            if method == 'center':
                if is_label_inside_box(label_bbox, box_bbox):
                    box_to_labels[box_idx].append(label_idx)
                    matched_labels.add(label_idx)
                    matched = True
                    break  # Label matched to first containing box
            
            elif method == 'iou':
                iou = calculate_iou(label_bbox, box_bbox)
                if iou > 0.3:  # Threshold for IoU matching
                    box_to_labels[box_idx].append(label_idx)
                    matched_labels.add(label_idx)
                    matched = True
                    break
    
    # Find orphan labels (not matched to any box)
    orphan_labels = [i for i in range(len(labels)) if i not in matched_labels]
    
    return box_to_labels, orphan_labels


def get_bbox_area(bbox: List[int]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def expand_bbox(bbox: List[int], margin: int) -> List[int]:
    """
    Expand a bounding box by a margin.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        margin: Pixels to expand in each direction
        
    Returns:
        Expanded bounding box as [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        x2 + margin,
        y2 + margin
    ]
