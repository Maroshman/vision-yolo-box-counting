"""
Utility functions for the YOLO Box Counting Engine
"""

import os
import cv2
import numpy as np
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def save_config(config: Dict, config_path: str = "config.yaml"):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logging.error(f"Error saving config: {e}")

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Output file path
    """
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
        else:
            cv2.imwrite(output_path, image)
    except Exception as e:
        logging.error(f"Error saving image to {output_path}: {e}")

def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def calculate_box_area(box: List[int]) -> int:
    """
    Calculate area of a bounding box
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Box area in pixels
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two boxes
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def filter_overlapping_boxes(boxes: List[List[int]], confidences: List[float], 
                           iou_threshold: float = 0.5) -> Tuple[List[List[int]], List[float]]:
    """
    Filter overlapping boxes using Non-Maximum Suppression
    
    Args:
        boxes: List of bounding boxes
        confidences: List of confidence scores
        iou_threshold: IoU threshold for filtering
        
    Returns:
        Filtered boxes and confidences
    """
    if not boxes:
        return [], []
    
    # Sort by confidence (descending)
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    
    keep_indices = []
    
    for i in sorted_indices:
        keep = True
        for j in keep_indices:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)
    
    filtered_boxes = [boxes[i] for i in keep_indices]
    filtered_confidences = [confidences[i] for i in keep_indices]
    
    return filtered_boxes, filtered_confidences

def create_detection_summary(results_list: List[Dict]) -> Dict:
    """
    Create summary statistics from detection results
    
    Args:
        results_list: List of detection results
        
    Returns:
        Summary statistics
    """
    if not results_list:
        return {}
    
    total_images = len(results_list)
    total_boxes = sum(result['count'] for result in results_list)
    avg_boxes_per_image = total_boxes / total_images if total_images > 0 else 0
    
    box_counts = [result['count'] for result in results_list]
    confidence_scores = []
    for result in results_list:
        confidence_scores.extend(result['confidences'])
    
    summary = {
        'total_images': total_images,
        'total_boxes': total_boxes,
        'avg_boxes_per_image': avg_boxes_per_image,
        'min_boxes_per_image': min(box_counts) if box_counts else 0,
        'max_boxes_per_image': max(box_counts) if box_counts else 0,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'min_confidence': min(confidence_scores) if confidence_scores else 0,
        'max_confidence': max(confidence_scores) if confidence_scores else 0,
    }
    
    return summary

def plot_detection_statistics(results_list: List[Dict], output_path: str = None):
    """
    Create visualization of detection statistics
    
    Args:
        results_list: List of detection results
        output_path: Path to save plot (optional)
    """
    if not results_list:
        return
    
    # Prepare data
    box_counts = [result['count'] for result in results_list]
    confidence_scores = []
    for result in results_list:
        confidence_scores.extend(result['confidences'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Box Detection Statistics', fontsize=16)
    
    # Box count distribution
    axes[0, 0].hist(box_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Box Counts per Image')
    axes[0, 0].set_xlabel('Number of Boxes')
    axes[0, 0].set_ylabel('Frequency')
    
    # Confidence score distribution
    if confidence_scores:
        axes[0, 1].hist(confidence_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution of Confidence Scores')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Box count over images
    axes[1, 0].plot(range(1, len(box_counts) + 1), box_counts, marker='o', linewidth=1, markersize=3)
    axes[1, 0].set_title('Box Count per Image')
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Number of Boxes')
    
    # Summary statistics
    summary = create_detection_summary(results_list)
    stats_text = f"""
    Total Images: {summary['total_images']}
    Total Boxes: {summary['total_boxes']}
    Avg Boxes/Image: {summary['avg_boxes_per_image']:.2f}
    Avg Confidence: {summary['avg_confidence']:.3f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat'))
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def export_results_to_csv(results_list: List[Dict], output_path: str):
    """
    Export detection results to CSV file
    
    Args:
        results_list: List of detection results
        output_path: Output CSV file path
    """
    try:
        import pandas as pd
        
        # Prepare data for CSV
        data = []
        for result in results_list:
            image_name = result.get('image_name', 'unknown')
            count = result['count']
            avg_confidence = np.mean(result['confidences']) if result['confidences'] else 0
            
            data.append({
                'image_name': image_name,
                'box_count': count,
                'avg_confidence': avg_confidence,
                'max_confidence': max(result['confidences']) if result['confidences'] else 0,
                'min_confidence': min(result['confidences']) if result['confidences'] else 0
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Results exported to {output_path}")
        
    except ImportError:
        logging.warning("Pandas not available, cannot export to CSV")
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")

def create_detection_report(results_list: List[Dict], output_dir: str):
    """
    Create a comprehensive detection report
    
    Args:
        results_list: List of detection results
        output_dir: Output directory for report files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary
    summary = create_detection_summary(results_list)
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create and save plots
    fig = plot_detection_statistics(results_list, os.path.join(output_dir, 'statistics.png'))
    if fig:
        plt.close(fig)
    
    # Export to CSV if possible
    export_results_to_csv(results_list, os.path.join(output_dir, 'results.csv'))
    
    # Create markdown report
    report_content = f"""# Box Detection Report

## Summary Statistics

- **Total Images Processed**: {summary['total_images']}
- **Total Boxes Detected**: {summary['total_boxes']}
- **Average Boxes per Image**: {summary['avg_boxes_per_image']:.2f}
- **Min Boxes per Image**: {summary['min_boxes_per_image']}
- **Max Boxes per Image**: {summary['max_boxes_per_image']}
- **Average Confidence**: {summary['avg_confidence']:.3f}
- **Confidence Range**: {summary['min_confidence']:.3f} - {summary['max_confidence']:.3f}

## Files Generated

- `summary.json`: Detailed summary statistics
- `results.csv`: Per-image detection results
- `statistics.png`: Visualization of detection statistics

## Individual Results

"""
    
    for result in results_list[:10]:  # Show first 10 results
        image_name = result.get('image_name', 'unknown')
        report_content += f"- **{image_name}**: {result['count']} boxes detected\n"
    
    if len(results_list) > 10:
        report_content += f"... and {len(results_list) - 10} more images\n"
    
    with open(os.path.join(output_dir, 'report.md'), 'w') as f:
        f.write(report_content)
    
    logging.info(f"Detection report created in {output_dir}")

def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(image_path: str) -> Dict:
    """
    Get basic information about an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
    except Exception as e:
        logging.error(f"Error getting image info for {image_path}: {e}")
        return {}

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )