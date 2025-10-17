"""
YOLO Box Detection Engine

A comprehensive box detection and counting system using YOLO v8 and Roboflow.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

class BoxDetector:
    """
    Main class for box detection using YOLO v8
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the box detector
        
        Args:
            model_path: Path to the YOLO model file
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.logger.info(f"Loaded custom model from {self.model_path}")
            else:
                self.model = YOLO('yolov8n.pt')  # Load default model
                self.logger.info("Loaded default YOLOv8n model")
                
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def detect_boxes(self, image: np.ndarray, return_crops: bool = False) -> Dict:
        """
        Detect boxes in an image
        
        Args:
            image: Input image as numpy array
            return_crops: Whether to return cropped box images
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence, iou=self.iou_threshold)
            
            # Extract results
            boxes = []
            confidences = []
            crops = []
            
            for result in results:
                if result.boxes is not None:
                    # Get boxes in xyxy format
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, confidence) in enumerate(zip(xyxy, conf)):
                        x1, y1, x2, y2 = map(int, box)
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        
                        if return_crops:
                            crop = image[y1:y2, x1:x2]
                            crops.append(crop)
            
            return {
                'boxes': boxes,
                'confidences': confidences,
                'count': len(boxes),
                'crops': crops if return_crops else None,
                'image_shape': image.shape
            }
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return {
                'boxes': [],
                'confidences': [],
                'count': 0,
                'crops': None,
                'image_shape': image.shape
            }
    
    def visualize_detections(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """
        Draw bounding boxes on the image
        
        Args:
            image: Original image
            detection_results: Results from detect_boxes()
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        boxes = detection_results['boxes']
        confidences = detection_results['confidences']
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Box {i+1}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add count text
        count_text = f"Total Boxes: {detection_results['count']}"
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def process_image_file(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process a single image file
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save results (optional)
            
        Returns:
            Detection results dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run detection
        results = self.detect_boxes(image)
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save visualized image
            vis_image = self.visualize_detections(image, results)
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
            cv2.imwrite(output_path, vis_image)
            
            # Save detection data
            import json
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
            with open(json_path, 'w') as f:
                json.dump({
                    'image_path': image_path,
                    'count': results['count'],
                    'boxes': results['boxes'],
                    'confidences': results['confidences']
                }, f, indent=2)
        
        return results
    
    def batch_process(self, image_dir: str, output_dir: str) -> List[Dict]:
        """
        Process multiple images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results
            
        Returns:
            List of detection results for each image
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image_file(str(image_path), output_dir)
                result['image_name'] = image_path.name
                results.append(result)
                self.logger.info(f"Processed {image_path.name}: {result['count']} boxes detected")
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
        
        return results
    
    def train_custom_model(self, dataset_path: str, epochs: int = 100, batch_size: int = 16):
        """
        Train a custom YOLO model
        
        Args:
            dataset_path: Path to the dataset directory
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        try:
            # Initialize model for training
            model = YOLO('yolov8n.pt')  # Start with pretrained model
            
            # Train the model
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device=self.device
            )
            
            self.logger.info(f"Training completed. Best model saved to: {results.save_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise