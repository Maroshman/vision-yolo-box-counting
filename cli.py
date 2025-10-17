#!/usr/bin/env python3
"""
Command Line Interface for YOLO Box Counting Engine

Provides CLI access to box detection and counting functionality.
"""

import argparse
import sys
import os
from pathlib import Path
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.box_detector import BoxDetector
from src.rf_api_detector import RoboflowAPIDetector
from src.utils import load_config, create_detection_report

def main():
    parser = argparse.ArgumentParser(description='YOLO Box Counting Engine CLI')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect boxes in images')
    detect_parser.add_argument('input', help='Input image path or directory')
    detect_parser.add_argument('-o', '--output', help='Output directory', default='results')
    detect_parser.add_argument('-b', '--backend', choices=['roboflow', 'local'], default='roboflow', help='Detection backend to use')
    detect_parser.add_argument('-m', '--model', help='Model path (when using local backend)', default='yolov8n.pt')
    detect_parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold')
    detect_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    detect_parser.add_argument('--rf-endpoint', default=os.getenv('ROBOFLOW_API_ENDPOINT', 'https://detect.roboflow.com/shoeboxes-rwv5h/2'), help='Roboflow API endpoint URL')
    detect_parser.add_argument('--rf-api-key', default=os.getenv('ROBOFLOW_API_KEY', ''), help='Roboflow API key')
    detect_parser.add_argument('--save-crops', action='store_true', help='Save detected box crops')
    detect_parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    # Count command
    count_parser = subparsers.add_parser('count', help='Count boxes in images')
    count_parser.add_argument('input', help='Input image path or directory')
    count_parser.add_argument('-b', '--backend', choices=['roboflow', 'local'], default='roboflow', help='Detection backend to use')
    count_parser.add_argument('-m', '--model', help='Model path (when using local backend)', default='yolov8n.pt')
    count_parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence threshold')
    count_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    count_parser.add_argument('--rf-endpoint', default=os.getenv('ROBOFLOW_API_ENDPOINT', 'https://detect.roboflow.com/shoeboxes-rwv5h/2'), help='Roboflow API endpoint URL')
    count_parser.add_argument('--rf-api-key', default=os.getenv('ROBOFLOW_API_KEY', ''), help='Roboflow API key')
    count_parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json', help='Output format')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('data', help='Dataset YAML file path')
    train_parser.add_argument('-m', '--model', help='Base model', default='yolov8n.pt')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--device', help='Training device (auto, cpu, 0, 1, etc.)', default='auto')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('model', help='Model path')
    eval_parser.add_argument('data', help='Dataset YAML file path')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        detect_boxes(args)
    elif args.command == 'count':
        count_boxes(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()

def detect_boxes(args):
    """Detect boxes in images"""
    print(f"🔍 Detecting boxes in: {args.input}")
    
    # Initialize detector
    detector = (
        RoboflowAPIDetector(api_url=args.rf_endpoint, api_key=args.rf_api_key, confidence=args.confidence, overlap=args.iou)
        if args.backend == 'roboflow'
        else BoxDetector(args.model, args.confidence, args.iou)
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        result = detector.process_image_file(args.input, args.output if not args.no_viz else None)
        print(f"📦 Detected {result['count']} boxes")
        
        # Save crops if requested
        if args.save_crops and result['count'] > 0:
            import cv2
            image = cv2.imread(args.input)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            crops_result = detector.detect_boxes(image_rgb, return_crops=True)
            
            crops_dir = os.path.join(args.output, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            
            for i, crop in enumerate(crops_result['crops']):
                crop_path = os.path.join(crops_dir, f"box_{i+1}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            print(f"💾 Saved {len(crops_result['crops'])} box crops to {crops_dir}")
        
    elif os.path.isdir(args.input):
        # Directory of images
        results = detector.batch_process(args.input, args.output)
        total_boxes = sum(r['count'] for r in results)
        print(f"📊 Processed {len(results)} images, detected {total_boxes} total boxes")
        
        # Create report
        create_detection_report(results, args.output)
        print(f"📄 Detection report saved to {args.output}")
    else:
        print(f"❌ Input not found: {args.input}")

def count_boxes(args):
    """Count boxes in images and output results"""
    print(f"🔢 Counting boxes in: {args.input}")
    
    # Initialize detector
    detector = (
        RoboflowAPIDetector(api_url=args.rf_endpoint, api_key=args.rf_api_key, confidence=args.confidence, overlap=args.iou)
        if args.backend == 'roboflow'
        else BoxDetector(args.model, args.confidence, args.iou)
    )
    
    results = []
    
    if os.path.isfile(args.input):
        # Single image
        import cv2
        image = cv2.imread(args.input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect_boxes(image)
        count = result['count'] if isinstance(result, dict) else int(result)
        results.append({
            'image': args.input,
            'count': count
        })
    elif os.path.isdir(args.input):
        # Directory of images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for file_path in Path(args.input).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                try:
                    import cv2
                    image = cv2.imread(str(file_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = detector.detect_boxes(image)
                    count = result['count'] if isinstance(result, dict) else int(result)
                    results.append({
                        'image': str(file_path),
                        'count': count
                    })
                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {e}")
    
    # Output results
    if args.format == 'json':
        print(json.dumps(results, indent=2))
    elif args.format == 'csv':
        print("image,count")
        for result in results:
            print(f"{result['image']},{result['count']}")
    elif args.format == 'txt':
        for result in results:
            print(f"{result['image']}: {result['count']} boxes")
    
    # Summary
    total_images = len(results)
    total_boxes = sum(r['count'] for r in results)
    avg_boxes = total_boxes / total_images if total_images > 0 else 0
    
    print(f"\n📊 Summary: {total_images} images, {total_boxes} total boxes, {avg_boxes:.1f} avg boxes/image", file=sys.stderr)

def train_model(args):
    """Train a new YOLO model"""
    print(f"🏋️ Training model with data: {args.data}")
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        imgsz=640
    )
    
    print(f"✅ Training completed! Best model saved to: {results.save_dir}/weights/best.pt")

def evaluate_model(args):
    """Evaluate model performance"""
    print(f"📊 Evaluating model: {args.model}")
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(args.model)
    
    # Validate
    results = model.val(data=args.data)
    
    # Print metrics
    metrics = results.results_dict
    print(f"📈 Evaluation Results:")
    if 'metrics/mAP50(B)' in metrics:
        print(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in metrics:
        print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
    if 'metrics/precision(B)' in metrics:
        print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
    if 'metrics/recall(B)' in metrics:
        print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")

if __name__ == '__main__':
    main()