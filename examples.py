"""
Example scripts for YOLO Box Counting Engine

This module provides example usage scripts for common tasks.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.box_detector import BoxDetector
from src.roboflow_client import RoboflowClient
from src.utils import load_config, create_detection_report

def example_single_image_detection():
    """Example: Detect boxes in a single image"""
    print("🔍 Example: Single Image Detection")
    
    # Initialize detector
    detector = BoxDetector("yolov8n.pt", confidence=0.5)
    
    # Example image path (replace with your image)
    image_path = "data/images/sample.jpg"
    
    if os.path.exists(image_path):
        # Detect boxes
        results = detector.detect_boxes(cv2.imread(image_path), return_crops=True)
        
        print(f"📦 Detected {results['count']} boxes")
        print(f"📊 Confidence scores: {results['confidences']}")
        
        # Visualize results
        import cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image = detector.visualize_detections(image_rgb, results)
        
        # Save result
        cv2.imwrite("results/detected_example.jpg", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print("💾 Results saved to results/detected_example.jpg")
    else:
        print(f"❌ Image not found: {image_path}")
        print("💡 Add a sample image to data/images/sample.jpg to run this example")

def example_batch_processing():
    """Example: Process multiple images"""
    print("📁 Example: Batch Processing")
    
    # Initialize detector
    detector = BoxDetector("yolov8n.pt", confidence=0.5)
    
    # Process all images in a directory
    input_dir = "data/images"
    output_dir = "results/batch_processing"
    
    if os.path.exists(input_dir):
        results = detector.batch_process(input_dir, output_dir)
        
        total_boxes = sum(r['count'] for r in results)
        print(f"📊 Processed {len(results)} images")
        print(f"📦 Total boxes detected: {total_boxes}")
        print(f"📈 Average boxes per image: {total_boxes/len(results):.2f}")
        
        # Create detailed report
        create_detection_report(results, output_dir)
        print(f"📄 Detailed report saved to {output_dir}")
    else:
        print(f"❌ Directory not found: {input_dir}")
        print("💡 Add images to data/images/ to run this example")

def example_roboflow_integration():
    """Example: Roboflow integration"""
    print("🌐 Example: Roboflow Integration")
    
    # Check if API key is available
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("❌ ROBOFLOW_API_KEY not found in environment variables")
        print("💡 Set your Roboflow API key to run this example")
        return
    
    # Initialize Roboflow client
    rf_client = RoboflowClient(api_key)
    
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "your-workspace")
    project = os.getenv("ROBOFLOW_PROJECT", "box-detection")
    
    try:
        # Get project info
        info = rf_client.get_project_info(workspace, project)
        print(f"📊 Project Info: {info}")
        
        # Download dataset
        dataset_path = rf_client.download_dataset(workspace, project, 1, "yolov8", "./datasets")
        print(f"📥 Dataset downloaded to: {dataset_path}")
        
        # Prepare for training
        prepared_path = rf_client.prepare_yolo_dataset(dataset_path, "./datasets/prepared")
        print(f"🔧 Dataset prepared for training: {prepared_path}")
        
    except Exception as e:
        print(f"❌ Error with Roboflow integration: {e}")
        print("💡 Check your workspace and project names")

def example_model_training():
    """Example: Train a custom model"""
    print("🏋️ Example: Model Training")
    
    # Check if dataset is available
    dataset_yaml = "datasets/prepared/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset not found: {dataset_yaml}")
        print("💡 Download a dataset using Roboflow integration first")
        return
    
    # Initialize detector for training
    detector = BoxDetector("yolov8n.pt")
    
    try:
        # Train model
        print("🚀 Starting training...")
        results = detector.train_custom_model(
            dataset_path=dataset_yaml,
            epochs=10,  # Small number for example
            batch_size=16
        )
        
        print(f"✅ Training completed!")
        print(f"📁 Results saved to: {results.save_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

def example_performance_evaluation():
    """Example: Evaluate model performance"""
    print("📊 Example: Performance Evaluation")
    
    # Check if trained model exists
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"  # Fall back to pretrained
        print("⚠️ Using pretrained model for evaluation")
    
    dataset_yaml = "datasets/prepared/data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset not found: {dataset_yaml}")
        print("💡 Prepare a dataset first")
        return
    
    # Initialize detector
    detector = BoxDetector(model_path)
    
    try:
        # Run validation
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.val(data=dataset_yaml)
        
        # Print metrics
        metrics = results.results_dict
        print("📈 Evaluation Metrics:")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/precision(B)' in metrics:
            print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def example_real_time_detection():
    """Example: Real-time detection setup"""
    print("📹 Example: Real-time Detection")
    
    # Initialize detector
    detector = BoxDetector("yolov8n.pt", confidence=0.5)
    
    print("💡 Real-time detection can be used for:")
    print("  - Webcam feed processing")
    print("  - Video file processing")
    print("  - Live stream analysis")
    
    print("\n🔧 Example code for webcam detection:")
    print("""
import cv2
from src.box_detector import BoxDetector

detector = BoxDetector("yolov8n.pt")

cap = cv2.VideoCapture(0)  # Use webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect_boxes(frame, return_details=True)
    vis_frame = detector.visualize_detections(frame, results)
    
    cv2.imshow('Box Detection', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    """)

def run_all_examples():
    """Run all examples"""
    print("🚀 Running All Examples")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    
    examples = [
        ("Single Image Detection", example_single_image_detection),
        ("Batch Processing", example_batch_processing),
        ("Roboflow Integration", example_roboflow_integration),
        ("Model Training", example_model_training),
        ("Performance Evaluation", example_performance_evaluation),
        ("Real-time Detection", example_real_time_detection),
    ]
    
    for name, func in examples:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
        print("\n")
    
    print("✅ All examples completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Box Counting Examples")
    parser.add_argument("--example", choices=[
        "single", "batch", "roboflow", "train", "eval", "realtime", "all"
    ], default="all", help="Which example to run")
    
    args = parser.parse_args()
    
    examples_map = {
        "single": example_single_image_detection,
        "batch": example_batch_processing,
        "roboflow": example_roboflow_integration,
        "train": example_model_training,
        "eval": example_performance_evaluation,
        "realtime": example_real_time_detection,
        "all": run_all_examples
    }
    
    examples_map[args.example]()