#!/usr/bin/env python3
"""
Dataset Finder for YOLO Box Counting Engine

Helps you find and download datasets for box detection training.
"""

import requests
import json
import os
from pathlib import Path

class DatasetFinder:
    """Find and download datasets for box detection"""
    
    def __init__(self):
        self.roboflow_universe_url = "https://universe.roboflow.com"
        
    def list_popular_datasets(self):
        """List popular box detection datasets"""
        datasets = [
            {
                "name": "Amazon Bin Image Dataset",
                "source": "Kaggle",
                "url": "https://www.kaggle.com/datasets/google-cloud/amazon-bin-image-dataset",
                "description": "Images of bins containing objects, good for box/container detection",
                "size": "500MB",
                "images": "500+",
                "free": True
            },
            {
                "name": "Package Detection Dataset",
                "source": "Roboflow Universe",
                "url": "https://universe.roboflow.com/search?q=package+detection",
                "description": "Various package and box detection datasets",
                "size": "Varies",
                "images": "1000+",
                "free": True
            },
            {
                "name": "COCO Dataset (Partial)",
                "source": "Microsoft",
                "url": "https://cocodataset.org/",
                "description": "Large dataset with various objects including boxes",
                "size": "20GB+",
                "images": "200,000+",
                "free": True
            },
            {
                "name": "Open Images Dataset",
                "source": "Google",
                "url": "https://storage.googleapis.com/openimages/web/index.html",
                "description": "Massive dataset with box/container annotations",
                "size": "500GB+",
                "images": "9M+",
                "free": True
            },
            {
                "name": "Warehouse Object Detection",
                "source": "Roboflow Universe",
                "url": "https://universe.roboflow.com/search?q=warehouse",
                "description": "Warehouse scenes with boxes and containers",
                "size": "100-500MB",
                "images": "500-2000",
                "free": True
            }
        ]
        
        return datasets
    
    def list_roboflow_models(self):
        """List available Roboflow pre-trained models"""
        models = [
            {
                "name": "General Object Detection",
                "type": "Pre-trained YOLO",
                "classes": ["box", "package", "container", "person", "car", "etc."],
                "accuracy": "High",
                "speed": "Fast",
                "cost": "Free (with limits)"
            },
            {
                "name": "Custom Box Detection",
                "type": "Trainable",
                "classes": ["Custom box types"],
                "accuracy": "Very High",
                "speed": "Fast",
                "cost": "Free tier available"
            },
            {
                "name": "Roboflow Inference API",
                "type": "API-based",
                "classes": ["100+ object types including boxes"],
                "accuracy": "High",
                "speed": "Very Fast",
                "cost": "Free tier: 1000 images/month"
            }
        ]
        
        return models
    
    def print_datasets(self):
        """Print formatted dataset information"""
        datasets = self.list_popular_datasets()
        
        print("🗂️  POPULAR BOX DETECTION DATASETS")
        print("=" * 60)
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   Source: {dataset['source']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Images: {dataset['images']}")
            print(f"   Free: {'✅' if dataset['free'] else '❌'}")
            print(f"   URL: {dataset['url']}")
    
    def print_models(self):
        """Print formatted model information"""
        models = self.list_roboflow_models()
        
        print("\n🤖 ROBOFLOW MODEL OPTIONS")
        print("=" * 60)
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Classes: {', '.join(model['classes'][:3])}{'...' if len(model['classes']) > 3 else ''}")
            print(f"   Accuracy: {model['accuracy']}")
            print(f"   Speed: {model['speed']}")
            print(f"   Cost: {model['cost']}")
    
    def get_quick_start_guide(self):
        """Get quick start guide for datasets"""
        guide = """
🚀 QUICK START GUIDE

1. FOR IMMEDIATE USE (No training needed):
   • Use the current YOLO model - it already detects many box types!
   • Just upload images to the web app at http://localhost:8501

2. FOR CUSTOM TRAINING:
   
   Option A - Roboflow (Recommended):
   • Sign up at https://roboflow.com (free tier available)
   • Create a project and upload ~100+ images
   • Annotate boxes in their web interface
   • Download dataset in YOLO format
   • Use our training notebook

   Option B - Manual Dataset:
   • Download from Kaggle or other sources
   • Use LabelImg to annotate (free tool)
   • Convert to YOLO format
   • Train with our notebook

3. USING ROBOFLOW PRE-TRAINED MODELS:
   • Browse universe.roboflow.com
   • Find box detection models
   • Use via API (1000 free predictions/month)
   • Integrate with our app

💡 RECOMMENDATION:
Start with the existing YOLO model in our app. If you need better accuracy
for specific box types, then consider custom training with Roboflow.
"""
        return guide

def main():
    finder = DatasetFinder()
    
    print("📦 YOLO Box Counting - Dataset & Model Finder")
    print("=" * 60)
    
    # Print datasets
    finder.print_datasets()
    
    # Print models
    finder.print_models()
    
    # Print quick start guide
    print(finder.get_quick_start_guide())
    
    print("\n🎯 NEXT STEPS:")
    print("1. Try the current app first: http://localhost:8501")
    print("2. If you need custom training, choose a dataset above")
    print("3. For help, check the training notebook in notebooks/")

if __name__ == "__main__":
    main()