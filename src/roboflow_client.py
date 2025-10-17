"""
Roboflow Client for Dataset Management

Handles integration with Roboflow for dataset download, upload, and management.
"""

import os
import requests
from roboflow import Roboflow
from typing import Optional, Dict, List
import logging
from pathlib import Path
import shutil

class RoboflowClient:
    """
    Client for interacting with Roboflow API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Roboflow client
        
        Args:
            api_key: Roboflow API key
        """
        self.api_key = api_key
        self.rf = Roboflow(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
    def get_project(self, workspace: str, project: str):
        """
        Get a Roboflow project
        
        Args:
            workspace: Workspace name
            project: Project name
            
        Returns:
            Roboflow project object
        """
        try:
            workspace_obj = self.rf.workspace(workspace)
            project_obj = workspace_obj.project(project)
            return project_obj
        except Exception as e:
            self.logger.error(f"Error accessing project {workspace}/{project}: {e}")
            raise
    
    def download_dataset(self, workspace: str, project: str, version: int, 
                        format: str = "yolov8", location: str = "./data") -> str:
        """
        Download dataset from Roboflow
        
        Args:
            workspace: Workspace name
            project: Project name
            version: Dataset version
            format: Dataset format (yolov8, coco, etc.)
            location: Download location
            
        Returns:
            Path to downloaded dataset
        """
        try:
            project = self.get_project(workspace, project)
            version_obj = project.version(version)
            
            # Download dataset
            dataset = version_obj.download(format, location=location)
            
            self.logger.info(f"Dataset downloaded to: {dataset.location}")
            return dataset.location
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            raise
    
    def upload_image(self, workspace: str, project: str, image_path: str, 
                    split: str = "train") -> Dict:
        """
        Upload an image to Roboflow project
        
        Args:
            workspace: Workspace name
            project: Project name
            image_path: Path to image file
            split: Dataset split (train, valid, test)
            
        Returns:
            Upload response
        """
        try:
            project_obj = self.get_project(workspace, project)
            
            # Upload image
            response = project_obj.upload(image_path, split=split)
            
            self.logger.info(f"Image uploaded: {image_path}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error uploading image {image_path}: {e}")
            raise
    
    def batch_upload_images(self, workspace: str, project: str, 
                           images_dir: str, split: str = "train") -> List[Dict]:
        """
        Upload multiple images to Roboflow
        
        Args:
            workspace: Workspace name
            project: Project name
            images_dir: Directory containing images
            split: Dataset split
            
        Returns:
            List of upload responses
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(images_dir).glob(f"*{ext}"))
            image_paths.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        responses = []
        for image_path in image_paths:
            try:
                response = self.upload_image(workspace, project, str(image_path), split)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to upload {image_path}: {e}")
        
        return responses
    
    def create_project(self, workspace: str, project_name: str, 
                      project_type: str = "object-detection") -> Dict:
        """
        Create a new Roboflow project
        
        Args:
            workspace: Workspace name
            project_name: New project name
            project_type: Type of project
            
        Returns:
            Project creation response
        """
        try:
            workspace_obj = self.rf.workspace(workspace)
            project = workspace_obj.create_project(
                project_name=project_name,
                project_type=project_type
            )
            
            self.logger.info(f"Project created: {project_name}")
            return project
            
        except Exception as e:
            self.logger.error(f"Error creating project {project_name}: {e}")
            raise
    
    def get_project_info(self, workspace: str, project: str) -> Dict:
        """
        Get project information
        
        Args:
            workspace: Workspace name
            project: Project name
            
        Returns:
            Project information dictionary
        """
        try:
            project_obj = self.get_project(workspace, project)
            
            # Get project details
            info = {
                'name': project_obj.name,
                'id': project_obj.id,
                'workspace': workspace,
                'classes': getattr(project_obj, 'classes', []),
                'images_count': getattr(project_obj, 'images', 0),
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting project info: {e}")
            return {}
    
    def prepare_yolo_dataset(self, dataset_path: str, output_path: str) -> str:
        """
        Prepare downloaded dataset for YOLO training
        
        Args:
            dataset_path: Path to downloaded dataset
            output_path: Path for prepared dataset
            
        Returns:
            Path to data.yaml file
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Copy dataset structure
            dataset_dir = Path(dataset_path)
            output_dir = Path(output_path)
            
            # Copy train, valid, test directories
            for split in ['train', 'valid', 'test']:
                split_dir = dataset_dir / split
                if split_dir.exists():
                    shutil.copytree(split_dir, output_dir / split, dirs_exist_ok=True)
            
            # Copy data.yaml if it exists
            data_yaml = dataset_dir / 'data.yaml'
            if data_yaml.exists():
                shutil.copy2(data_yaml, output_dir / 'data.yaml')
                return str(output_dir / 'data.yaml')
            
            # Create data.yaml if it doesn't exist
            data_yaml_content = f"""
path: {output_path}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['box']
"""
            
            data_yaml_path = output_dir / 'data.yaml'
            with open(data_yaml_path, 'w') as f:
                f.write(data_yaml_content.strip())
            
            self.logger.info(f"Dataset prepared at: {output_path}")
            return str(data_yaml_path)
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise
    
    def inference_api(self, workspace: str, project: str, version: int, 
                     image_path: str, confidence: float = 0.5) -> Dict:
        """
        Use Roboflow inference API for predictions
        
        Args:
            workspace: Workspace name
            project: Project name
            version: Model version
            image_path: Path to image
            confidence: Confidence threshold
            
        Returns:
            Inference results
        """
        try:
            project_obj = self.get_project(workspace, project)
            model = project_obj.version(version).model
            
            # Run inference
            prediction = model.predict(image_path, confidence=confidence)
            
            return prediction.json()
            
        except Exception as e:
            self.logger.error(f"Error in inference API: {e}")
            return {}
    
    def get_public_models(self, search_term: str = "box") -> List[Dict]:
        """
        Search for public models on Roboflow Universe
        
        Args:
            search_term: Search term for models
            
        Returns:
            List of public models
        """
        try:
            # This would typically use the Roboflow API to search public models
            # For now, return some popular box detection models
            popular_models = [
                {
                    'name': 'Box Detection Model',
                    'workspace': 'roboflow-universe',
                    'project': 'box-detection',
                    'version': 1,
                    'description': 'General purpose box detection model',
                    'classes': ['box', 'package', 'container']
                },
                {
                    'name': 'Package Detection',
                    'workspace': 'shipping-detection',
                    'project': 'package-detection',
                    'version': 2,
                    'description': 'Shipping package detection model',
                    'classes': ['package', 'envelope', 'box']
                },
                {
                    'name': 'Container Detection',
                    'workspace': 'warehouse-ai',
                    'project': 'container-detection',
                    'version': 1,
                    'description': 'Warehouse container detection',
                    'classes': ['container', 'crate', 'bin']
                }
            ]
            
            return [model for model in popular_models if search_term.lower() in model['name'].lower()]
            
        except Exception as e:
            self.logger.error(f"Error getting public models: {e}")
            return []
    
    def download_public_dataset(self, dataset_url: str, output_dir: str = "./datasets") -> str:
        """
        Download a public dataset from Roboflow Universe
        
        Args:
            dataset_url: URL to the public dataset
            output_dir: Output directory
            
        Returns:
            Path to downloaded dataset
        """
        try:
            # Extract workspace and project from URL
            # Example URL: https://universe.roboflow.com/workspace/project/version
            url_parts = dataset_url.split('/')
            workspace = url_parts[-3]
            project = url_parts[-2]
            version = int(url_parts[-1]) if url_parts[-1].isdigit() else 1
            
            return self.download_dataset(workspace, project, version, "yolov8", output_dir)
            
        except Exception as e:
            self.logger.error(f"Error downloading public dataset: {e}")
            return ""
    
    def export_dataset(self, workspace: str, project: str, version: int,
                      format: str = "yolov8", output_dir: str = "./exports") -> str:
        """
        Export dataset in specified format
        
        Args:
            workspace: Workspace name
            project: Project name
            version: Dataset version
            format: Export format
            output_dir: Output directory
            
        Returns:
            Path to exported dataset
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Download and prepare dataset
            dataset_path = self.download_dataset(workspace, project, version, format, output_dir)
            
            if format == "yolov8":
                prepared_path = self.prepare_yolo_dataset(dataset_path, f"{output_dir}/prepared")
                return prepared_path
            
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {e}")
            raise