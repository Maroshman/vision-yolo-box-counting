"""
Streamlit Web Application for YOLO Box Counting Engine

A user-friendly interface for box detection and counting using YOLO v8.
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
import yaml
from dotenv import load_dotenv
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.box_detector import BoxDetector
from src.rf_api_detector import RoboflowAPIDetector
from src.roboflow_client import RoboflowClient
from src.utils import (
    load_config, save_image, resize_image, create_detection_summary,
    plot_detection_statistics, create_detection_report, validate_image_file
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YOLO Box Counting Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.success-message {
    color: #28a745;
    font-weight: bold;
}
.error-message {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector(backend, model_path, confidence, iou_threshold, rf_api_key, rf_api_endpoint):
    """Load detector (local YOLO or Roboflow API) with caching"""
    if backend == "Roboflow API":
        return RoboflowAPIDetector(api_url=rf_api_endpoint, api_key=rf_api_key, confidence=confidence, overlap=iou_threshold)
    # Default to local
    return BoxDetector(model_path, confidence, iou_threshold)

@st.cache_data
def load_app_config():
    """Load application configuration"""
    try:
        return load_config("config.yaml")
    except:
        return {
            'model': {'confidence': 0.5, 'iou_threshold': 0.45},
            'app': {'title': 'YOLO Box Counting Engine', 'max_file_size': 200}
        }

def main():
    # Load configuration
    config = load_app_config()
    
    # Main header
    st.markdown('<h1 class="main-header">📦 YOLO Box Counting Engine</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        backend = st.selectbox(
            "Detection Backend",
            options=["Roboflow API", "Local YOLO"],
            index=0 if os.getenv("DETECTION_BACKEND", "roboflow").lower() == "roboflow" else 1,
            help="Choose between cloud-hosted Roboflow model or local YOLO weights"
        )
        
        model_path = None
        if backend == "Local YOLO":
            model_path = st.text_input(
                "Model Path",
                value=os.getenv("MODEL_PATH", "yolov8n.pt"),
                help="Path to YOLO model file (local path or model name)"
            )
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(os.getenv("IOU_THRESHOLD", 0.45)),
            step=0.05,
            help="IoU threshold for non-maximum suppression"
        )

        rf_api_key = None
        rf_api_endpoint = None
        if backend == "Roboflow API":
            rf_api_key = st.text_input(
                "Roboflow API Key",
                type="password",
                value=os.getenv("ROBOFLOW_API_KEY", ""),
                help="Your Roboflow API key"
            )
            rf_api_endpoint = st.text_input(
                "Roboflow API Endpoint",
                value=os.getenv("ROBOFLOW_API_ENDPOINT", "https://detect.roboflow.com/shoeboxes-rwv5h/2"),
                help="Full Roboflow Hosted Inference endpoint URL"
            )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_image_size = st.number_input(
                "Max Image Size",
                min_value=480,
                max_value=2048,
                value=1280,
                step=32,
                help="Maximum image dimension for processing"
            )
            
            save_results = st.checkbox(
                "Save Results",
                value=True,
                help="Save detection results to files"
            )
            
            show_crops = st.checkbox(
                "Show Detected Boxes",
                value=False,
                help="Display cropped detected boxes"
            )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Image", "📁 Batch Processing", "🎯 Training", "📊 Analytics"])
    
    with tab1:
        st.header("Single Image Detection")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to detect and count boxes"
        )
        
        col1, col2 = st.columns([2, 1])
        
        if uploaded_file is not None:
            try:
                # Load detector
                with st.spinner("Loading model..."):
                    detector = load_detector(backend, model_path or "yolov8n.pt", confidence, iou_threshold, rf_api_key or "", rf_api_endpoint or "")
                
                # Read image
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if max_image_size:
                    image_rgb = resize_image(image_rgb, max_image_size)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image_rgb, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                # Run detection
                with st.spinner("Detecting boxes..."):
                    results = detector.detect_boxes(image_rgb, return_crops=show_crops)
                
                # Display results
                with col2:
                    st.subheader("Detection Results")
                    
                    # Display class counts if available
                    class_counts = results.get('class_counts', {})
                    if class_counts:
                        st.markdown("### 📊 Detections by Class")
                        for cls, count in class_counts.items():
                            icon = "📦" if cls.lower() == "box" else "🏷️" if cls.lower() == "label" else "🔷"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{icon} {cls.capitalize()}s: {count}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Total count metric
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Total Detections: {results['count']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if results['confidences']:
                        avg_conf = np.mean(results['confidences'])
                        st.markdown(f"""
                        <div class="metric-card">
                            <p><strong>Average Confidence:</strong> {avg_conf:.3f}</p>
                            <p><strong>Min Confidence:</strong> {min(results['confidences']):.3f}</p>
                            <p><strong>Max Confidence:</strong> {max(results['confidences']):.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualize detections
                if results['count'] > 0:
                    st.subheader("Detection Visualization")
                    vis_image = detector.visualize_detections(image_rgb, results)
                    st.image(vis_image, caption=f"Detected {results['count']} boxes", use_column_width=True)
                    
                    # Show individual detections
                    if show_crops and results['crops']:
                        st.subheader("Detected Boxes")
                        cols = st.columns(min(4, len(results['crops'])))
                        for i, crop in enumerate(results['crops'][:12]):  # Show max 12 crops
                            with cols[i % 4]:
                                st.image(crop, caption=f"Box {i+1}", use_column_width=True)
                    
                    # Download button for results
                    if save_results:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            save_image(vis_image, tmp_file.name)
                            with open(tmp_file.name, 'rb') as f:
                                st.download_button(
                                    "📥 Download Result Image",
                                    data=f.read(),
                                    file_name=f"detected_{uploaded_file.name}",
                                    mime="image/jpeg"
                                )
                
                else:
                    st.warning("No boxes detected in the image. Try adjusting the confidence threshold.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.header("Batch Processing")
        
        # Directory input or file upload
        option = st.radio("Input Method", ["Upload Multiple Files", "Process Directory"])
        
        if option == "Upload Multiple Files":
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files and st.button("🚀 Process Batch"):
                try:
                    detector = load_detector(backend, model_path or "yolov8n.pt", confidence, iou_threshold, rf_api_key or "", rf_api_endpoint or "")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_list = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process image
                        image_bytes = uploaded_file.read()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        if max_image_size:
                            image_rgb = resize_image(image_rgb, max_image_size)
                        
                        result = detector.detect_boxes(image_rgb)
                        result['image_name'] = uploaded_file.name
                        results_list.append(result)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Display summary
                    status_text.text("Processing complete!")
                    summary = create_detection_summary(results_list)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", summary['total_images'])
                    with col2:
                        st.metric("Total Boxes", summary['total_boxes'])
                    with col3:
                        st.metric("Avg Boxes/Image", f"{summary['avg_boxes_per_image']:.1f}")
                    
                    # Results table
                    st.subheader("Individual Results")
                    results_df = []
                    for result in results_list:
                        results_df.append({
                            'Image': result['image_name'],
                            'Boxes': result['count'],
                            'Avg Confidence': f"{np.mean(result['confidences']):.3f}" if result['confidences'] else "N/A"
                        })
                    
                    st.dataframe(results_df)
                    
                    # Visualization
                    if len(results_list) > 1:
                        st.subheader("Statistics")
                        fig = plot_detection_statistics(results_list)
                        if fig:
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error in batch processing: {str(e)}")
        
        else:  # Process Directory
            directory_path = st.text_input("Directory Path", help="Path to directory containing images")
            
            if directory_path and os.path.exists(directory_path) and st.button("🚀 Process Directory"):
                try:
                    detector = load_detector(backend, model_path or "yolov8n.pt", confidence, iou_threshold, rf_api_key or "", rf_api_endpoint or "")

                    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                    image_paths = [p for p in Path(directory_path).rglob('*') if p.suffix.lower() in image_extensions]

                    results_list = []
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for image_path in image_paths:
                            image = cv2.imread(str(image_path))
                            if image is None:
                                continue
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            if max_image_size:
                                image_rgb = resize_image(image_rgb, max_image_size)
                            result = detector.detect_boxes(image_rgb)
                            result['image_name'] = image_path.name

                            # Save visualization
                            vis = detector.visualize_detections(image_rgb, result)
                            out_path = os.path.join(tmp_dir, f"{image_path.stem}_detected.jpg")
                            save_image(vis, out_path)

                            results_list.append(result)
                        
                        if results_list:
                            summary = create_detection_summary(results_list)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Images", summary['total_images'])
                            with col2:
                                st.metric("Total Boxes", summary['total_boxes'])
                            with col3:
                                st.metric("Avg Boxes/Image", f"{summary['avg_boxes_per_image']:.1f}")
                            
                            st.success("✅ Batch processing completed!")
                        else:
                            st.warning("No images found in the directory.")
                
                except Exception as e:
                    st.error(f"Error processing directory: {str(e)}")
    
    with tab3:
        st.header("Model Training")
        st.info("🚧 Training functionality - Connect your Roboflow dataset")
        
        # Roboflow integration
        st.subheader("Roboflow Integration")
        
        roboflow_api_key = st.text_input(
            "Roboflow API Key",
            type="password",
            value=os.getenv("ROBOFLOW_API_KEY", ""),
            help="Your Roboflow API key"
        )
        
        workspace = st.text_input(
            "Workspace Name",
            value=os.getenv("ROBOFLOW_WORKSPACE", ""),
            help="Your Roboflow workspace name"
        )
        
        project = st.text_input(
            "Project Name",
            value=os.getenv("ROBOFLOW_PROJECT", "box-detection"),
            help="Your Roboflow project name"
        )
        
        if roboflow_api_key and workspace and project:
            try:
                rf_client = RoboflowClient(roboflow_api_key)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Download Dataset"):
                        with st.spinner("Downloading dataset..."):
                            version = st.number_input("Dataset Version", min_value=1, value=1)
                            dataset_path = rf_client.download_dataset(workspace, project, version)
                            st.success(f"✅ Dataset downloaded to: {dataset_path}")
                
                with col2:
                    if st.button("ℹ️ Get Project Info"):
                        with st.spinner("Fetching project info..."):
                            info = rf_client.get_project_info(workspace, project)
                            if info:
                                st.json(info)
                
                # Training parameters
                st.subheader("Training Parameters")
                
                col1, col2 = st.columns(2)
                with col1:
                    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
                
                with col2:
                    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
                    patience = st.number_input("Patience", min_value=1, max_value=100, value=50)
                
                if st.button("🏋️ Start Training"):
                    st.warning("Training will start in a separate process. This may take several hours depending on your dataset size and hardware.")
                    st.info("💡 Tip: Use GPU acceleration for faster training")
                    
            except Exception as e:
                st.error(f"Error with Roboflow integration: {str(e)}")
    
    with tab4:
        st.header("Analytics & Reports")
        
        # Upload results for analysis
        results_file = st.file_uploader(
            "Upload Results JSON",
            type=['json'],
            help="Upload a JSON file containing detection results for analysis"
        )
        
        if results_file:
            try:
                import json
                results_data = json.load(results_file)
                
                if isinstance(results_data, list):
                    results_list = results_data
                elif 'results' in results_data:
                    results_list = results_data['results']
                else:
                    st.error("Invalid results format")
                    results_list = []
                
                if results_list:
                    # Summary statistics
                    summary = create_detection_summary(results_list)
                    
                    st.subheader("📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Images", summary['total_images'])
                    with col2:
                        st.metric("Total Boxes", summary['total_boxes'])
                    with col3:
                        st.metric("Avg Boxes/Image", f"{summary['avg_boxes_per_image']:.1f}")
                    with col4:
                        st.metric("Avg Confidence", f"{summary['avg_confidence']:.3f}")
                    
                    # Detailed metrics
                    with st.expander("Detailed Metrics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Min Boxes/Image", summary['min_boxes_per_image'])
                            st.metric("Max Boxes/Image", summary['max_boxes_per_image'])
                        with col2:
                            st.metric("Min Confidence", f"{summary['min_confidence']:.3f}")
                            st.metric("Max Confidence", f"{summary['max_confidence']:.3f}")
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    fig = plot_detection_statistics(results_list)
                    if fig:
                        st.pyplot(fig)
                    
                    # Export options
                    st.subheader("📤 Export Options")
                    
                    if st.button("Generate Report"):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            create_detection_report(results_list, tmp_dir)
                            
                            # Create zip file with all report files
                            import zipfile
                            zip_path = os.path.join(tmp_dir, "detection_report.zip")
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for file in os.listdir(tmp_dir):
                                    if file != "detection_report.zip":
                                        zipf.write(os.path.join(tmp_dir, file), file)
                            
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    "📥 Download Complete Report",
                                    data=f.read(),
                                    file_name="detection_report.zip",
                                    mime="application/zip"
                                )
                
            except Exception as e:
                st.error(f"Error analyzing results: {str(e)}")
        
        else:
            st.info("Upload a results JSON file to view analytics and generate reports.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "YOLO Box Counting Engine | Built with ❤️ using Streamlit & Ultralytics YOLO"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()