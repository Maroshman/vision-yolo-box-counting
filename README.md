# 📦 YOLO Box Counting Engine

A comprehensive computer vision system for detecting and counting boxes using YOLO v8 and Roboflow integration.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-web%20app-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **🎯 YOLO v8 Integration**: State-of-the-art object detection with multiple model sizes
- **🌐 Roboflow API Integration**: Cloud-hosted inference for boxes and labels
- **📦 Box-Centric Detection**: Validates labels are inside boxes, flags orphans
- **📊 Barcode/QR Reading**: Automatic extraction of codes from shipping labels
- **📝 OCR Fallback**: Text extraction when barcodes not present
- **🖥️ Web Interface**: Beautiful Streamlit-based user interface
- **📊 Batch Processing**: Process hundreds of images efficiently
- ** Training Pipeline**: Complete model training and evaluation workflow
- **🐳 Docker Support**: Easy deployment with containerization
- **💻 CLI Interface**: Command-line tools for automation
- **� REST API**: FastAPI endpoint for programmatic integration

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- Modern web browser

## 🛠️ Quick Start

### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Yolo-boxCounting

# Complete setup (install dependencies + create directories)
make setup

# Start the web application
make run-app
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/images data/annotations results models

# Copy environment file
cp .env.example .env

# Edit .env with your Roboflow API key (optional if using local model)
nano .env

# Run the application
streamlit run app.py
```

### Option 3: Docker

```bash
# Build and run with Docker
docker-compose up --build

# Or use individual Docker commands
docker build -t yolo-box-counting .
docker run -p 8501:8501 yolo-box-counting
```

## 📁 Project Structure

```
Yolo-boxCounting/
├── 📱 app.py                    # Streamlit web application
├── 💻 cli.py                    # Command-line interface
├── 📚 examples.py               # Usage examples
├── � run_api.py                # FastAPI server runner (production)
├── 🔧 run_api_dev.py            # FastAPI server runner (development)
├── �🐳 Dockerfile               # Docker configuration
├── 📝 Makefile                 # Build automation
├── 📊 config.yaml              # Configuration settings
├── 📦 requirements.txt         # Python dependencies
├── 
├── 📂 src/                     # Core source code
│   ├── 🎯 box_detector.py      # YOLO detection engine
│   ├── 🌐 roboflow_client.py   # Roboflow integration
│   ├── 🚀 api_server.py        # FastAPI server implementation
│   ├── 📊 label_processor.py   # Barcode/QR/OCR processing
│   ├── 📐 geometry_utils.py    # Geometric calculations
│   ├── 📋 api_logger.py        # API usage logging
│   └── 🛠️ utils.py             # Utility functions
├── 
├── 📂 notebooks/               # Jupyter training notebooks
│   └── 🏋️ yolo_box_training.ipynb
├── 
├── 📂 data/                    # Input data
│   ├── 🖼️ images/              # Training/test images
│   └── 📝 annotations/         # Label files
├── 
├── 📂 models/                  # Trained models
├── 📂 results/                 # Detection outputs
└── 📂 datasets/                # Downloaded datasets
```

## 🎮 Usage Guide

### 🌐 REST API (Recommended for Production)

Start the FastAPI server:

```bash
# Production mode
python run_api.py

# Development mode with hot reload
python run_api_dev.py

# Using Make commands
make run-api      # Production
make run-api-dev  # Development

# Direct uvicorn (alternative)
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

**Example API Request:**

```bash
# Detect boxes and process labels with annotated image
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "process_labels=true" \
  -F "include_annotated_image=true" \
  -F "ocr_confidence=0.5"
```

**Example Response:**

```json
{
  "boxes": [
    {
      "bbox": [100, 150, 400, 450],
      "confidence": 0.95,
      "label": "0.93",
      "detected": {
        "barcodes": ["98842510", "98842513"],
        "qrcodes": [],
        "text": "Handling Unit Number\n24105976"
      }
    },
    {
      "bbox": [450, 150, 750, 450],
      "confidence": 0.91,
      "label": "false",
      "detected": null
    }
  ],
  "orphan_labels": [],
  "summary": {
    "total_boxes": 4,
    "boxes_with_labels": 3,
    "orphan_labels": 0,
    "barcodes_found": 6,
    "qrcodes_found": 0,
    "ocr_used": 2
  }
}
```

**Python Client Example:**

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("image.jpg", "rb")}
params = {"process_labels": True, "ocr_confidence": 0.5}

response = requests.post(url, files=files, params=params)
data = response.json()

print(f"Found {data['summary']['total_boxes']} boxes")
for box in data['boxes']:
    if box['detected']:
        print(f"Box has {len(box['detected']['barcodes'])} barcodes")
```

### �🖥️ Web Interface

1. **Launch the app**: Run `streamlit run app.py` or `make run-app`
2. **Upload images**: Use the file uploader in the "Single Image" tab
3. **Adjust settings**: Choose Detection Backend (Roboflow API or Local YOLO) and configure thresholds in the sidebar
4. **View results**: See detection visualizations and box counts
5. **Batch processing**: Process multiple images in the "Batch Processing" tab
6. **Analytics**: View comprehensive statistics in the "Analytics" tab

### 💻 Command Line Interface

```bash
# Detect boxes using Roboflow API (default)
python cli.py detect image.jpg -o results --backend roboflow --rf-api-key "$ROBOFLOW_API_KEY"

# Count boxes in multiple images using Roboflow API
python cli.py count data/images --format json --backend roboflow --rf-api-key "$ROBOFLOW_API_KEY"

# Detect boxes with local model
python cli.py detect image.jpg -o results --backend local -m yolov8n.pt

# Train a custom model
python cli.py train datasets/data.yaml --epochs 100 --batch-size 16

# Evaluate model performance
python cli.py evaluate models/best.pt datasets/data.yaml
```

### 🐍 Python API

```python
from src.box_detector import BoxDetector

# Initialize detector
detector = BoxDetector("yolov8n.pt", confidence=0.5)

# Detect boxes in an image
results = detector.detect_boxes("image.jpg", return_details=True)
print(f"Detected {results['count']} boxes")

# Batch process multiple images
results = detector.batch_process("data/images", "results")
```

### 📓 Jupyter Notebooks

Open the training notebook for a complete walkthrough:

```bash
# Start Jupyter Lab
jupyter lab notebooks/yolo_box_training.ipynb

# Or use the make command
make notebook
```

## 🏋️ Training Your Own Model

### 1. Prepare Your Dataset

**Option A: Use Roboflow (Recommended)**
1. Create an account at [Roboflow](https://roboflow.com)
2. Create a new object detection project
3. Upload and annotate your box images
4. Generate and download dataset in YOLOv8 format

**Option B: Manual Preparation**
1. Organize images in `data/images/`
2. Create YOLO format annotations in `data/annotations/`
3. Create a `data.yaml` configuration file

### 2. Configure Roboflow Integration

```bash
# Set your Roboflow credentials
export ROBOFLOW_API_KEY="your_api_key_here"
export ROBOFLOW_WORKSPACE="your_workspace"
export ROBOFLOW_PROJECT="box-detection"
```

### 3. Download Dataset

```python
from src.roboflow_client import RoboflowClient

client = RoboflowClient("your_api_key")
dataset_path = client.download_dataset("workspace", "project", 1)
```

### 4. Train the Model

```bash
# Using CLI
python cli.py train datasets/data.yaml --epochs 100

# Using Make
make train

# Using Python
from src.box_detector import BoxDetector
detector = BoxDetector("yolov8n.pt")
detector.train_custom_model("datasets/data.yaml", epochs=100)
```

### 5. Evaluate Performance

```bash
# Evaluate model
python cli.py evaluate models/best.pt datasets/data.yaml

# Run benchmark
make benchmark
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Roboflow Hosted Inference (optional)
ROBOFLOW_API_KEY=your_roboflow_api_key_here
ROBOFLOW_API_ENDPOINT=https://detect.roboflow.com/shoeboxes-rwv5h/2
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=box-detection

# Detection Backend
DETECTION_BACKEND=roboflow  # roboflow or local

# Local Model Configuration (used when DETECTION_BACKEND=local)
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Application Settings
DEBUG=False
MAX_IMAGE_SIZE=1280
BATCH_SIZE=16
```

### Model Configuration (config.yaml)

```yaml
model:
  name: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  confidence: 0.5
  iou_threshold: 0.45

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01

detection:
  save_crops: true
  save_txt: true
  max_detections: 1000
```

## 🚀 Deployment

### 🐳 Docker Deployment

```bash
# Production deployment with Docker Compose
docker-compose up -d

# Custom Docker build
docker build -t yolo-box-counting .
docker run -p 8501:8501 -v $(pwd)/data:/app/data yolo-box-counting
```

### ☁️ Cloud Deployment

**Streamlit Cloud:**
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

**AWS/GCP/Azure:**
1. Use the provided Dockerfile
2. Deploy to container service
3. Configure environment variables

### 📦 Package Creation

```bash
# Create deployment package
make deploy

# This creates a zip file with all necessary components
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_box_detector.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Check system info
make info
```

### Performance Optimization

```bash
# Run performance benchmark
make benchmark

# Results show inference time and approximate FPS
```

## 📊 Model Performance

| Model | Size | Speed (CPU) | Speed (GPU) | mAP@0.5 |
|-------|------|-------------|-------------|---------|
| YOLOv8n | 6.2MB | ~50ms | ~2ms | 0.85+ |
| YOLOv8s | 21.5MB | ~80ms | ~3ms | 0.87+ |
| YOLOv8m | 49.7MB | ~150ms | ~5ms | 0.89+ |
| YOLOv8l | 83.7MB | ~250ms | ~8ms | 0.91+ |
| YOLOv8x | 136.7MB | ~400ms | ~12ms | 0.92+ |

*Performance may vary based on image size and hardware*

## 🔬 Advanced Features

### Custom Loss Functions
- Implement custom loss functions for specific box types
- Support for weighted loss based on box size

### Multi-Class Detection
- Extend to detect different types of boxes
- Support for box classification and counting by type

### Integration APIs
- RESTful API endpoints for external integration
- Webhook support for real-time notifications

### Batch Processing Optimization
- Parallel processing for large image sets
- Memory-efficient processing for resource-constrained environments

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size in config.yaml
batch_size: 8  # Instead of 16
```

**Roboflow API Errors:**
```bash
# Check your API key and project settings
python -c "from src.roboflow_client import RoboflowClient; client = RoboflowClient('your_key'); print(client.rf.workspace())"
```

**Streamlit Issues:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502
```

### Performance Tips

1. **Use GPU acceleration** when available
2. **Optimize image sizes** - resize large images before processing
3. **Adjust confidence thresholds** based on your use case
4. **Use appropriate model size** - balance speed vs accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the amazing YOLO implementation
- [Roboflow](https://roboflow.com/) for dataset management tools
- [Streamlit](https://streamlit.io/) for the web framework
- The computer vision community for continuous innovation

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 📚 Documentation: [Full docs](https://docs.example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/example/issues)

---

**Built with ❤️ for the computer vision community**