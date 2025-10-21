"""
FastAPI Server for Box Detection and Label Processing

Provides REST API endpoint for detecting boxes, extracting labels,
and reading barcodes/QR codes/text from shipping labels.
"""

import os
import io
import cv2
import base64
import secrets
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

from .rf_api_detector import RoboflowAPIDetector
from .box_detector import BoxDetector
from .label_processor import LabelProcessor
from .geometry_utils import match_labels_to_boxes
from .api_logger import APILogger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLO Box Counting & Label Processing API",
    description="Detect boxes, read labels with barcodes/QR codes/text",
    version="2.0.0"
)

# Basic Authentication Setup
security = HTTPBasic()

# Get authentication credentials from environment variables
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "secure123")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify basic authentication credentials.
    """
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Global instances (initialized on startup)
detector = None
label_processor = None
api_logger = None


# Pydantic Models for API Response
class DetectedData(BaseModel):
    """Data extracted from labels within a box."""
    barcodes: List[str] = Field(default_factory=list, description="Detected barcodes (CODE128, EAN, etc.)")
    qrcodes: List[str] = Field(default_factory=list, description="Detected QR codes")
    text: Optional[str] = Field(None, description="OCR text if no codes found")


class BoxDetection(BaseModel):
    """Single box detection with contained labels."""
    bbox: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    label: str = Field(..., description="Label confidence if present, or 'false'")
    detected: Optional[DetectedData] = Field(None, description="Data extracted from labels")


class OrphanLabel(BaseModel):
    """Label detected outside of any box."""
    bbox: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    detected: Optional[DetectedData] = Field(None, description="Data extracted from label")


class DetectionSummary(BaseModel):
    """Summary statistics of the detection."""
    total_boxes: int = Field(..., description="Total number of boxes detected")
    boxes_with_labels: int = Field(..., description="Boxes containing labels")
    orphan_labels: int = Field(..., description="Labels not inside any box")
    barcodes_found: int = Field(..., description="Total barcodes detected")
    qrcodes_found: int = Field(..., description="Total QR codes detected")
    ocr_used: int = Field(..., description="Number of labels processed with OCR")


class DetectionResponse(BaseModel):
    """Complete API response."""
    boxes: List[BoxDetection] = Field(..., description="Detected boxes with labels")
    orphan_labels: List[OrphanLabel] = Field(default_factory=list, description="Labels outside boxes")
    summary: DetectionSummary = Field(..., description="Detection statistics")
    annotated_image: Optional[str] = Field(None, description="Base64-encoded image with bounding boxes and labels drawn")


@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup."""
    global detector, label_processor, api_logger
    
    logger.info("Initializing detection models...")
    
    # Initialize detector (use Roboflow API by default)
    backend = os.getenv("DETECTION_BACKEND", "roboflow")
    
    if backend == "roboflow":
        api_key = os.getenv("ROBOFLOW_API_KEY", "")
        api_endpoint = os.getenv("ROBOFLOW_API_ENDPOINT", "https://detect.roboflow.com/shoeboxes-rwv5h/2")
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        iou = float(os.getenv("IOU_THRESHOLD", "0.45"))
        
        detector = RoboflowAPIDetector(
            api_url=api_endpoint,
            api_key=api_key,
            confidence=confidence,
            overlap=iou
        )
        logger.info(f"Initialized Roboflow API detector: {api_endpoint}")
    else:
        model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        iou = float(os.getenv("IOU_THRESHOLD", "0.45"))
        
        detector = BoxDetector(model_path, confidence, iou)
        logger.info(f"Initialized local YOLO detector: {model_path}")
    
    # Initialize label processor
    # EasyOCR can trigger native code that may crash on some hardware/CPUs
    # so enable it only when explicitly requested via ENABLE_OCR env var.
    enable_ocr = os.getenv("ENABLE_OCR", "false").lower() in ("1", "true", "yes")
    label_processor = LabelProcessor(use_ocr=enable_ocr, ocr_languages=['en'], use_gpu=False)
    logger.info(f"Label processor initialized (ocr_enabled={enable_ocr})")
    
    # Initialize API logger
    api_logger = APILogger()
    logger.info("API logger initialized")


def create_annotated_image(
    image: np.ndarray, 
    boxes_data: List[Dict], 
    labels_data: List[Dict], 
    box_to_labels: Dict[int, List[int]], 
    orphan_label_indices: List[int],
    label_data_results: Dict[int, Dict] = None
) -> str:
    """
    Create an annotated image with bounding boxes and labels drawn.
    
    Args:
        image: Original RGB image
        boxes_data: List of detected boxes
        labels_data: List of detected labels
        box_to_labels: Mapping of box indices to label indices
        orphan_label_indices: Indices of labels not inside any box
        label_data_results: Results from label processing (barcodes, QR codes, text)
    
    Returns:
        Base64-encoded JPEG image with annotations
    """
    # Convert RGB to BGR for OpenCV
    annotated_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # Define colors
    box_color = (0, 255, 0)      # Green for boxes
    label_color = (255, 0, 0)    # Blue for labels in boxes
    orphan_color = (0, 0, 255)   # Red for orphan labels
    text_color = (255, 255, 255) # White for text
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_thickness = 1
    
    # Draw boxes
    for box_idx, box in enumerate(boxes_data):
        x1, y1, x2, y2 = box['bbox']
        
        # Draw box rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)
        
        # Prepare box label text
        label_indices = box_to_labels.get(box_idx, [])
        box_text = f"Box {box_idx + 1} ({box['confidence']:.2f})"
        
        if label_indices:
            box_text += f" +{len(label_indices)} label(s)"
            
            # Add detected data info if available
            if label_data_results:
                all_barcodes = []
                all_qrcodes = []
                has_text = False
                
                for label_idx in label_indices:
                    if label_idx in label_data_results:
                        result = label_data_results[label_idx]
                        all_barcodes.extend(result.get('barcodes', []))
                        all_qrcodes.extend(result.get('qrcodes', []))
                        if result.get('text'):
                            has_text = True
                
                if all_barcodes:
                    box_text += f" [{len(all_barcodes)} barcode(s)]"
                if all_qrcodes:
                    box_text += f" [{len(all_qrcodes)} QR]"
                if has_text:
                    box_text += " [text]"
        
        # Draw box label
        text_size = cv2.getTextSize(box_text, font, font_scale, text_thickness)[0]
        cv2.rectangle(annotated_img, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0], y1), box_color, -1)
        cv2.putText(annotated_img, box_text, (x1, y1 - 5), 
                   font, font_scale, text_color, text_thickness)
    
    # Draw labels inside boxes
    for box_idx, label_indices in box_to_labels.items():
        for label_idx in label_indices:
            if label_idx < len(labels_data):
                label = labels_data[label_idx]
                x1, y1, x2, y2 = label['bbox']
                
                # Draw label rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), label_color, thickness)
                
                # Prepare label text
                label_text = f"Label {label_idx + 1} ({label['confidence']:.2f})"
                
                # Add detected data if available
                if label_data_results and label_idx in label_data_results:
                    result = label_data_results[label_idx]
                    data_parts = []
                    
                    if result.get('barcodes'):
                        data_parts.append(f"BC:{result['barcodes'][0][:8]}...")
                    if result.get('qrcodes'):
                        data_parts.append(f"QR:{result['qrcodes'][0][:8]}...")
                    if result.get('text') and not result.get('barcodes') and not result.get('qrcodes'):
                        data_parts.append(f"TXT:{result['text'][:8]}...")
                    
                    if data_parts:
                        label_text += f" [{', '.join(data_parts)}]"
                
                # Draw label text
                text_size = cv2.getTextSize(label_text, font, font_scale, text_thickness)[0]
                cv2.rectangle(annotated_img, (x1, y2), 
                             (x1 + text_size[0], y2 + text_size[1] + 10), label_color, -1)
                cv2.putText(annotated_img, label_text, (x1, y2 + text_size[1] + 5), 
                           font, font_scale, text_color, text_thickness)
    
    # Draw orphan labels
    for label_idx in orphan_label_indices:
        if label_idx < len(labels_data):
            label = labels_data[label_idx]
            x1, y1, x2, y2 = label['bbox']
            
            # Draw orphan label rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), orphan_color, thickness)
            
            # Prepare orphan label text
            orphan_text = f"Orphan {label_idx + 1} ({label['confidence']:.2f})"
            
            # Add detected data if available
            if label_data_results and label_idx in label_data_results:
                result = label_data_results[label_idx]
                data_parts = []
                
                if result.get('barcodes'):
                    data_parts.append(f"BC:{result['barcodes'][0][:8]}...")
                if result.get('qrcodes'):
                    data_parts.append(f"QR:{result['qrcodes'][0][:8]}...")
                if result.get('text') and not result.get('barcodes') and not result.get('qrcodes'):
                    data_parts.append(f"TXT:{result['text'][:8]}...")
                
                if data_parts:
                    orphan_text += f" [{', '.join(data_parts)}]"
            
            # Draw orphan label text
            text_size = cv2.getTextSize(orphan_text, font, font_scale, text_thickness)[0]
            cv2.rectangle(annotated_img, (x1, y2), 
                         (x1 + text_size[0], y2 + text_size[1] + 10), orphan_color, -1)
            cv2.putText(annotated_img, orphan_text, (x1, y2 + text_size[1] + 5), 
                       font, font_scale, text_color, text_thickness)
    
    # Add legend
    legend_y = 30
    cv2.putText(annotated_img, "Legend:", (10, legend_y), font, font_scale, text_color, text_thickness)
    legend_y += 25
    cv2.putText(annotated_img, "Green: Boxes", (10, legend_y), font, font_scale, box_color, text_thickness)
    legend_y += 25
    cv2.putText(annotated_img, "Blue: Labels in boxes", (10, legend_y), font, font_scale, label_color, text_thickness)
    legend_y += 25
    cv2.putText(annotated_img, "Red: Orphan labels", (10, legend_y), font, font_scale, orphan_color, text_thickness)
    
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image.
    
    Args:
        base64_string: Base64 encoded image (with or without data URL prefix)
        
    Returns:
        RGB image as numpy array
        
    Raises:
        ValueError: If the base64 string is invalid or cannot be decoded
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image with OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from base64 data")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
        
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")


@app.get("/")
async def root(username: str = Depends(authenticate)):
    """API root endpoint. Requires authentication."""
    return {
        "name": "YOLO Box Counting & Label Processing API",
        "version": "2.0.0",
        "authenticated_user": username,
        "endpoints": {
            "detect": "/detect (POST) - Accepts file upload OR base64_image form field",
            "health": "/health (GET)",
            "stats": "/stats (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - public access."""
    return {
        "status": "healthy",
        "detector": "initialized" if detector is not None else "not initialized",
        "label_processor": "initialized" if label_processor is not None else "not initialized",
        "api_logger": "initialized" if api_logger is not None else "not initialized"
    }


@app.get("/auth-info")
async def auth_info():
    """Authentication information - public access."""
    return {
        "message": "This API requires Basic Authentication",
        "authentication_type": "HTTP Basic Auth",
        "protected_endpoints": ["/", "/detect", "/stats"],
        "public_endpoints": ["/health", "/auth-info"],
        "note": "Use your username and password in the Authorization header"
    }


@app.get("/stats")
async def get_api_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    username: str = Depends(authenticate)
):
    """
    Get API usage statistics.
    
    Args:
        days: Number of days to include in statistics
        
    Returns:
        Statistics dictionary with call counts, processing times, detection metrics
    """
    if api_logger is None:
        raise HTTPException(status_code=503, detail="API logger not initialized")
    
    try:
        stats = api_logger.get_stats(days=days)
        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.post("/detect", response_model=DetectionResponse)
async def detect_boxes_and_labels(
    file: Optional[UploadFile] = File(None, description="Image file to process (alternative to base64_image)"),
    base64_image: Optional[str] = Form(None, description="Base64 encoded image data (alternative to file)"),
    process_labels: bool = Form(True, description="Process labels for barcodes/QR/text"),
    include_annotated_image: bool = Form(True, description="Include annotated image with bounding boxes"),
    ocr_confidence: float = Form(0.5, ge=0.0, le=1.0, description="OCR confidence threshold"),
    username: str = Depends(authenticate)
):
    """
    Detect boxes and labels in an image, extract barcode/QR/text from labels.
    
    Supports both file upload and base64 image input via form data. Provide either 'file' or 'base64_image', not both.
    
    Pipeline:
    1. Run YOLO detection to find boxes and labels
    2. Match labels to boxes (check if label center is inside box)
    3. For each label, try barcode/QR detection, fallback to OCR
    4. Return box-centric response with orphan labels flagged
    5. Optionally include annotated image with bounding boxes drawn
    
    Args:
        file: Image file upload (JPEG, PNG, etc.) - mutually exclusive with base64_image
        base64_image: Base64 encoded image data (form field) - mutually exclusive with file
        process_labels: Whether to process labels for data extraction
        include_annotated_image: Whether to include annotated image with bounding boxes
        ocr_confidence: Minimum confidence for OCR text
        
    Returns:
        DetectionResponse with boxes, orphan labels, summary, and optional annotated image
    """
    import time
    start_time = time.time()
    error_message = None
    
    try:
        # Validate input - must have either file or base64_image, but not both
        if not file and not base64_image:
            raise HTTPException(
                status_code=400, 
                detail="Either 'file' (multipart upload) or 'base64_image' (form field) must be provided"
            )
        
        if file and base64_image:
            raise HTTPException(
                status_code=400, 
                detail="Provide either 'file' or 'base64_image', not both"
            )
        
        # Process image based on input type
        if file:
            # Handle file upload
            logger.info("Processing uploaded file...")
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        else:
            # Handle base64 input
            logger.info("Processing base64 image...")
            try:
                image_rgb = decode_base64_image(base64_image)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(ve)}")
        
        # Run YOLO detection
        logger.info("Running YOLO detection...")
        detection_results = detector.detect_boxes(image_rgb, return_crops=process_labels)
        
        # Separate boxes and labels
        boxes_data = []
        labels_data = []
        
        for idx, (bbox, conf, cls) in enumerate(zip(
            detection_results['boxes'],
            detection_results['confidences'],
            detection_results.get('classes', [''] * len(detection_results['boxes']))
        )):
            item = {
                'bbox': bbox,
                'confidence': conf,
                'class': cls.lower() if cls else 'unknown',
                'index': idx
            }
            
            if cls.lower() == 'box':
                boxes_data.append(item)
            elif cls.lower() == 'label':
                labels_data.append(item)
            else:
                # Unknown class, treat as box
                boxes_data.append(item)
        
        logger.info(f"Detected {len(boxes_data)} boxes and {len(labels_data)} labels")
        
        # Match labels to boxes
        box_to_labels, orphan_label_indices = match_labels_to_boxes(boxes_data, labels_data)
        
        # Process labels if requested
        label_data_results = {}
        ocr_count = 0
        total_barcodes = 0
        total_qrcodes = 0
        
        if process_labels and detection_results.get('crops'):
            logger.info("Processing labels for barcode/QR/text extraction...")
            
            for label_idx, label in enumerate(labels_data):
                crop_idx = label['index']
                if crop_idx < len(detection_results['crops']):
                    label_crop = detection_results['crops'][crop_idx]
                    
                    # Process label through pipeline
                    result = label_processor.process_label(
                        label_crop,
                        try_codes_first=True,
                        ocr_fallback=True,
                        ocr_confidence=ocr_confidence
                    )
                    
                    label_data_results[label_idx] = result
                    
                    # Update counters
                    total_barcodes += len(result['barcodes'])
                    total_qrcodes += len(result['qrcodes'])
                    if result['detection_method'] in ['ocr', 'both']:
                        ocr_count += 1
        
        # Build response
        response_boxes = []
        boxes_with_labels_count = 0
        
        for box_idx, box in enumerate(boxes_data):
            label_indices = box_to_labels.get(box_idx, [])
            
            # Determine label status
            if label_indices:
                boxes_with_labels_count += 1
                # Use highest confidence label
                label_confidences = [labels_data[idx]['confidence'] for idx in label_indices]
                label_status = f"{max(label_confidences):.2f}"
            else:
                label_status = "false"
            
            # Aggregate detected data from all labels in this box
            detected_data = None
            if label_indices and process_labels:
                all_barcodes = []
                all_qrcodes = []
                all_text = []
                
                for label_idx in label_indices:
                    if label_idx in label_data_results:
                        result = label_data_results[label_idx]
                        all_barcodes.extend(result['barcodes'])
                        all_qrcodes.extend(result['qrcodes'])
                        if result['text']:
                            all_text.append(result['text'])
                
                if all_barcodes or all_qrcodes or all_text:
                    detected_data = DetectedData(
                        barcodes=all_barcodes,
                        qrcodes=all_qrcodes,
                        text='\n'.join(all_text) if all_text else None
                    )
            
            response_boxes.append(BoxDetection(
                bbox=box['bbox'],
                confidence=box['confidence'],
                label=label_status,
                detected=detected_data
            ))
        
        # Build orphan labels
        response_orphans = []
        for label_idx in orphan_label_indices:
            label = labels_data[label_idx]
            
            detected_data = None
            if process_labels and label_idx in label_data_results:
                result = label_data_results[label_idx]
                if result['barcodes'] or result['qrcodes'] or result['text']:
                    detected_data = DetectedData(
                        barcodes=result['barcodes'],
                        qrcodes=result['qrcodes'],
                        text=result['text']
                    )
            
            response_orphans.append(OrphanLabel(
                bbox=label['bbox'],
                confidence=label['confidence'],
                detected=detected_data
            ))
        
        # Build summary
        summary = DetectionSummary(
            total_boxes=len(boxes_data),
            boxes_with_labels=boxes_with_labels_count,
            orphan_labels=len(orphan_label_indices),
            barcodes_found=total_barcodes,
            qrcodes_found=total_qrcodes,
            ocr_used=ocr_count
        )
        
        logger.info(f"Detection complete: {summary.dict()}")
        
        # Generate annotated image if requested
        annotated_image_b64 = None
        if include_annotated_image:
            try:
                annotated_image_b64 = create_annotated_image(
                    image_rgb, 
                    boxes_data, 
                    labels_data, 
                    box_to_labels, 
                    orphan_label_indices,
                    label_data_results if process_labels else None
                )
                logger.info("Annotated image generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate annotated image: {e}")
                annotated_image_b64 = None
        
        # Prepare response
        response_dict = {
            "boxes": [box.dict() for box in response_boxes],
            "orphan_labels": [label.dict() for label in response_orphans],
            "summary": summary.dict(),
            "annotated_image": annotated_image_b64
        }
        
        # Log API call
        processing_time_ms = (time.time() - start_time) * 1000
        if api_logger:
            try:
                # Prepare image data for logging based on input type
                if file:
                    log_image_data = contents
                else:
                    # For base64, log only first 1000 chars to avoid huge logs
                    log_image_data = base64_image[:1000].encode('utf-8')
                
                api_logger.log_api_call(
                    image_data=log_image_data,
                    response=response_dict,
                    processing_time_ms=processing_time_ms,
                    error=error_message
                )
            except Exception as log_err:
                logger.warning(f"Failed to log API call: {log_err}")
        
        return DetectionResponse(
            boxes=response_boxes,
            orphan_labels=response_orphans,
            summary=summary,
            annotated_image=annotated_image_b64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing image: {e}", exc_info=True)
        
        # Log failed call
        processing_time_ms = (time.time() - start_time) * 1000
        if api_logger:
            try:
                # Prepare image data for logging based on available input
                log_image_data = b''
                if 'contents' in locals():
                    log_image_data = contents
                elif base64_image:
                    log_image_data = base64_image[:1000].encode('utf-8')
                
                api_logger.log_api_call(
                    image_data=log_image_data,
                    response={},
                    processing_time_ms=processing_time_ms,
                    error=error_message
                )
            except Exception as log_err:
                logger.warning(f"Failed to log failed API call: {log_err}")
        
        raise HTTPException(status_code=500, detail=f"Error processing image: {error_message}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
