"""
FastAPI Server for Box Detection and Label Processing

Provides REST API endpoint for detecting boxes, extracting labels,
and reading barcodes/QR codes/text from shipping labels.
"""

import os
import io
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

from src.rf_api_detector import RoboflowAPIDetector
from src.box_detector import BoxDetector
from src.label_processor import LabelProcessor
from src.geometry_utils import match_labels_to_boxes
from src.api_logger import APILogger

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


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "YOLO Box Counting & Label Processing API",
        "version": "2.0.0",
        "endpoints": {
            "detect": "/detect (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "detector": "initialized" if detector is not None else "not initialized",
        "label_processor": "initialized" if label_processor is not None else "not initialized",
        "api_logger": "initialized" if api_logger is not None else "not initialized"
    }


@app.get("/stats")
async def get_api_stats(days: int = Query(30, ge=1, le=365, description="Number of days to include")):
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
    file: UploadFile = File(..., description="Image file to process"),
    process_labels: bool = Query(True, description="Process labels for barcodes/QR/text"),
    ocr_confidence: float = Query(0.5, ge=0.0, le=1.0, description="OCR confidence threshold")
):
    """
    Detect boxes and labels in an image, extract barcode/QR/text from labels.
    
    Pipeline:
    1. Run YOLO detection to find boxes and labels
    2. Match labels to boxes (check if label center is inside box)
    3. For each label, try barcode/QR detection, fallback to OCR
    4. Return box-centric response with orphan labels flagged
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        process_labels: Whether to process labels for data extraction
        ocr_confidence: Minimum confidence for OCR text
        
    Returns:
        DetectionResponse with boxes, orphan labels, and summary
    """
    import time
    start_time = time.time()
    error_message = None
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
        
        # Prepare response
        response_dict = {
            "boxes": [box.dict() for box in response_boxes],
            "orphan_labels": [label.dict() for label in response_orphans],
            "summary": summary.dict()
        }
        
        # Log API call
        processing_time_ms = (time.time() - start_time) * 1000
        if api_logger:
            try:
                api_logger.log_api_call(
                    image_data=contents,
                    response=response_dict,
                    processing_time_ms=processing_time_ms,
                    error=error_message
                )
            except Exception as log_err:
                logger.warning(f"Failed to log API call: {log_err}")
        
        return DetectionResponse(
            boxes=response_boxes,
            orphan_labels=response_orphans,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing image: {e}", exc_info=True)
        
        # Log failed call
        processing_time_ms = (time.time() - start_time) * 1000
        if api_logger and 'contents' in locals():
            try:
                api_logger.log_api_call(
                    image_data=contents,
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
